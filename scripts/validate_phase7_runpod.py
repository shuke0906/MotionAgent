"""Run Phase 7 real-GEM validation on RunPod and save compact evidence."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

from motion_agent.agent.guards import validate_planner_decision
from motion_agent.agent.planner import ScriptedPlanner
from motion_agent.app.orchestrator import MotionAgentOrchestrator
from motion_agent.compiler.schemas import GEMTextCondition, MotionSpecification
from motion_agent.constraints.schemas import HardMotionCondition, VerificationSpec
from motion_agent.constraints.store import ConditionStore
from motion_agent.generation import assemble_generation_conditions, build_generation_request, generate_motion
from motion_agent.generation.candidate_store import CandidateStore
from motion_agent.generation.model_manager import GEMModelManager
from motion_agent.generation.schemas import GenerationOutputPolicy
from motion_agent.generation.schemas import GenerationRequest
from motion_agent.generation.worker import GEMGenerationWorker
from motion_agent.keyframes.schemas import KeyframeSpec, KeyframeVerificationSpec
from motion_agent.keyframes.store import KeyframeStore
from motion_agent.state.context_builder import PlannerContextBuilder
from motion_agent.state.schemas import GEMAdapterContext


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "phase7_validation"
ARTIFACTS = ROOT / "artifacts" / "phase7_validation"
CKPT = ROOT / "vendor" / "GENMO" / "inputs" / "pretrained" / "gem_smpl.ckpt"
SMPLX = ROOT / "vendor" / "GENMO" / "inputs" / "checkpoints" / "body_models" / "smplx" / "SMPLX_NEUTRAL.npz"
PROMPT = (
    "Walk forward for 2 seconds. While continuing to walk, wave your right hand twice. "
    "Then stop, turn 90 degrees to the left, and sit down."
)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compact_result(result) -> dict[str, Any]:
    return {
        "generation_id": result.generation_id,
        "status": result.status,
        "candidate_ids": [candidate.candidate_id for candidate in result.candidates],
        "seeds": [candidate.metadata.seed for candidate in result.candidates],
        "failed_candidates": [failure.model_dump(mode="json") for failure in result.failed_candidates],
        "condition_fingerprint": result.condition_fingerprint,
        "runtime_ms": result.runtime_ms,
        "cache_hits": result.cache_hits,
        "inference_calls": result.inference_calls,
    }


def validate_candidate(store: CandidateStore, candidate_id: str) -> dict[str, Any]:
    loaded = store.load(candidate_id)
    motion = loaded["motion_repr"]
    smpl = loaded["body_params_global"]
    return {
        "candidate_id": candidate_id,
        "pred_x_exists": motion is not None,
        "pred_x_shape": list(motion.shape),
        "pred_x_last_dim": int(motion.shape[-1]),
        "nan_count": int(torch.isnan(motion).sum().item()),
        "inf_count": int(torch.isinf(motion).sum().item()),
        "global_smpl_exists": bool(smpl),
        "global_smpl_keys": sorted(smpl.keys()),
        "frame_count": int(motion.shape[0]),
    }


def with_output(request, subdir: str):
    return request.model_copy(
        update={
            "output_policy": GenerationOutputPolicy(
                output_root=str(OUT / subdir),
                artifact_root=str(ARTIFACTS / subdir),
            )
        }
    )


def prepare_semantic_run(root: Path):
    """Run the existing scripted planner and compiler node, stopping before GENERATE."""
    decisions = [
        {"action": "COMPILE_MOTION", "reason_code": "INITIAL_COMPILE", "reason_summary": "Compile the composite request.",
         "payload": {"mode": "initial", "focus": ["semantic", "timeline", "body_part", "frequency", "gem_caption"]}},
        {"action": "GENERATE", "reason_code": "CONDITIONS_READY", "reason_summary": "Generate two Frozen GEM candidates.",
         "payload": {"strategy": "normal", "scope": "full", "num_candidates": 2, "reward_targets": []}},
    ]
    planner = ScriptedPlanner(decisions)
    orchestrator = MotionAgentOrchestrator(root, planner)
    initial = orchestrator.create_run(PROMPT)
    state = orchestrator.invoke(initial, interrupt_after=["compile_motion"])
    decision = planner.step(PlannerContextBuilder().build(state), state)
    validate_planner_decision(state, decision)
    spec = MotionSpecification.model_validate(orchestrator.artifact_store.get(state.plan.motion_spec_id)["motion_spec"])
    text = GEMTextCondition.model_validate(orchestrator.artifact_store.get(state.plan.gem_text_condition_id)["gem_text_condition"])
    return state, decision, spec, text, decisions


class RecordingGEM:
    """Record the actual adapter payload received at the Frozen GEM boundary."""

    def __init__(self, model, condition_store=None):
        self.model = model
        self.condition_store = condition_store
        self.receipt = None

    def predict(self, data, **kwargs):
        bundle = data["meta"][0]["condition_bundle"]
        handle = bundle["hard_motion_condition_handle"]
        mask = data.get("motion_mask_3d")
        observed = data.get("observed_motion_3d")
        forwarded = True
        if handle:
            supplied = self.condition_store.load_hard_condition(handle)
            forwarded = torch.equal(observed, supplied.values) and torch.equal(mask, supplied.mask)
        self.receipt = {
            "length": int(data["length"]),
            "captions": data["meta"][0]["multi_text_data"]["caption"],
            "window_start": data["meta"][0]["multi_text_data"]["window_start"].tolist(),
            "window_end": data["meta"][0]["multi_text_data"]["window_end"].tolist(),
            "condition_bundle": bundle,
            "motion_specification": data["meta"][0].get("motion_specification"),
            "condition_forwarded_exactly": forwarded,
            "mask_shape": list(mask.shape) if mask is not None else None,
            "mask_nonzero": int(mask.sum()) if mask is not None else 0,
        }
        return self.model.predict(data, **kwargs)


def condition_interface_evidence(request, result, store, receipt):
    """Validate construction, forwarding and persistence independently of satisfaction."""
    checks = {
        "request_schema_valid": GenerationRequest.model_validate(request.model_dump(mode="json")) == request,
        "generation_succeeded": result.status == "success" and len(result.candidates) == request.num_candidates,
        "adapter_received_condition": bool(receipt and receipt["condition_forwarded_exactly"] and receipt["mask_nonzero"] > 0),
        "mask_valid": bool(receipt and receipt["mask_shape"] == [request.total_frames, 151]),
        "condition_metadata_preserved": bool(result.candidates),
    }
    for candidate in result.candidates:
        persisted = store.load(candidate.candidate_id)["candidate"].metadata.generation_request
        checks["condition_metadata_preserved"] &= bool(
            persisted is not None
            and persisted.condition_bundle.model_dump(mode="json") == receipt["condition_bundle"]
            and persisted.motion_spec == request.motion_spec
            and persisted.text_condition == request.text_condition
        )
    return {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks}


def satisfaction_evidence(metrics, tolerance_met):
    return {**metrics, "tolerance_met": bool(tolerance_met), "status": "DEFERRED",
            "reason": "Deferred to Guided Generation; requires generation-time conditioning"}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    test_run = subprocess.run([sys.executable, "-m", "pytest", "tests", "-q"], cwd=ROOT, capture_output=True, text=True)
    write_json(OUT / "test_results.json", {"command": "python -m pytest tests -q", "exit_code": test_run.returncode,
        "stdout": test_run.stdout, "stderr": test_run.stderr})
    if test_run.returncode:
        raise RuntimeError(f"Project tests failed before GPU validation: {test_run.stdout}\n{test_run.stderr}")

    environment = {
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "vram_total_mb": int(torch.cuda.get_device_properties(0).total_memory / (1024 * 1024))
        if torch.cuda.is_available()
        else None,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "checkpoint_path": str(CKPT),
        "checkpoint_size_bytes": CKPT.stat().st_size,
        "checkpoint_sha256": sha256(CKPT),
        "smplx_path": str(SMPLX),
        "smplx_size_bytes": SMPLX.stat().st_size if SMPLX.exists() else None,
        "gem_config_hashes": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in [
                ROOT / "vendor" / "GENMO" / "configs" / "exp" / "gem_smpl.yaml",
                ROOT / "vendor" / "GENMO" / "configs" / "model" / "gem.yaml",
                ROOT / "vendor" / "GENMO" / "configs" / "demo.yaml",
            ]
            if path.exists()
        },
    }
    write_json(OUT / "environment.json", environment)

    manager = GEMModelManager(CKPT)
    model = manager.load()
    adapter_context = GEMAdapterContext(total_frames=180, checkpoint_version=environment["checkpoint_sha256"])
    state, decision, motion_spec, gem_text_condition, decisions = prepare_semantic_run(ARTIFACTS / "planner_run")
    write_json(OUT / "semantic_case" / "compiler_discrepancy.json", {
        "status": "RESOLVED", "historical_error": "caption missing required terms for segment 0: ['right', 'hand']",
        "manual_fallback_used": False, "current_path": "ScriptedPlanner -> compiler node -> MotionSpecification",
    })
    write_json(OUT / "semantic_case" / "planner_decision.json", decision.model_dump(mode="json"))
    write_json(OUT / "semantic_case" / "planner_decisions.json", {"decisions": decisions, "backend": "ScriptedPlanner"})
    write_json(OUT / "semantic_case" / "timeline.json", {"segments": [s.model_dump(mode="json") for s in motion_spec.segments]})
    write_json(OUT / "semantic_case" / "motion_specification.json", motion_spec.model_dump(mode="json"))
    write_json(OUT / "semantic_case" / "gem_text_condition.json", gem_text_condition.model_dump(mode="json"))
    (OUT / "semantic_case" / "prompt.txt").write_text(PROMPT, encoding="utf-8")

    semantic_store = CandidateStore(ARTIFACTS / "semantic_case")
    condition_store = ConditionStore(ARTIFACTS / "conditions")
    recorder = RecordingGEM(model, condition_store)
    semantic_request = with_output(
        build_generation_request(
            condition=gem_text_condition,
            motion_spec=motion_spec,
            fps=30,
            generation_id=f"gen_phase7_semantic_{state.run.run_id}",
            strategy=decision.payload["strategy"],
            scope=decision.payload["scope"],
            num_candidates=decision.payload["num_candidates"],
            seeds=[7, 8],
        ),
        "semantic_case",
    )
    semantic_result = generate_motion(
        semantic_request,
        model=recorder,
        checkpoint_version=environment["checkpoint_sha256"],
        adapter_context=adapter_context,
        candidate_store=semantic_store,
    )
    semantic_validations = [
        validate_candidate(semantic_store, candidate.candidate_id) for candidate in semantic_result.candidates
    ]
    write_json(OUT / "semantic_case" / "generation_request.json", semantic_request.model_dump(mode="json"))
    write_json(OUT / "semantic_case" / "generation_result.json", compact_result(semantic_result))
    write_json(OUT / "semantic_case" / "candidate_summaries.json", {"candidates": semantic_validations})
    write_json(OUT / "semantic_case" / "adapter_receipt.json", recorder.receipt)

    cache_repeat = generate_motion(
        semantic_request,
        model=model,
        checkpoint_version=environment["checkpoint_sha256"],
        adapter_context=adapter_context,
        candidate_store=semantic_store,
    )
    cache_miss_request = semantic_request.model_copy(update={"seeds": [9, 10]})
    cache_miss = generate_motion(
        cache_miss_request,
        model=model,
        checkpoint_version=environment["checkpoint_sha256"],
        adapter_context=adapter_context,
        candidate_store=semantic_store,
    )
    write_json(
        OUT / "cache" / "metrics.json",
        {
            "repeat_cache_hits": cache_repeat.cache_hits,
            "repeat_inference_calls": cache_repeat.inference_calls,
            "cache_miss_inference_calls": cache_miss.inference_calls,
            "candidate_fingerprint_stable": [
                a.fingerprint == b.fingerprint
                for a, b in zip(semantic_result.candidates, cache_repeat.candidates)
            ],
        },
    )

    first = semantic_store.load(semantic_result.candidates[0].candidate_id)["motion_repr"]
    hard_values = torch.zeros_like(first)
    hard_mask = torch.zeros_like(first)
    hard_values[30, 0:6] = first[30, 0:6]
    hard_mask[30, 0:6] = 1
    hard_handle = condition_store.save_hard_condition(
        HardMotionCondition(values=hard_values, mask=hard_mask, source_constraint_id="phase7_hard_condition")
    )
    hard_assembled = assemble_generation_conditions(
        text_condition_id="phase7_text",
        text_condition_payload=gem_text_condition.model_dump(mode="json"),
        total_frames=180,
        hard_condition_handles=[hard_handle],
        active_constraint_ids=["phase7_hard_condition"],
        verification_specs=[VerificationSpec(
            verification_spec_id="phase7_hard_verification", constraint_id="phase7_hard_condition",
            metric_type="feature_linf_error", target_segments=[0], frame_indices=[30],
            target={"hard_condition_handle": hard_handle}, pass_threshold=1e-4, units="normalized_features",
        )],
        store=condition_store,
    )
    hard_request = with_output(
        build_generation_request(
            condition=gem_text_condition,
            motion_spec=motion_spec,
            fps=30,
            generation_id=f"gen_phase7_hard_{state.run.run_id}",
            seed=17,
            condition_bundle=hard_assembled.bundle,
        ),
        "hard_condition",
    )
    hard_result = generate_motion(
        hard_request,
        model=recorder,
        checkpoint_version=environment["checkpoint_sha256"],
        adapter_context=adapter_context,
        candidate_store=CandidateStore(ARTIFACTS / "hard_condition"),
        condition_store=condition_store,
    )
    hard_motion = CandidateStore(ARTIFACTS / "hard_condition").load(hard_result.candidates[0].candidate_id)["motion_repr"]
    hard_error = float(torch.max(torch.abs((hard_motion - hard_values)[hard_mask.bool()])).item())
    write_json(OUT / "hard_condition" / "condition.json", {"handle": hard_handle, "mask_nonzero": int(hard_mask.sum().item())})
    write_json(OUT / "hard_condition" / "result.json", compact_result(hard_result))
    hard_interface = condition_interface_evidence(hard_request, hard_result, CandidateStore(ARTIFACTS / "hard_condition"), recorder.receipt)
    write_json(OUT / "hard_condition" / "interface.json", hard_interface)
    write_json(OUT / "hard_condition" / "adapter_receipt.json", recorder.receipt)
    write_json(OUT / "hard_condition" / "generation_request.json", hard_request.model_dump(mode="json"))
    write_json(OUT / "hard_condition" / "verification_spec.json", hard_assembled.bundle.verification_specs[0].model_dump(mode="json"))
    write_json(OUT / "hard_condition" / "metrics.json", satisfaction_evidence({"preservation_error": hard_error, "tolerance": 1e-4}, hard_error <= 1e-4))

    key_values = torch.zeros_like(first)
    key_mask = torch.zeros_like(first)
    key_values[90, 0:148] = first[90, 0:148]
    key_mask[90, 0:148] = 1
    key_handle = condition_store.save_hard_condition(
        HardMotionCondition(values=key_values, mask=key_mask, source_constraint_id="phase7_keyframe")
    )
    keyframe_store = KeyframeStore(ARTIFACTS / "keyframe_poses")
    source_smpl = semantic_store.load(semantic_result.candidates[0].candidate_id)["body_params_global"]
    pose_handle = keyframe_store.save_pose({name: value[90].clone() for name, value in source_smpl.items()}, source="candidate")
    key_verification = KeyframeVerificationSpec(
        keyframe_id="phase7_keyframe", target_frame=90, temporal_tolerance_frames=3,
        pose_handle=pose_handle, metrics=["joint_rotation_error", "joint_position_error", "root_orientation_error"],
    )
    key_spec = KeyframeSpec(
        keyframe_id="phase7_keyframe", target_segment=1, target_time_s=3.0, target_frame=90,
        temporal_tolerance_frames=3, description="Candidate whole-body state at frame 90",
        source_type="candidate", source_candidate_id=semantic_result.candidates[0].candidate_id,
        pose_handle=pose_handle, hard_condition_handle=key_handle, verification_spec=key_verification,
        metadata={"feature_target_source": "candidate_pred_x", "satisfaction_status": "DEFERRED"},
    )
    key_assembled = assemble_generation_conditions(
        text_condition_id="phase7_text",
        text_condition_payload=gem_text_condition.model_dump(mode="json"),
        total_frames=180,
        hard_condition_handles=[key_handle],
        active_keyframe_ids=["phase7_keyframe"],
        keyframe_specs=[key_spec],
        store=condition_store,
    )
    key_request = with_output(
        build_generation_request(
            condition=gem_text_condition,
            motion_spec=motion_spec,
            fps=30,
            generation_id=f"gen_phase7_keyframe_{state.run.run_id}",
            seed=18,
            condition_bundle=key_assembled.bundle,
        ),
        "keyframe",
    )
    key_result = generate_motion(
        key_request,
        model=recorder,
        checkpoint_version=environment["checkpoint_sha256"],
        adapter_context=adapter_context,
        candidate_store=CandidateStore(ARTIFACTS / "keyframe"),
        condition_store=condition_store,
    )
    key_motion = CandidateStore(ARTIFACTS / "keyframe").load(key_result.candidates[0].candidate_id)["motion_repr"]
    key_error = float(torch.max(torch.abs((key_motion - key_values)[key_mask.bool()])).item())
    key_interface = condition_interface_evidence(key_request, key_result, CandidateStore(ARTIFACTS / "keyframe"), recorder.receipt)
    key_interface["checks"]["keyframe_spec_valid"] = KeyframeSpec.model_validate(key_spec.model_dump(mode="json")) == key_spec
    key_interface["checks"]["pose_artifact_exists"] = bool(keyframe_store.load_pose(pose_handle))
    key_interface["status"] = "PASS" if all(key_interface["checks"].values()) else "FAIL"
    write_json(OUT / "keyframe" / "interface.json", key_interface)
    write_json(OUT / "keyframe" / "adapter_receipt.json", recorder.receipt)
    write_json(OUT / "keyframe" / "generation_request.json", key_request.model_dump(mode="json"))
    write_json(OUT / "keyframe" / "keyframe_spec.json", key_spec.model_dump(mode="json"))
    write_json(OUT / "keyframe" / "verification_spec.json", key_verification.model_dump(mode="json"))
    write_json(OUT / "keyframe" / "result.json", compact_result(key_result))
    write_json(OUT / "keyframe" / "metrics.json", satisfaction_evidence({"error": key_error, "tolerance": 1e-4, "metric": "feature_linf_error"}, key_error <= 1e-4))

    segment_request = with_output(
        build_generation_request(
            condition=gem_text_condition,
            motion_spec=motion_spec,
            fps=30,
            generation_id=f"gen_phase7_segment_{state.run.run_id}",
            scope="segment",
            target_segments=[1],
            previous_candidate_id=semantic_result.candidates[0].candidate_id,
            seed=19,
        ),
        "segment_inpaint",
    )
    segment_result = generate_motion(
        segment_request,
        model=recorder,
        checkpoint_version=environment["checkpoint_sha256"],
        adapter_context=adapter_context,
        candidate_store=semantic_store,
        condition_store=condition_store,
    )
    segment_motion = semantic_store.load(segment_result.candidates[0].candidate_id)["motion_repr"]
    start = round(gem_text_condition.window_start[1] * 180)
    end = round(gem_text_condition.window_end[1] * 180)
    outside_error = float(max(torch.max(torch.abs(segment_motion[:start] - first[:start])).item(), torch.max(torch.abs(segment_motion[end:] - first[end:])).item()))
    target_change = float(torch.mean(torch.abs(segment_motion[start:end, 0:126] - first[start:end, 0:126])).item())
    betas_error = float(torch.max(torch.abs(segment_motion[:, 126:136] - first[:, 126:136])).item())
    boundary = float(
        torch.linalg.vector_norm(segment_motion[start] - segment_motion[start - 1]).item()
        + torch.linalg.vector_norm(segment_motion[end] - segment_motion[end - 1]).item()
    )
    write_json(OUT / "segment_inpaint" / "request.json", segment_request.model_dump(mode="json"))
    write_json(OUT / "segment_inpaint" / "result.json", compact_result(segment_result))
    segment_interface = condition_interface_evidence(segment_request, segment_result, semantic_store, recorder.receipt)
    expected_mask = torch.ones(180, 151)
    expected_mask[start:end] = 0
    expected_mask[:, 126:136] = 1
    effective_request = segment_result.candidates[0].metadata.generation_request
    supplied = condition_store.load_hard_condition(effective_request.condition_bundle.hard_motion_condition_handle)
    segment_interface["checks"]["preservation_mask_valid"] = torch.equal(supplied.mask, expected_mask)
    segment_interface["checks"]["source_values_preserved"] = torch.equal(supplied.values[supplied.mask.bool()], first[supplied.mask.bool()])
    segment_interface["status"] = "PASS" if all(segment_interface["checks"].values()) else "FAIL"
    write_json(OUT / "segment_inpaint" / "interface.json", segment_interface)
    write_json(OUT / "segment_inpaint" / "adapter_receipt.json", recorder.receipt)
    write_json(OUT / "segment_inpaint" / "effective_request.json", effective_request.model_dump(mode="json"))
    write_json(
        OUT / "segment_inpaint" / "metrics.json",
        satisfaction_evidence({
            "outside_segment_preservation_error": outside_error,
            "target_segment_change": target_change,
            "betas_error": betas_error,
            "boundary_discontinuity": boundary,
        }, outside_error <= 1e-4 and betas_error <= 1e-4 and target_change > 0),
    )

    worker = GEMGenerationWorker(
        worker_id="phase7_real_worker",
        model_factory=lambda: model,
        checkpoint_version=environment["checkpoint_sha256"],
        adapter_context=adapter_context,
        candidate_store=semantic_store,
    )
    worker_result = worker.run(semantic_request)
    write_json(OUT / "failures" / "failure_tests.json", {
        "status": "PASS" if test_run.returncode == 0 else "FAIL", "backend": "mock failure injection",
        "partial_failure": "tests/unit/test_phase7_generation.py::Phase7GenerationTests::test_partial_failure_preserves_successful_candidates_with_typed_oom",
        "oom_injection": "torch.cuda.OutOfMemoryError injected; no real GPU OOM induced",
        "project_test_evidence": "test_results.json",
    })
    write_json(OUT / "worker_health.json", worker.health(queue_depth=0).model_dump(mode="json"))
    segments = motion_spec.segments
    checks = {
        "composite_compilation": [(s.action, s.start_frame, s.end_frame) for s in segments] == [("walk", 0, 120), ("wave", 60, 120), ("turn", 120, 150), ("sit_down", 150, 180)],
        "semantic_metadata": semantic_request.motion_spec == motion_spec and segments[1].repetition == 2
            and "right_hand" in segments[1].body_parts and segments[1].simultaneous_with == 0
            and segments[2].direction == "left" and segments[2].angle_deg == 90,
        "valid_frozen_gem_output": len(semantic_validations) == 2 and all(v["pred_x_shape"] == [180, 151] and v["nan_count"] == 0 and v["inf_count"] == 0 and v["global_smpl_exists"] for v in semantic_validations),
        "k2_candidates": semantic_result.status == "success" and len({c.candidate_id for c in semantic_result.candidates}) == 2,
        "candidate_persistence": all(semantic_store.load(c.candidate_id)["candidate"].metadata.generation_request == semantic_request for c in semantic_result.candidates),
        "cache_idempotency": cache_repeat.cache_hits == 2 and cache_repeat.inference_calls == 0 and cache_miss.inference_calls == 2
            and [c.fingerprint for c in semantic_result.candidates] == [c.fingerprint for c in cache_repeat.candidates],
        "worker_health": worker.health().available and worker.health().model_loaded and worker_result.status == "success",
        "condition_interfaces": all(e["status"] == "PASS" for e in [hard_interface, key_interface, segment_interface]),
        "failure_handling": test_run.returncode == 0,
    }
    write_json(OUT / "gate.json", {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
        "failure_handling_evidence": "Run tests/unit/test_phase7_generation.py; injected failures use mocks",
        "deferred": ["hard_condition_satisfaction", "exact_keyframe_forcing", "true_segment_inpainting", "guided_generation"]})
    write_json(
        OUT / "metrics.json",
        {
            "semantic": compact_result(semantic_result),
            "hard_condition_interface": hard_interface["status"],
            "keyframe_interface": key_interface["status"],
            "segment_interface": segment_interface["status"],
            "satisfaction_status": "DEFERRED",
            "cache_repeat_inference_calls": cache_repeat.inference_calls,
            "peak_vram_mb": max(
                [candidate.metadata.peak_gpu_memory_mb or 0 for candidate in semantic_result.candidates]
                + [0]
            ),
        },
    )


if __name__ == "__main__":
    started = time.perf_counter()
    main()
    print(f"phase7 validation completed in {time.perf_counter() - started:.2f}s")
