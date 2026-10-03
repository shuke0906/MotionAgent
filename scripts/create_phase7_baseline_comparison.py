"""External controlled experiment; existing production modules are read-only."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor" / "GENMO"
DEMO_PROMPT = "A person walks forward slowly, waves their right hand three times, turns to the left, and then sits down."


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def protected_sources():
    paths = list((ROOT / "motion_agent").rglob("*.py"))
    paths += list((VENDOR / "gem").rglob("*.py"))
    paths += list((VENDOR / "configs").rglob("*.yaml"))
    return {str(path.relative_to(ROOT)): digest(path) for path in sorted(paths)}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def prepare_native_request(output, prompt, frames, fps, seed):
    """Run existing planner/compiler tools and forward their outputs verbatim."""
    from motion_agent.agent.guards import validate_planner_decision
    from motion_agent.agent.planner import ScriptedPlanner
    from motion_agent.app.orchestrator import MotionAgentOrchestrator
    from motion_agent.compiler.schemas import GEMTextCondition, MotionSpecification
    from motion_agent.generation import build_generation_request
    from motion_agent.generation.schemas import GenerationOutputPolicy
    from motion_agent.state.context_builder import PlannerContextBuilder

    decisions = [
        {"action": "COMPILE_MOTION", "reason_code": "INITIAL_COMPILE", "reason_summary": "Compile the input prompt.",
         "payload": {"mode": "initial", "focus": ["semantic", "timeline", "body_part", "frequency", "gem_caption"]}},
        {"action": "GENERATE", "reason_code": "CONDITIONS_READY", "reason_summary": "Generate using the compiler output unchanged.",
         "payload": {"strategy": "normal", "scope": "full", "num_candidates": 1, "reward_targets": []}},
    ]
    planner = ScriptedPlanner(decisions)
    orchestrator = MotionAgentOrchestrator(output / "artifacts" / "planner", planner)
    initial = orchestrator.create_run(prompt)
    previous_version = initial.state_version
    initial.task.target_duration_s = frames / fps
    initial.task.fps = fps
    initial.task.total_frames = frames
    initial.state_version += 1
    initial = orchestrator.state_store.commit(initial, expected_version=previous_version,
        event_type="demo_duration_configured", summary="Only the overall demo duration is configured.")
    state = orchestrator.invoke(initial, interrupt_after=["compile_motion"])
    decision = planner.step(PlannerContextBuilder().build(state), state)
    validate_planner_decision(state, decision)
    spec = MotionSpecification.model_validate(orchestrator.artifact_store.get(state.plan.motion_spec_id)["motion_spec"])
    text = GEMTextCondition.model_validate(orchestrator.artifact_store.get(state.plan.gem_text_condition_id)["gem_text_condition"])
    native_spec, native_text = spec.model_dump(mode="json"), text.model_dump(mode="json")
    request = build_generation_request(condition=text, motion_spec=spec, fps=fps, seeds=[seed],
        strategy=decision.payload["strategy"], scope=decision.payload["scope"], num_candidates=1,
        generation_id=f"gen_current_behavior_{state.run.run_id}", text_condition_id=state.plan.gem_text_condition_id)
    request.output_policy = GenerationOutputPolicy(output_root=str(output), artifact_root=str(output / "artifacts" / "motionagent"))
    if request.motion_spec.model_dump(mode="json") != native_spec or request.text_condition.model_dump(mode="json") != native_text:
        raise RuntimeError("Compiler output changed before generation")
    return request, {"planner_backend": "ScriptedPlanner", "planner_decisions": decisions,
        "planner_run_id": state.run.run_id, "compiler_motion_spec_artifact_id": state.plan.motion_spec_id,
        "compiler_text_condition_artifact_id": state.plan.gem_text_condition_id,
        "generate_decision": decision.model_dump(mode="json"), "motion_specification": native_spec,
        "native_compiler_output": {"motion_spec": native_spec, "gem_text_condition": native_text},
        "gem_text_condition": native_text, "manual_timing_override": False,
        "compiler_output_forwarded_unchanged": True}


def run_experiment(output, prompt, frames, fps, seed):
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    before = protected_sources()
    checkpoint = VENDOR / "inputs" / "pretrained" / "gem_smpl.ckpt"
    checkpoint_hash = digest(checkpoint)
    sys.path.insert(0, str(VENDOR))
    from gem.motionagent import MotionAgentCameraContext, MotionAgentTextSegment, build_motionagent_text_data
    from scripts.demo.demo_utils import load_model, run_inference

    old_cwd = Path.cwd()
    try:
        os.chdir(VENDOR)
        model = load_model(str(checkpoint), load_text_encoder=True)
    finally:
        os.chdir(old_cwd)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    shared = {"prompt": prompt, "fps": fps, "frames": frames, "duration": frames / fps, "seed": seed,
              "checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_hash,
              "sampler": "gem_checkpoint_default", "postprocess": True,
              "static_camera": True, "all_model_weights_frozen": all(not p.requires_grad for p in model.parameters()),
              "comparison_scope": "conditioning strategy, not identity",
              "body_identity_matched": False, "constraint_satisfaction_evaluated": False,
              "manual_timing_override": False}

    # Baseline follows the vendor text demo, with no MotionAgent planning/tool calls.
    data, _ = build_motionagent_text_data(
        [MotionAgentTextSegment(caption=prompt, length=frames, name="raw_user_text")],
        MotionAgentCameraContext(width=1280, height=720, static_camera=True), seed=seed,
    )
    with torch.inference_mode():
        baseline = run_inference(model, data, static_cam=True)
    baseline_smpl = {k: v.detach().cpu() for k, v in baseline["body_params_global"].items()}
    if not all(torch.isfinite(v).all() and len(v) == frames for v in baseline_smpl.values()):
        raise ValueError("Baseline SMPL output is invalid")
    torch.save(baseline_smpl, output / "baseline_smpl.pt")
    write_json(output / "baseline_trace.json", {
        **shared, "pipeline": ["raw user text", "original GEM text adapter", "Frozen GEM", "SMPL parameters"],
        "planner_used": False, "motion_compiler_used": False, "constraint_compiler_used": False,
        "keyframe_tool_used": False, "captions": [prompt], "window_start": [0.0], "window_end": [1.0],
        "candidate_id": f"baseline_raw_text_seed{seed}", "source_motion": str(output / "baseline_smpl.pt"),
        "output_artifact_path": str(output / "baseline_smpl.pt"), "output_video": str(output / "baseline.mp4"),
        "smpl_shapes": {k: list(v.shape) for k, v in baseline_smpl.items()},
    })
    del baseline

    sys.path.insert(0, str(ROOT))
    from motion_agent.generation import generate_motion
    from motion_agent.generation.candidate_store import CandidateStore
    from motion_agent.state.schemas import GEMAdapterContext

    request, compiler_trace = prepare_native_request(output, prompt, frames, fps, seed)
    store = CandidateStore(request.output_policy.artifact_root)
    result = generate_motion(request, model=model, checkpoint_version=checkpoint_hash,
        adapter_context=GEMAdapterContext(width=1280, height=720, fps=fps, total_frames=frames), candidate_store=store)
    if result.status != "success" or result.inference_calls != 1:
        raise RuntimeError(f"MotionAgent comparison generation failed: {result.model_dump(mode='json')}")
    candidate = result.candidates[0]
    persisted = candidate.metadata.generation_request
    if (persisted.motion_spec.model_dump(mode="json") != compiler_trace["native_compiler_output"]["motion_spec"]
            or persisted.text_condition.model_dump(mode="json") != compiler_trace["native_compiler_output"]["gem_text_condition"]):
        raise RuntimeError("Persisted request differs from native compiler output")
    loaded = store.load(candidate.candidate_id)
    torch.save(loaded["body_params_global"], output / "motionagent_smpl.pt")
    write_json(output / "motionagent_trace.json", {
        **shared, "pipeline": ["user text", "ScriptedPlanner", "Motion Compiler", "MotionSpecification",
            "condition metadata", "GenerationRequest", "Frozen GEM", "CandidateStore", "SMPL parameters"],
        **compiler_trace, "condition_bundle": request.condition_bundle.model_dump(mode="json"),
        "generation_request": request.model_dump(mode="json"), "candidate_id": candidate.candidate_id,
        "source_motion": str(output / "motionagent_smpl.pt"), "candidate_metadata": candidate.metadata.model_dump(mode="json"),
        "output_artifact_path": str(output / "motionagent_smpl.pt"), "output_video": str(output / "motionagent.mp4"),
        "hard_constraints_applied": False, "keyframes_applied": False,
        "generation_satisfaction": "not evaluated", "inference_calls": result.inference_calls,
    })
    after = protected_sources()
    write_json(output / "source_integrity.json", {"unchanged": before == after, "before": before, "after": after})
    if before != after:
        raise RuntimeError("Protected production sources changed during the external experiment")
    print(json.dumps({"output_dir": str(output), "candidate_id": candidate.candidate_id, "seed": seed, "frames": frames,
        "protected_sources_unchanged": True}), flush=True)


def main():
    parser = argparse.ArgumentParser(description="Current-behavior demo with no per-action timing overrides.")
    parser.add_argument("--output_dir", type=Path, default=ROOT / "outputs" / "demo_comparison")
    parser.add_argument("--prompt", default=DEMO_PROMPT)
    parser.add_argument("--frames", type=int, default=420)
    parser.add_argument("--fps", type=int, choices=[30], default=30)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if not args.prompt.strip() or not 12 <= args.frames / args.fps <= 15:
        parser.error("Supply a natural prompt and overall duration between 12 and 15 seconds")
    run_experiment(args.output_dir, args.prompt, args.frames, args.fps, args.seed)


if __name__ == "__main__":
    main()
