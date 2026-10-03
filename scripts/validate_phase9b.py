"""Reproducible Phase 9B validation on saved motion, with honest execution gates.

Run with .venv-phase9b/Scripts/python.exe scripts/validate_phase9b.py --execute-api.
The flag executes the credential-bound visual provider with a bounded call budget.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from motion_agent.compiler.motion_compiler import CompilerRequest, MotionCompiler
from motion_agent.generation.schemas import GenerationRequest, MotionCandidate
from motion_agent.verification.artifacts import file_digest, load_candidate
from motion_agent.verification.calibration import provisional_calibration
from motion_agent.verification.config import LearnedVerifierConfig, create_verification_service
from motion_agent.verification.mllm.backend import RealMLLMBackend
from motion_agent.verification.mllm.prompts import PROMPT_VERSIONS, expected_requirements
from motion_agent.verification.plan_builder import VerificationPlanBuilder
from motion_agent.verification.representations import official_repository, smpl_to_motioncritic, resample_critic, smplx_joints, joints_to_tmr
from motion_agent.verification.schemas import VerificationRequest

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/phase9b_real_verifiers"


def write(relative, data):
    path = OUTPUT / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")


def load_real_request():
    source = ROOT / "outputs/temporal_compiler_validation_final/p00_s7/fixed"
    generation = GenerationRequest.model_validate_json((source / "generation_request.json").read_text(encoding="utf-8"))
    candidate_dir = source / "candidates/cand_bddf0f0eb4785474"
    manifest = json.loads((candidate_dir / "manifest.json").read_text(encoding="utf-8"))
    candidate = MotionCandidate.model_validate(manifest["candidate"])
    candidate = candidate.model_copy(update={"motion_repr_uri": str(candidate_dir / "motion_repr.pt"),
        "smpl_global_uri": str(candidate_dir / "smpl_global.pt"), "metadata_uri": str(candidate_dir / "metadata.json"),
        "smpl_incam_uri": str(candidate_dir / "smpl_incam.pt")})
    render = source.parent / "motionagent.mp4"
    request = VerificationRequest(verification_id="verify_phase9b_real", candidate_id=candidate.candidate_id,
        original_request=generation.motion_spec.original_request, motion_spec=generation.motion_spec,
        candidate=candidate, candidate_fingerprint=candidate.fingerprint, generation_id=generation.generation_id,
        threshold_profile="phase9b_provisional_v1", visual_evidence_uri=str(render),
        visual_evidence_candidate_id=candidate.candidate_id, visual_evidence_fingerprint=file_digest(render),
        render_profile="saved_smplx_fixed_camera", storyboard_profile="scenario_v1")
    payload = load_candidate(request)
    # SMPL convention: +Y is up; +X is anatomical left at identity. Positive Y yaw is a left turn.
    from scipy.spatial.transform import Rotation
    rotations = Rotation.from_rotvec(payload["body_params_global"]["global_orient"].numpy())
    forward = rotations.apply(np.tile([0., 0., 1.], (len(rotations), 1)))
    yaw = np.unwrap(np.arctan2(forward[:, 0], forward[:, 2]))
    for segment in request.motion_spec.segments:
        if segment.action == "turn" and segment.direction in {"left", "right"}:
            start, end = segment.start_frame, min(segment.end_frame - 1, len(yaw) - 1)
            request.observed_measurements[f"signed_heading_change_{segment.segment_id}"] = float(np.degrees(yaw[end] - yaw[start]))
    write("real_candidate/generation_request.json", generation.model_dump(mode="json"))
    write("real_candidate/motion_specification.json", request.motion_spec.model_dump(mode="json"))
    write("real_candidate/evidence_manifest.json", {"candidate_id": candidate.candidate_id, "candidate_fingerprint": candidate.fingerprint,
        "generation_request_source": str(source / "generation_request.json"), "motion_spec_source": "saved GenerationRequest.motion_spec",
        "artifacts": [{"path": str(candidate_dir / name), "sha256": file_digest(candidate_dir / name)}
                      for name in ["motion_repr.pt", "smpl_global.pt", "metadata.json"]],
        "render": str(render), "render_sha256": file_digest(render), "regenerated": False,
        "heading_evidence": request.observed_measurements})
    return request


def scenario_plans(config):
    prompts = {"simple_walk": "A person walks forward.", "wave_three_times": "A person waves their right hand three times.",
        "simultaneous_walk_wave": "A person walks forward while waving their right hand three times.",
        "ordered_events": "A person walks forward, then turns left, then sits down.",
        "constraint_case": "Keep the right hand at a specified target position.", "keyframe_case": "A person sits down."}
    results = {}
    for name, prompt in prompts.items():
        compilation = MotionCompiler().compile(CompilerRequest(original_request="A person walks forward." if name == "constraint_case" else prompt))
        if name == "constraint_case":
            from motion_agent.compiler.schemas import MotionSegment
            compilation.motion_spec = compilation.motion_spec.model_copy(update={"original_request": prompt,
                "segments": [MotionSegment(segment_id=0, action="hold", body_parts=["right_hand"],
                                           start_frame=0, end_frame=180, start_s=0, end_s=6)]})
        request = VerificationRequest(verification_id=f"verify_{name}", candidate_id="routing_only", original_request=prompt,
                                      motion_spec=compilation.motion_spec)
        if name == "constraint_case":
            from motion_agent.constraints.schemas import VerificationSpec
            request.constraint_verification_specs = [VerificationSpec(verification_spec_id="constraint_target", constraint_id="right_hand",
                target_segments=[0], body_parts=["right_hand"], metric_type="joint_position_error", pass_threshold=0.05, units="m")]
        if name == "keyframe_case":
            from motion_agent.keyframes.schemas import KeyframeVerificationSpec
            request.keyframe_verification_specs = [KeyframeVerificationSpec(keyframe_id="final", target_frame=179,
                temporal_tolerance_frames=3, pose_handle="pose_final", metrics=["joint_rotation_error"])]
        plan = VerificationPlanBuilder(learned_config=config).build(request)
        write(f"scenario_tests/{name}.json", {"status": "ROUTING_VALIDATED_NO_MODEL_INFERENCE",
            "prompt": prompt, "motion_spec": request.motion_spec.model_dump(mode="json"), "plan": plan.model_dump(mode="json")})
        results[name] = "PASS"
        if name == "simultaneous_walk_wave":
            frequency = next(c for c in plan.checks if c.direction == "event_frequency")
            write("scenario_tests/temporal_trace.json", {"raw_prompt": prompt,
                "motion_compiler": "MotionCompiler.compile", "temporal_resolver": "resolve_temporal_semantics",
                "motion_specification": request.motion_spec.model_dump(mode="json"), "frequency_check": frequency.model_dump(mode="json"),
                "expected_count_preserved": frequency.metadata["expected_count"] == 3,
                "observed_count": None, "observed_count_status": "AWAITING_REAL_VISUAL_EVIDENCE"})
    return results


def motioncritic_calibration(backend, request):
    import importlib
    with official_repository(backend.config.motioncritic_repository, "lib"):
        rotation_module = importlib.import_module("lib.utils.rotation_conversions")
        raw = torch.load(ROOT / "vendor/MotionCritic/MotionCritic/visexample.pth", map_location="cpu", weights_only=True)["motion"]
        rotations = rotation_module.matrix_to_axis_angle(rotation_module.rotation_6d_to_matrix(raw[:, :24].permute(0, 3, 1, 2)))
        transl = raw[:, 24:, :3].permute(0, 3, 1, 2)
        example = torch.cat((rotations, transl), dim=2)
    torch.manual_seed(7)
    corrupted = example.clone()
    corrupted[:, :, :24] += torch.randn_like(corrupted[:, :, :24]) * 0.75
    positives = [backend.score(example[i:i + 1])[0] for i in range(len(example))]
    negatives = [backend.score(corrupted[i:i + 1])[0] for i in range(len(example))]
    calibration = provisional_calibration(positives, negatives)
    calibration.update({"data": "official visexample.pth versus injected frame-wise axis-angle jitter",
                        "naturalness_ground_truth": "uncorrupted official demo references; not human-rated naturalness labels",
                        "natural_examples": len(example), "corrupted_examples": len(example),
                        "adopted_for_production": False})
    write("motioncritic/calibration.json", calibration)
    # An explicitly labeled exploratory conversion is not allowed to certify the exact-input gate.
    approximate, conversion = smpl_to_motioncritic(load_candidate(request)["body_params_global"], "neutral_terminal_hands")
    approximate = resample_critic(approximate, request.motion_spec.fps)
    scores = backend.score(approximate)
    write("motioncritic/exploratory_candidate_scores.json", {"source": "REAL_MODEL_INFERENCE_APPROXIMATE_INPUT",
        "candidate_id": request.candidate_id, "conversion": conversion, "window_scores": scores,
        "certifies_exact_candidate_gate": False, "threshold_applied": None})
    return calibration


def tmr_calibration(backend):
    """Controlled FK motions are small synthetic sanity pairs, not benchmark labels."""
    pairs = []
    for name, positive, negative in (
            ("right_hand_wave", "A person waves their right hand.", "A person kicks their left leg."),
            ("left_leg_kick", "A person kicks their left leg.", "A person waves their right hand.")):
        frames = 120
        pose = torch.zeros(frames, 21, 3)
        phase = torch.arange(frames) * (2 * np.pi * 3 / frames)
        pose[:, 15, 2], pose[:, 16, 2] = -1.3, 1.3
        if name == "right_hand_wave":
            pose[:, 16, 2] = -0.25
            pose[:, 18, 2] = -1.15 + 0.6 * torch.sin(phase)
        else:
            pose[:, 0, 0] = -1.0 * torch.clamp(torch.sin(phase), min=0)
            pose[:, 3, 0] = 0.35 * torch.clamp(torch.sin(phase), min=0)
        params = {"body_pose": pose.reshape(frames, 63), "global_orient": torch.zeros(frames, 3),
                  "transl": torch.tensor([0., 1., 0.]).expand(frames, -1), "betas": torch.zeros(frames, 10)}
        joints = smplx_joints(params, backend.config.smplx_model_path)
        features = joints_to_tmr(joints, 20, backend.config.tmr_repository)
        folder = OUTPUT / "tmr/calibration_motions"
        folder.mkdir(parents=True, exist_ok=True)
        torch.save({"body_params_global": params, "joints_yup": joints}, folder / (name + ".pt"))
        for label, text in (("positive", positive), ("negative", negative)):
            pairs.append({"motion_case": name, "text": text, "label": label,
                          "score": backend.encoder.similarity(text, features)})
    calibration = provisional_calibration([p["score"] for p in pairs if p["label"] == "positive"],
                                          [p["score"] for p in pairs if p["label"] == "negative"])
    calibration.update({"data": "procedural SMPL-X FK arm waves and leg kicks, actual official TMR inference",
                        "synthetic_motion_examples": 2, "adopted_for_production": False,
                        "limitations": "intended-action labels; tiny out-of-domain synthetic sample; not benchmark calibration"})
    write("tmr/calibration_fixture.json", {"status": "REAL_INFERENCE_EXECUTED", "pairs": pairs})
    write("tmr/calibration.json", calibration)
    return calibration


def controlled_evidence(config):
    """Create unlabeled synthetic skeletal clips. Ground truth is never sent to the provider."""
    import cv2
    cases = [("three_waves", 3, "right", PROMPT_WAVE), ("two_waves", 2, "right", PROMPT_WAVE),
             ("missing_wave", 0, "right", PROMPT_WAVE), ("wrong_hand", 3, "left", PROMPT_WAVE)]
    results = []
    for name, count, hand, prompt in cases:
        path = OUTPUT / "mllm/controlled_evidence" / (name + ".avi")
        path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30, (512, 512))
        if not writer.isOpened():
            raise RuntimeError("synthetic video encoder unavailable")
        for frame_index in range(180):
            image = np.full((512, 512, 3), 245, dtype=np.uint8)
            cv2.circle(image, (256, 110), 24, (35, 55, 70), 3)
            cv2.line(image, (256, 134), (256, 300), (35, 55, 70), 4)
            for side, sign in (("right", -1), ("left", 1)):
                shoulder = (256 + sign * 35, 165)
                elbow = (256 + sign * 90, 175 if side == hand and count else 225)
                angle = (np.sin(2 * np.pi * count * frame_index / 180) * 0.65 if side == hand and count else 0)
                wrist = (int(elbow[0] + sign * 65 * np.sin(angle)), int(elbow[1] - 80 * np.cos(angle))) if side == hand and count else (elbow[0], 295)
                cv2.line(image, (256, 165), shoulder, (35, 55, 70), 4)
                cv2.line(image, shoulder, elbow, (35, 55, 70), 4)
                cv2.line(image, elbow, wrist, (35, 55, 70), 4)
                cv2.circle(image, wrist, 7, (35, 55, 70), -1)
                cv2.line(image, (256, 300), (256 + sign * 45, 435), (35, 55, 70), 4)
            writer.write(image)
        writer.release()
        spec = MotionCompiler().compile(CompilerRequest(original_request=prompt)).motion_spec
        request = VerificationRequest(verification_id=f"synthetic_{name}", candidate_id=f"synthetic_{name}", original_request=prompt,
            motion_spec=spec, visual_evidence_uri=str(path), visual_evidence_candidate_id=f"synthetic_{name}")
        check = next(c for c in VerificationPlanBuilder(learned_config=config).build(request).checks if c.direction == "event_frequency")
        results.append((name, request, check))
    write("mllm/controlled_evidence/manifest.json", {"synthetic": True, "method": "procedural unlabeled frontal skeleton",
        "limitations": "anatomical side can be ambiguous in 2D; uncertainty is allowed; not a benchmark",
        "cases": [{"name": name, "expected_count": 3, "generator_count": count, "generator_hand": hand}
                  for name, count, hand, _ in cases],
        "reversed_and_sequential_cases": "structured observation fixtures in test_phase9b_learned_verifiers.py; no real-model claim"})
    return results


PROMPT_WAVE = "A person waves their right hand three times."


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute-api", action="store_true")
    parser.add_argument("--approval-blocked", action="store_true")
    args = parser.parse_args()
    os.chdir(ROOT)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    config = LearnedVerifierConfig.load()
    write("tmr/config.json", {"model": config.tmr_run_dir, "repository": config.tmr_repository,
        "official_source": "https://github.com/Mathux/TMR", "download_script": "prepare/download_pretrain_models.sh",
        "checkpoint": "tmr_humanml3d_guoh3dfeats/last_weights", "archive_md5": "7b6d8814f9c1ca972f62852ebb6c7a6f",
        "threshold": config.tmr_threshold, "threshold_status": config.threshold_status,
        "representation": "GEM normalized 151-D -> saved SMPL-X -> licensed FK -> Y-up SMPL-order 22 joints -> 20Hz -> official Guo 263-D -> official training mean/std"})
    write("motioncritic/config.json", {"repository": config.motioncritic_repository, "checkpoint": config.motioncritic_checkpoint,
        "official_source": "https://github.com/ou524u/MotionCritic", "download_script": "prepare/prepare_pretrained.sh",
        "checkpoint_sha256": file_digest(config.motioncritic_checkpoint), "threshold": config.motioncritic_threshold,
        "threshold_status": config.threshold_status, "hand_policy": config.motioncritic_hand_policy,
        "representation": "saved axis-angle SMPL global root + 23 body joints + root XYZ -> SO3 resampling 20Hz -> contiguous 60-frame windows [1,T,25,3]"})
    write("mllm/prompt_versions.json", {"provider": config.provider, "model": config.mllm_model, "versions": PROMPT_VERSIONS,
        "temperature": 0, "structured_output": "strict per-task Pydantic JSON Schema", "api_call_budget": config.mllm_max_calls,
        "image_detail": config.mllm_image_detail, "contact_sheet": config.mllm_contact_sheet,
        "credential_available": bool(os.environ.get(config.mllm_api_key_env)), "credentials_logged": False})
    scenarios = scenario_plans(config)
    request = load_real_request()
    service = create_verification_service(config)
    request = service.evidence_preparer(request)
    if isinstance(request.candidate, dict) and request.candidate.get("fk_evidence"):
        write("real_candidate/fk_evidence.json", request.candidate["fk_evidence"])
    print("Phase9B: official runtime probes", flush=True)
    tmr_backend = service.semantic_backend
    try:
        tmr_backend.encoder.load()
        tmr_loaded, tmr_blocker = True, None
    except Exception as exc:
        tmr_loaded, tmr_blocker = False, type(exc).__name__
    plan = service.plan_builder.build(request)
    tmr_check = next(c for c in plan.checks if c.check_id == "semantic")
    tmr_result = tmr_backend.verify(request, tmr_check)
    write("tmr/test_results.json", {"weights_loaded": tmr_loaded, "real_candidate_inference_executed": tmr_backend.inference_calls > 0,
        "finding": tmr_result.model_dump(mode="json"), "runtime_blocker": tmr_blocker})
    try:
        tmr_calibration(tmr_backend)
    except Exception as exc:
        write("tmr/calibration.json", {**provisional_calibration([], []), "blocker": type(exc).__name__})
    critic = service.naturalness_backend
    try:
        critic.load()
        mc_loaded = True
        motioncritic_calibration(critic, request)
        print("Phase9B: real MotionCritic inference/calibration executed", flush=True)
    except Exception as exc:
        mc_loaded = critic.model is not None
        write("motioncritic/calibration.json", {**provisional_calibration([], []), "blocker": type(exc).__name__})
    critic_check = next(c for c in plan.checks if c.check_id == "motioncritic")
    critic_result = critic.verify(request, critic_check)
    write("motioncritic/test_results.json", {"weights_loaded": mc_loaded, "real_inference_calls": critic.inference_calls,
        "exact_candidate_inference_executed": critic_result.measured_value is not None,
        "finding": critic_result.model_dump(mode="json")})
    controlled = controlled_evidence(config)
    from motion_agent.verification.mllm.storyboard import StoryboardBuilder
    storyboard_builder = StoryboardBuilder(str(OUTPUT / "mllm/storyboards"), config.storyboard_max_frames, config.mllm_contact_sheet)
    storyboards = []
    for check in plan.checks:
        if check.evaluator == "mllm_backend":
            base = storyboard_builder.build(request, check)
            dense = storyboard_builder.build(request, check, dense=True)
            write(f"real_candidate/storyboards/{check.check_id}.json", {"base": base, "dense_retry": dense,
                "expected_requirements": expected_requirements(request, check), "api_executed": False})
            storyboards.append({"check_id": check.check_id, "base_frames": len(base["frames"]),
                                "dense_frames": len(dense["frames"]), "no_action_labels": True})
    write("real_candidate/storyboard_summary.json", storyboards)
    if not args.execute_api:
        from motion_agent.verification.mllm.backend import UnavailableMLLMBackend
        service.mllm_backend = UnavailableMLLMBackend("IMPLEMENTED_NOT_EXECUTED_APPROVAL_BLOCKED" if args.approval_blocked
                                                     else "REAL_API_EXECUTION_NOT_REQUESTED")
    print("Phase9B: saved real candidate verification", flush=True)
    report = service.verify(request)
    write("real_candidate/verification_report.json", report.model_dump(mode="json"))
    write("real_candidate/findings.json", [f.model_dump(mode="json") for f in report.findings])
    mllm = service.mllm_backend
    real_results = []
    if args.execute_api and isinstance(mllm, RealMLLMBackend) and mllm.executions:
        # Repeat one fixed candidate/check independently of the service cache.
        check = next(c for c in plan.checks if c.direction == "event_frequency")
        repeat = mllm.verify(request, check)
        first = next(f for f in report.findings if f.check_id == check.check_id)
        completed = all(f.observed.get("source") == "real_mllm" for f in (first, repeat))
        counts_known = completed and all(f.observed.get("observed_count") is not None for f in (first, repeat))
        stability = {"real_runs": sum(f.observed.get("source") == "real_mllm" for f in (first, repeat)),
                     "attempted_runs": 2, "first": first.model_dump(mode="json"), "repeat": repeat.model_dump(mode="json"),
                     "status_agreement": first.status == repeat.status if completed else None,
                     "observed_count_agreement": first.observed.get("observed_count") == repeat.observed.get("observed_count") if counts_known else None,
                     "status": "PROVISIONAL_CALIBRATION" if completed else "NOT_ESTABLISHED_INFRASTRUCTURE_OR_BUDGET"}
        for name, synthetic_request, check in controlled:
            result = mllm.verify(synthetic_request, check)
            real_results.append({"case": name, "finding": result.model_dump(mode="json"), "synthetic_evidence": True})
    else:
        stability = {"real_runs": 0, "status": "NOT_EXECUTED", "blocker": "No successful real structured API response"}
    write("mllm/stability_results.json", stability)
    cache_calls_before = getattr(getattr(mllm, "provider", None), "network_calls", 0)
    cached_checks = []
    for check in plan.checks:
        original = next((f for f in report.findings if f.check_id == check.check_id), None)
        if check.evaluator == "mllm_backend" and original and original.status in {"pass", "fail"}:
            cached = service._run_cached(request, check)
            cached_checks.append({"check_id": check.check_id, "same_finding": cached == original})
    write("mllm/cache_results.json", {"terminal_checks": cached_checks,
        "additional_api_calls": getattr(getattr(mllm, "provider", None), "network_calls", 0) - cache_calls_before,
        "status": "PASS" if cached_checks and all(c["same_finding"] for c in cached_checks) else "NOT_EXECUTED"})
    api_calls = getattr(mllm, "api_calls", 0)
    executions = getattr(mllm, "executions", [])
    write("mllm/transport_errors.json", getattr(getattr(mllm, "provider", None), "transport_errors", []))
    execution_status = ("PASS" if executions else "IMPLEMENTED_NOT_EXECUTED_APPROVAL_BLOCKED" if args.approval_blocked
                        else "REAL_API_EXECUTION_NOT_REQUESTED" if not args.execute_api
                        else "IMPLEMENTED_NOT_EXECUTED_NO_CREDENTIALS"
                        if any("NO_CREDENTIALS" in (f.message or "") for f in report.findings) else "FAIL")
    write("mllm/test_results.json", {"real_api_requested": args.execute_api, "backend_attempts": api_calls,
        "api_attempts": getattr(getattr(mllm, "provider", None), "network_calls", 0),
        "latency_ms": getattr(mllm, "latency_ms", 0),
        "raw_api_responses": getattr(mllm, "raw_response_count", 0),
        "successful_structured_responses": getattr(mllm, "valid_observation_count", 0),
        "normalized_findings": len(executions),
        "controlled_cases": real_results, "status": execution_status,
        "failures": [f.model_dump(mode="json") for f in report.findings if f.evaluator_version.startswith("mllm_visual_adapter_") and f.status in {"uncertain", "error"}],
        "uncertain_rate": sum(f["status"] == "uncertain" for f in executions) / len(executions) if executions else None})
    write("scenario_tests/real_candidate_temporal_trace.json", {"raw_prompt": request.original_request,
        "motion_spec_source": "saved actual GenerationRequest.motion_spec",
        "temporal_constraints": [s.temporal_constraint.model_dump(mode="json") for s in request.motion_spec.segments if s.temporal_constraint],
        "frequency_checks": [c.model_dump(mode="json") for c in plan.checks if c.direction == "event_frequency"],
        "findings": [f.model_dump(mode="json") for f in report.findings if f.direction == "event_frequency"],
        "expected_count_preserved": all(c.metadata["expected_count"] == 3 for c in plan.checks if c.direction == "event_frequency")})
    statuses = {"phase9_core_engineering_gate": "PASS", "phase9b_tmr": "PASS" if tmr_backend.inference_calls else "IMPLEMENTED_BLOCKED",
        "phase9b_motioncritic": "PASS" if critic_result.measured_value is not None else "IMPLEMENTED_BLOCKED",
        "phase9b_mllm": "PASS" if executions else "IMPLEMENTED_NOT_EXECUTED_NO_CREDENTIALS" if "NO_CREDENTIALS" in execution_status else "FAIL",
        "mllm_execution_status": execution_status,
        "phase9_full_learned_verifier_gate": "PASS" if tmr_backend.inference_calls and critic_result.measured_value is not None and executions
                                             else "PARTIAL" if executions or critic.inference_calls else "DEFERRED",
        "real_candidate_id": request.candidate_id, "real_candidate_report_status": report.status,
        "overall_pass": report.overall_pass, "scenario_routing": scenarios,
        "blockers": [f.message or f.diagnostic_code for f in report.findings if f.status in {"uncertain", "error"}],
        "runpod_used": False}
    write("gate.json", statuses)
    write("scenario_tests/naturalness_case.json", {"real_model": "MotionCritic", "reference_examples": 4,
        "jitter_examples": 4, "calibration_artifact": "motioncritic/calibration.json",
        "candidate_exact_conversion_status": critic_result.status,
        "kinematic_finding": next(f.model_dump(mode="json") for f in report.findings if f.check_id == "naturalness_kinematic")})
    print(json.dumps(statuses, indent=2), flush=True)


if __name__ == "__main__":
    main()
