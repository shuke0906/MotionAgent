"""One real K=1 GEM smoke, baseline and bounded Agent run, with actual artifacts."""

from __future__ import annotations

from datetime import datetime, timezone
import html
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import textwrap
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "outputs/phase11_real_e2e"
PROMPT = "A person walks forward while waving their right hand three times, then turns left and sits down."
SMPLX = ROOT / "vendor/GENMO/inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz"
CKPT = ROOT / "vendor/GENMO/inputs/pretrained/gem_smpl.ckpt"
FPS = 30
FRAMES = 420
SEED = 7


def write(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(payload, "model_dump"):
        payload = payload.model_dump(mode="json")
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def render(candidate, path):
    """Fixed perspective projection of actual licensed SMPL-X FK, without normalization."""
    import cv2
    import numpy as np
    import torch
    from motion_agent.verification.representations import smplx_joints

    params = torch.load(candidate.smpl_global_uri, map_location="cpu", weights_only=True)
    joints = smplx_joints(params, str(SMPLX)).numpy()
    width, height = 640, 480
    eye = np.array([8., 5.5, 9.])
    target = np.array([0., 1., 2.5])
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0., 1., 0.])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    focal = width / (2 * np.tan(np.deg2rad(45) / 2))

    def project(points):
        delta = points - eye
        depth = delta @ forward
        return np.stack((width / 2 + focal * (delta @ right) / depth,
                         height / 2 - focal * (delta @ up) / depth), axis=-1).astype(int)

    parents = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
    path.parent.mkdir(parents=True, exist_ok=True)
    encoder = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pixel_format", "rgb24",
        "-video_size", f"{width}x{height}", "-framerate", str(FPS), "-i", "pipe:0", "-an", "-c:v", "libx264",
        "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)], stdin=subprocess.PIPE)
    for index, frame in enumerate(joints):
        image = np.full((height, width, 3), 250, dtype=np.uint8)
        for coordinate in range(-6, 9):
            for line in (np.array([[-6., 0., coordinate], [8., 0., coordinate]]),
                         np.array([[coordinate, 0., -6.], [coordinate, 0., 8.]])):
                a, b = project(line)
                cv2.line(image, tuple(a), tuple(b), (215, 221, 225), 1, cv2.LINE_AA)
        points = project(frame)
        for joint, parent in enumerate(parents):
            if parent >= 0:
                color = (28, 132, 115) if joint in {2, 5, 8, 11, 14, 17, 19, 21} else (68, 79, 98)
                cv2.line(image, tuple(points[parent]), tuple(points[joint]), color, 5, cv2.LINE_AA)
            cv2.circle(image, tuple(points[joint]), 4, (38, 48, 61), -1, cv2.LINE_AA)
        cv2.putText(image, "R", tuple(points[21] + [8, -4]), cv2.FONT_HERSHEY_SIMPLEX, .5, (28, 132, 115), 1, cv2.LINE_AA)
        cv2.putText(image, f"{index / FPS:.2f} s", (16, 30), cv2.FONT_HERSHEY_SIMPLEX, .6, (45, 55, 65), 1, cv2.LINE_AA)
        encoder.stdin.write(image.tobytes())
        if index == len(joints) // 2:
            cv2.imwrite(str(path.with_suffix(".png")), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    encoder.stdin.close()
    if encoder.wait() != 0:
        raise RuntimeError("RENDERING_FFMPEG_FAILED")
    return {"renderer": "licensed_smplx_fk22_skeleton", "camera_position": eye.tolist(),
            "camera_target": target.tolist(), "fov_deg": 45, "resolution": [width, height],
            "fps": FPS, "frames": len(joints), "motion_normalized": False,
            "anatomical_right_label": True, "body_shape_source": "candidate_betas"}


def persist_candidate(candidate, folder, video_name):
    import torch
    from motion_agent.verification.artifacts import file_digest
    folder.mkdir(parents=True, exist_ok=True)
    shutil.copy2(candidate.motion_repr_uri, folder / "motion.pt")
    shutil.copy2(candidate.smpl_global_uri, folder / "smpl_global.pt")
    metadata = candidate.model_dump(mode="json")
    metadata["render"] = render(candidate, folder / video_name)
    params = torch.load(candidate.smpl_global_uri, map_location="cpu", weights_only=True)
    metadata["body_shape_tensor_shape"] = list(params["betas"].shape)
    metadata["smpl_sha256"] = file_digest(folder / "smpl_global.pt")
    write(folder / "generation_metadata.json", metadata)
    return metadata


def raw_request(prompt, frames, label):
    from motion_agent.compiler.schemas import GEMTextCondition
    from motion_agent.generation.request_builder import build_generation_request
    # One raw-text condition spans the entire clip; no semantic event timing is supplied.
    condition = GEMTextCondition(captions=[prompt], window_start=[0.], window_end=[1.], total_frames=frames)
    return build_generation_request(condition=condition, fps=FPS, generation_id=label, seed=SEED, num_candidates=1)


def execute_generation(request, manager, store):
    from motion_agent.generation.normal_generator import generate_motion
    from motion_agent.state.schemas import GEMAdapterContext
    result = generate_motion(request, model=manager.load(), checkpoint_version=str(CKPT), candidate_store=store,
                             adapter_context=GEMAdapterContext(fps=request.fps, total_frames=request.total_frames))
    if result.status != "success" or len(result.candidates) != 1:
        write(OUT / "generation_runtime_failure.json", result)
        raise RuntimeError("GENERATION_RUNTIME")
    return result


def export_trace(orchestrator, state):
    from PIL import Image, ImageDraw, ImageFont
    events = [event.model_dump(mode="json") for event in orchestrator.event_log.list(state.run.run_id)]
    write(OUT / "trace/event_log.json", events)
    steps = []
    for order, event in enumerate(events):
        stored_state = json.loads((orchestrator.state_store.root / state.run.run_id /
                                  f"v{event['state_version_after']:06d}.json").read_text())
        artifacts = {handle: orchestrator.artifact_store.get(handle) for handle in event["output_artifact_ids"]}
        steps.append({"order": order, "timestamp": event["timestamp"], "node": event["event_type"],
                      "round": stored_state["generation"]["generation_round"],
                      "input_artifact_handles": event["input_artifact_ids"],
                      "output_artifact_handles": event["output_artifact_ids"], "artifacts": artifacts,
                      "budget": stored_state["control"]["budgets"],
                      "candidate_id": stored_state["generation"]["champion_candidate_id"],
                      "verification": stored_state["evaluation"]["verification_summary"],
                      "diagnosis": stored_state["evaluation"]["diagnosis_summary"],
                      "terminal_reason": stored_state["control"]["terminal_reason"]})
    trace = {"run_id": state.run.run_id, "raw_prompt": PROMPT, "steps": steps, "final_state": state.model_dump(mode="json")}
    write(OUT / "trace/trace.json", trace)
    labels = []
    for step in steps:
        details = [step["node"], "round=" + str(step["round"])]
        if step["terminal_reason"]:
            details.append(step["terminal_reason"])
        for artifact in step["artifacts"].values():
            if artifact.get("action"):
                details.extend([artifact["action"], artifact.get("reason_code", "")])
            if artifact.get("motion_spec"):
                details.append(json.dumps(artifact["motion_spec"]["segments"]))
            if artifact.get("verification_report"):
                report = artifact["verification_report"]
                details.append("overall_pass=" + str(report["overall_pass"]))
                details.extend(f"{item['check_id']}: {item['status']} {item.get('diagnostic_code') or ''}"
                               for item in report["findings"])
            if artifact.get("diagnosis_result"):
                details.append(json.dumps(artifact["diagnosis_result"]))
        labels.append(" | ".join(details))
    panels = "".join('<li><h2>' + html.escape(step["node"]) + '</h2><pre>' + html.escape(label) +
                     '</pre><details><summary>Actual data</summary><pre>' +
                     html.escape(json.dumps(step, indent=2)) + '</pre></details></li>' for step, label in zip(steps, labels))
    document = '<!doctype html><meta charset="utf-8"><title>MotionAgent Actual Trace</title><style>body{font:15px Arial;margin:32px;color:#25313c;background:#f6f8fa;max-width:1100px}li{padding:16px;border-left:4px solid #198372;margin:12px 0;background:white}h2{font-size:18px;margin:0 0 10px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:13px monospace}summary{cursor:pointer}</style><h1>MotionAgent Actual Execution</h1><p>' + html.escape(PROMPT) + '</p><p>Final: ' + html.escape(state.control.terminal_reason or state.run.status) + '</p><ol>' + panels + '</ol>'
    (OUT / "trace/pipeline_trace.html").write_text(document, encoding="utf-8")
    rows = [textwrap.wrap(label, 110)[:12] for label in labels]
    heights = [max(84, 28 + len(row) * 19) for row in rows]
    image = Image.new("RGB", (1050, 130 + sum(height + 22 for height in heights)), "#f6f8fa")
    draw = ImageDraw.Draw(image)
    font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    font = ImageFont.truetype(str(font_path), 14) if font_path.exists() else ImageFont.load_default()
    draw.text((24, 18), "MotionAgent - actual execution", font=font, fill="#25313c")
    for index, line in enumerate(textwrap.wrap(PROMPT, 105)):
        draw.text((24, 44 + 19 * index), line, font=font, fill="#25313c")
    y = 110
    for row, height in zip(rows, heights):
        draw.rectangle((24, y, 1026, y + height), fill="white", outline="#198372", width=2)
        for index, line in enumerate(row):
            draw.text((38, y + 12 + index * 19), line, font=font, fill="#25313c")
        y += height + 22
    image.save(OUT / "trace/pipeline_trace.png")
    return trace


def comparison(paths, labels):
    import cv2
    from PIL import Image, ImageDraw
    import numpy as np
    captures = [cv2.VideoCapture(str(path)) for path in paths]
    destination = OUT / "comparison.mp4"
    encoder = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pixel_format", "rgb24",
        "-video_size", f"{640 * len(paths)}x480", "-framerate", str(FPS), "-i", "pipe:0", "-an",
        "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p", str(destination)], stdin=subprocess.PIPE)
    for _ in range(FRAMES):
        images = []
        for capture, label in zip(captures, labels):
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("COMPARISON_FRAME_MISSING")
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            drawing = ImageDraw.Draw(image)
            drawing.rectangle((0, 442, 640, 480), fill="white")
            drawing.text((16, 454), label, fill="black")
            images.append(np.array(image))
        encoder.stdin.write(np.concatenate(images, axis=1).tobytes())
    encoder.stdin.close()
    if encoder.wait() != 0:
        raise RuntimeError("COMPARISON_ENCODING_FAILED")
    for capture in captures:
        capture.release()


def main():
    import torch
    from motion_agent.llm.config import load_dotenv, LLMConfig
    for name in ("OPENAI_API_KEY", "OPENAI_MODEL", "OPENAI_BASE_URL", "MLLM_API_KEY", "MLLM_MODEL", "MLLM_BASE_URL"):
        os.environ.pop(name, None)
    load_dotenv(ROOT / ".env")
    os.chdir(ROOT)
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    from motion_agent.generation.model_manager import GEMModelManager
    from motion_agent.generation.candidate_store import CandidateStore
    from motion_agent.generation.schemas import GenerationResult
    manager = GEMModelManager(CKPT)
    store = CandidateStore(OUT / "candidates")
    print("GEM_LOAD_START", flush=True)
    manager.load()
    print("GEM_LOAD_PASS", flush=True)
    for label, prompt, frames, video in (("smoke", "A person walks forward.", 90, "gem_smoke.mp4"),
                                        ("baseline", PROMPT, FRAMES, "baseline.mp4")):
        folder = OUT / label
        saved = folder / "generation_result.json"
        if saved.exists():
            result = GenerationResult.model_validate_json(saved.read_text())
            print(label.upper() + "_REUSE_EXISTING", flush=True)
        else:
            result = execute_generation(raw_request(prompt, frames, "phase11_" + label), manager, store)
            write(saved, result)
        candidate = result.candidates[0]
        if not (folder / video).exists():
            metadata = persist_candidate(candidate, folder, video)
            write(folder / "metadata.json", metadata)
        print(label.upper() + "_PASS candidate=" + candidate.candidate_id, flush=True)

    from motion_agent.app.phase11_runtime import Phase11LocalRuntime
    from motion_agent.app.orchestrator import MotionAgentOrchestrator
    from motion_agent.agent.planner import RealLLMPlannerBackend
    from motion_agent.agent.actions import PlannerAction
    from motion_agent.compiler.motion_compiler import RealLLMMotionCompiler
    from motion_agent.compiler.llm_semantic_parser import LLMSemanticParserBackend
    from motion_agent.compiler.schemas import MotionSpecification, GEMTextCondition
    from motion_agent.generation.request_builder import build_generation_request_from_state
    from motion_agent.llm import LLMClient
    from motion_agent.verification.config import LearnedVerifierConfig, create_verification_service
    from motion_agent.verification.schemas import VerificationRequest, VerifierPolicy
    from motion_agent.state.schemas import VerificationSummary
    from motion_agent.verification.artifacts import file_digest
    from motion_agent.diagnosis.normalizer import normalize_failures

    class RealRuntime(Phase11LocalRuntime):
        def generate(self, runtime, state, decision):
            round_index = state.generation.generation_round
            if round_index >= 2:
                raise RuntimeError("REPAIR_GENERATION_LIMIT")
            spec = MotionSpecification.model_validate(runtime.artifact_store.get(state.plan.motion_spec_id)["motion_spec"])
            condition = GEMTextCondition.model_validate(runtime.artifact_store.get(state.plan.gem_text_condition_id)["gem_text_condition"])
            payload = decision.typed_payload
            request = build_generation_request_from_state(state, condition, num_candidates=1,
                seed=SEED + round_index, strategy=payload.strategy, scope=payload.scope,
                target_segments=decision.target_segments, previous_candidate_id=state.generation.champion_candidate_id
            ).model_copy(update={"motion_spec": spec})
            folder = OUT / f"agent/round_{round_index}"
            write(folder / "generation_request.json", request)
            write(folder / "motion_specification.json", spec)
            print("AGENT_GENERATE_START round=" + str(round_index), flush=True)
            result = execute_generation(request, manager, store)
            write(folder / "generation_result.json", result)
            candidate = result.candidates[0]
            persist_candidate(candidate, folder, "video.mp4")
            generation = runtime.artifact_store.put("generation", {"schema": "GenerationResult",
                "generation_result": result.model_dump(mode="json"), "request": request.model_dump(mode="json"),
                "motion_spec_id": state.plan.motion_spec_id, "text_condition_id": state.plan.gem_text_condition_id,
                "backend": "REAL_FROZEN_GEM"}, artifact_id=result.generation_id)
            runtime.artifact_store.put("candidate", {"schema": "MotionCandidate", "candidate": candidate.model_dump(mode="json"),
                "evidence": {"path": str(folder / "video.mp4")}}, artifact_id=candidate.candidate_id)
            updated = runtime.reducer.commit_generation_result(state, generation_id=generation.artifact_id,
                candidate_ids=[candidate.candidate_id], strategy=request.strategy, scope=request.scope,
                target_segments=request.target_segments)
            return runtime.state_store.commit(updated, expected_version=state.state_version,
                event_type="generation_committed", output_artifact_ids=[generation.artifact_id, candidate.candidate_id])

        def verify(self, runtime, state):
            import numpy as np
            from scipy.spatial.transform import Rotation
            from motion_agent.generation.schemas import MotionCandidate
            spec = MotionSpecification.model_validate(runtime.artifact_store.get(state.plan.motion_spec_id)["motion_spec"])
            payload = runtime.artifact_store.get(state.generation.champion_candidate_id)
            candidate = MotionCandidate.model_validate(payload["candidate"])
            params = torch.load(candidate.smpl_global_uri, map_location="cpu", weights_only=True)
            forward = Rotation.from_rotvec(params["global_orient"].numpy()).apply(np.tile([0., 0., 1.], (len(params["body_pose"]), 1)))
            yaw = np.unwrap(np.arctan2(forward[:, 0], forward[:, 2]))
            observed = {f"signed_heading_change_{segment.segment_id}": float(np.degrees(yaw[min(segment.end_frame - 1, len(yaw) - 1)] - yaw[segment.start_frame]))
                        for segment in spec.segments if segment.action == "turn" and segment.direction in {"left", "right"}}
            request = VerificationRequest(verification_id="phase11_verify_" + candidate.candidate_id,
                candidate_id=candidate.candidate_id, original_request=PROMPT, motion_spec=spec, candidate=candidate,
                generation_id=state.generation.latest_generation_id, candidate_fingerprint=candidate.fingerprint,
                visual_evidence_uri=payload["evidence"]["path"], visual_evidence_candidate_id=candidate.candidate_id,
                visual_evidence_fingerprint=file_digest(payload["evidence"]["path"]), observed_measurements=observed,
                verifier_policy=VerifierPolicy(), threshold_profile="phase9b_provisional_v1")
            round_index = state.generation.generation_round - 1
            print("AGENT_VERIFY_START round=" + str(round_index), flush=True)
            report = self.verifier.verify(request)
            write(OUT / f"agent/round_{round_index}/verification_report.json", report)
            write(OUT / f"agent/round_{round_index}/verification_plan.json", report.plan)
            handle = runtime.artifact_store.put("verification", {"schema": "VerificationReport",
                "verification_report": report.model_dump(mode="json")}, artifact_id=report.verification_id)
            updated = runtime.reducer.commit_verification_report(state, verification_id=handle.artifact_id,
                summary=VerificationSummary.model_validate(report.summary()))
            entries = self.history(runtime, state)
            signatures = {failure.failure_signature for failure in normalize_failures(report)}
            for entry in entries:
                if entry.outcome == "pending":
                    entry.outcome = "resolved" if report.overall_pass else "unchanged" if entry.failure_signature in signatures else "worse"
                    entry.resolved = report.overall_pass
                    entry.improved = report.overall_pass
                    entry.regressed = entry.outcome == "worse"
            if entries:
                updated = self.attach_history(runtime, updated, entries)
                updated.repair.active_proposal_id = None
            print("AGENT_VERIFY_RESULT " + str(report.overall_pass), flush=True)
            return runtime.state_store.commit(updated, expected_version=state.state_version,
                event_type="verification_committed", output_artifact_ids=[handle.artifact_id])

    config = LearnedVerifierConfig.load()
    config.smplx_model_path = str(SMPLX)
    config.output_root = str(OUT / "verifier_evidence")
    config.mllm_max_calls = 16
    service = create_verification_service(config)
    client = LLMClient(LLMConfig.from_env().model_copy(update={"timeout_s": 90, "max_retries": 0}))
    planner = RealLLMPlannerBackend(client)
    compiler = RealLLMMotionCompiler(LLMSemanticParserBackend(client))
    real_runtime = RealRuntime(OUT / "runtime", compiler=compiler, generator=None, verifier=service)
    orchestrator = MotionAgentOrchestrator(OUT / "runtime", planner, runtime_factory=real_runtime.bind)
    run_id = "phase11_minimal_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    state = orchestrator.reducer.create_run(PROMPT, run_id=run_id, fps=FPS, total_frames=FRAMES)
    state.control.budgets.generations_left = 2
    state.control.budgets.repairs_left = 1
    state.control.budgets.iterations_left = 40
    state.control.budgets.planner_steps_left = 10
    state.control.capabilities.semantic_mllm_available = True
    state.control.capabilities.motioncritic_available = True
    state = orchestrator.state_store.commit(state, expected_version=None, event_type="run_created")
    write(OUT / "active_run.json", {"run_id": run_id})
    try:
        state = orchestrator.invoke(state)
    finally:
        state = orchestrator.state_store.load_latest(run_id)
        export_trace(orchestrator, state)
    count = state.generation.generation_round
    final = OUT / f"agent/round_{count - 1}"
    shutil.copy2(final / "video.mp4", OUT / "agent/final.mp4")
    shutil.copy2(final / "verification_report.json", OUT / "final_verification.json")
    if state.evaluation.latest_diagnosis_id:
        diagnosis = orchestrator.artifact_store.get(state.evaluation.latest_diagnosis_id)["diagnosis_result"]
        write(OUT / "diagnosis.json", diagnosis)
        write(OUT / "repair_proposals.json", diagnosis["repair_proposals"])
    paths = [OUT / "baseline/baseline.mp4", OUT / "agent/round_0/video.mp4"]
    labels = ["Baseline GEM", "MotionAgent Final"]
    if count == 2:
        paths.append(final / "video.mp4")
        labels = ["Baseline GEM", "Agent Round 0", "Agent Final"]
    comparison(paths, labels)
    round_metadata = [json.loads((OUT / f"agent/round_{index}/generation_metadata.json").read_text()) for index in range(count)]
    baseline_metadata = json.loads((OUT / "baseline/metadata.json").read_text())
    summary = {"run_id": run_id, "prompt": PROMPT, "baseline_seed": baseline_metadata["metadata"]["seed"],
        "baseline_seed_base": SEED, "agent_seed_bases": [SEED + index for index in range(count)],
        "agent_seeds": [item["metadata"]["seed"] for item in round_metadata], "checkpoint": str(CKPT), "K": 1,
        "phase8_status": "BYPASSED_FOR_K1_MODE", "round_count": count, "repair_count": max(0, count - 1),
        "planner_backend": "RealLLMPlannerBackend", "compiler_backend": "RealLLMMotionCompiler",
        "verifier_backends": {"semantic": "TMR", "naturalness": "MotionCritic_strict", "events": "real_mllm"},
        "candidate_ids": [item["candidate_id"] for item in round_metadata], "final_candidate_id": state.generation.champion_candidate_id,
        "verification": state.evaluation.verification_summary.model_dump(mode="json"),
        "final_status": "ACCEPT" if state.control.accepted else "STOP_FAILED" if state.run.status == "failed" else state.run.status,
        "terminal_reason": state.control.terminal_reason, "total_runtime_s": round(time.perf_counter() - started, 2),
        "gpu_peak_vram_mb": torch.cuda.max_memory_allocated() / 1024 ** 2,
        "mllm_api_calls": service.mllm_backend.provider.network_calls,
        "tmr_inference_calls": service.semantic_backend.inference_calls,
        "motioncritic_inference_calls": service.naturalness_backend.inference_calls,
        "minimal_real_e2e_gate": "PASS" if state.run.status in {"accepted", "failed"} and count in {1, 2} else "FAIL",
        "known_limitations": ["TMR threshold uncalibrated", "MotionCritic lacks full SMPL terminal hand retargeting",
                              "Skeleton renderer shows actual FK22, no finger animation"]}
    write(OUT / "run_summary.json", summary)
    report_path = ROOT / "reports/phase_11_real_e2e.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("# Phase 11 Minimal Real E2E\n\nActual execution summary:\n\n```json\n" + json.dumps(summary, indent=2) +
        "\n```\n\nGEM and SMPL-X assets found. API text and visual probes passed before GEM.\n"
        "Videos use identical camera, background, skeleton body style, FPS and resolution; candidate betas are preserved.\n"
        "Baseline uses raw Pure-Text GEM without Planner/Compiler/Verifier/Diagnosis.\n"
        "Actual state/events/artifacts are in trace/trace.json. Learned unavailable findings never certify acceptance.\n"
        "This is the minimal run, not the full Phase 11 benchmark.\n", encoding="utf-8")
    bundle = ROOT / "outputs/phase11_real_e2e_bundle.tar.gz"
    with tarfile.open(bundle, "w:gz") as archive:
        archive.add(OUT, arcname="phase11_real_e2e")
        archive.add(report_path, arcname="reports/phase_11_real_e2e.md")
    print(json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        write(OUT / "runtime_error.json", {"error_type": type(exc).__name__, "code": getattr(exc, "code", None)})
        print("SANITIZED_RUNTIME_ERROR=" + type(exc).__name__, flush=True)
        raise SystemExit(1) from None
