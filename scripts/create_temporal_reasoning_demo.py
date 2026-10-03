"""Real raw-text versus native temporal-compiler experiment and rendering."""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PROMPT = "A person walks forward while waving their right hand three times, then turns left and sits down."
FRAMES, FPS, SEED = 420, 30, 7


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


class RecordingGEM:
    """Observe the unmodified predict boundary, including synchronized timings."""

    def __init__(self, model):
        self.model = model
        self.receipts = []

    def predict(self, data, **kwargs):
        import torch

        meta = data["meta"][0]
        texts = meta["multi_text_data"]
        receipt = {
            "captions": texts["caption"],
            "window_start": texts["window_start"].tolist(),
            "window_end": texts["window_end"].tolist(),
            "frames": int(data["length"]),
            "seed": kwargs.get("seed", data.get("seed")),
            "static_camera": kwargs.get("static_cam", True),
            "postprocess": kwargs.get("postproc", True),
            "motion_specification": meta.get("motion_specification"),
            "hard_motion_condition_supplied": "observed_motion_3d" in data,
        }
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        result = self.model.predict(data, **kwargs)
        torch.cuda.synchronize()
        receipt["inference_seconds"] = time.perf_counter() - start
        receipt["peak_gpu_allocated_mb"] = torch.cuda.max_memory_allocated() / 1024**2
        self.receipts.append(receipt)
        return result


def tensor_statistics(params, motion):
    import torch

    if not all(torch.isfinite(v).all() and len(v) == FRAMES for v in params.values()):
        raise ValueError("Generated SMPL output must be finite with the requested frame count")
    if not torch.isfinite(motion).all() or tuple(motion.shape) != (FRAMES, 151):
        raise ValueError("Invalid generated GEM motion representation")
    return {"smpl_shapes": {k: list(v.shape) for k, v in params.items()},
            "motion_shape": list(motion.shape), "nan_count": int(torch.isnan(motion).sum()),
            "inf_count": int(torch.isinf(motion).sum()), "motion_std": float(motion.std())}


def generate(output):
    import torch
    from scripts.create_phase7_baseline_comparison import digest, prepare_native_request, protected_sources

    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in ["baseline_smpl.pt", "motionagent_smpl.pt"]):
        raise FileExistsError("Use an empty output directory for a fresh generation comparison")
    before = protected_sources()
    vendor = ROOT / "vendor" / "GENMO"
    checkpoint = vendor / "inputs" / "pretrained" / "gem_smpl.ckpt"
    checkpoint_hash = digest(checkpoint)
    sys.path.insert(0, str(vendor))
    from gem.motionagent import MotionAgentCameraContext, MotionAgentTextSegment, build_motionagent_text_data
    from scripts.demo.demo_utils import load_model, run_inference
    from motion_agent.generation import generate_motion
    from motion_agent.generation.candidate_store import CandidateStore
    from motion_agent.generation.seed_manager import seeded_torch_rng
    from motion_agent.state.schemas import GEMAdapterContext

    previous_cwd = Path.cwd()
    start = time.perf_counter()
    try:
        os.chdir(vendor)
        model = load_model(str(checkpoint), load_text_encoder=True)
    finally:
        os.chdir(previous_cwd)
    torch.cuda.synchronize()
    model_load_seconds = time.perf_counter() - start
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    recorder = RecordingGEM(model)
    shared = {"prompt": PROMPT, "frames": FRAMES, "fps": FPS, "duration": FRAMES / FPS,
              "seed": SEED, "checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_hash,
              "gpu": torch.cuda.get_device_name(), "python": platform.python_version(),
              "python_executable": sys.executable, "torch": torch.__version__, "cuda": torch.version.cuda,
              "model_load_seconds": model_load_seconds, "sampler": "gem_checkpoint_default",
              "postprocess": True, "manual_timing_override": False, "repetition_expansion": False,
              "keyframes_applied": False, "constraint_satisfaction_evaluated": False,
              "motion_quality_evaluated": False, "all_model_weights_frozen": True}

    data, _ = build_motionagent_text_data(
        [MotionAgentTextSegment(caption=PROMPT, length=FRAMES, name="raw_user_text")],
        MotionAgentCameraContext(width=1280, height=720, static_camera=True), seed=SEED)
    with seeded_torch_rng(SEED, "cuda"), torch.inference_mode():
        baseline = run_inference(recorder, data, static_cam=True)
    baseline_params = {k: v.detach().cpu() for k, v in baseline["body_params_global"].items()}
    baseline_motion = baseline["net_outputs"]["model_output"]["pred_x"].detach().cpu().reshape(FRAMES, 151)
    torch.save(baseline_params, output / "baseline_smpl.pt")
    torch.save(baseline_motion, output / "baseline_motion.pt")
    baseline_meta = {**shared, "pipeline": ["raw text", "GEM", "SMPL"],
                     "candidate_id": "baseline_raw_text_seed7", "inference_calls": 1,
                     "cache_hits": 0, "gem_input_receipt": recorder.receipts[0],
                     **tensor_statistics(baseline_params, baseline_motion)}
    write_json(output / "baseline_metadata.json", baseline_meta)
    del baseline

    start = time.perf_counter()
    request, trace = prepare_native_request(output, PROMPT, FRAMES, FPS, SEED)
    compile_seconds = time.perf_counter() - start
    spec = request.motion_spec.model_dump(mode="json")
    if spec["original_request"] != PROMPT:
        raise ValueError("The original prompt changed")
    write_json(output / "compiler_trace.json", spec)
    write_json(output / "planner_trace.json", trace)
    write_json(output / "generation_request.json", request.model_dump(mode="json"))
    if request.condition_bundle.hard_motion_condition_handle or request.condition_bundle.keyframe_specs:
        raise ValueError("The comparison must use semantic compilation alone")
    store = CandidateStore(request.output_policy.artifact_root)
    with torch.inference_mode():
        result = generate_motion(request, model=recorder, checkpoint_version=checkpoint_hash,
            adapter_context=GEMAdapterContext(width=1280, height=720, fps=FPS, total_frames=FRAMES), candidate_store=store)
    write_json(output / "generation_result.json", result.model_dump(mode="json"))
    if result.status != "success" or result.inference_calls != 1 or result.cache_hits:
        raise RuntimeError(f"Fresh generation failed: {result.model_dump(mode='json')}")
    candidate = result.candidates[0]
    loaded = store.load(candidate.candidate_id)
    persisted = candidate.metadata.generation_request
    if persisted.motion_spec.model_dump(mode="json") != spec:
        raise ValueError("Native compiler specification changed before persistence")
    if persisted.text_condition != request.text_condition or recorder.receipts[1]["motion_specification"] != spec:
        raise ValueError("Native compiler conditions changed at the GEM boundary")
    if recorder.receipts[0]["seed"] != recorder.receipts[1]["seed"]:
        raise ValueError("Sampler seeds differ")
    torch.save(loaded["body_params_global"], output / "motionagent_smpl.pt")
    agent_meta = {**shared, "pipeline": ["raw text", "ScriptedPlanner", "Motion Compiler", "Temporal Resolver",
        "GenerationRequest", "GEM", "SMPL"], "planner_backend": "ScriptedPlanner",
        "candidate_id": candidate.candidate_id, "compile_seconds": compile_seconds,
        "generation_runtime_seconds": result.runtime_ms / 1000, "inference_calls": result.inference_calls,
        "cache_hits": result.cache_hits, "gem_input_receipt": recorder.receipts[1],
        "compiler_output_forwarded_unchanged": True, "compiler_trace_generated_successfully": True,
        "candidate_metadata": candidate.metadata.model_dump(mode="json"),
        **tensor_statistics(loaded["body_params_global"], loaded["motion_repr"])}
    write_json(output / "motionagent_metadata.json", agent_meta)
    after = protected_sources()
    write_json(output / "source_integrity.json", {"unchanged": before == after, "before": before, "after": after})
    if before != after:
        raise RuntimeError("Production sources changed during generation")
    print(json.dumps({"generation": "success", "frames": FRAMES, "fps": FPS, "seed": SEED,
        "baseline_inference_seconds": recorder.receipts[0]["inference_seconds"],
        "motionagent_inference_seconds": recorder.receipts[1]["inference_seconds"], "compiler_trace": str(output / "compiler_trace.json")}), flush=True)


def comparison_frames(left, right):
    import numpy as np
    from PIL import Image, ImageDraw
    from scripts.render_motion_comparison import overlay_fonts

    if len(left) != FRAMES or len(right) != FRAMES:
        raise ValueError("Both videos must contain exactly the requested number of frames")
    width, height = left[0].shape[1], left[0].shape[0]
    title, body, small = overlay_fonts()
    header, footer = 150, 44
    for index, (baseline, agent) in enumerate(zip(left, right)):
        canvas = Image.new("RGB", (2 * width, height + header + footer), (245, 248, 250))
        canvas.paste(Image.fromarray(baseline), (0, header))
        canvas.paste(Image.fromarray(agent), (width, header))
        draw = ImageDraw.Draw(canvas)
        draw.text((20, 12), "Raw GEM", font=title, fill=(30, 45, 52))
        draw.text((width + 20, 12), "MotionAgent + Temporal Resolver", font=title, fill=(30, 45, 52))
        draw.text((20, 55), f"Duration: {FRAMES / FPS:g} seconds | {FRAMES} frames | {FPS} FPS | Seed: {SEED}", font=body, fill=(45, 62, 70))
        line, row = "Prompt:", 0
        for word in PROMPT.split():
            trial = line + " " + word
            if draw.textlength(trial, font=body) > 2 * width - 40:
                draw.text((20, 85 + row * 24), line, font=body, fill=(45, 62, 70))
                line, row = word, row + 1
            else:
                line = trial
        draw.text((20, 85 + row * 24), line, font=body, fill=(45, 62, 70))
        draw.text((20, header + height + 10), f"Time: {index / FPS:.2f}s / {FRAMES / FPS:.2f}s | Frame: {index + 1}/{FRAMES}", font=small, fill=(45, 62, 70))
        draw.line((width, header, width, header + height), fill=(180, 190, 196), width=2)
        yield np.asarray(canvas)


def render(output):
    import cv2
    from scripts.render_motion_comparison import build_vertices, camera_for_sequences, encode_video, load_motion, render_sequence, validate_video

    start = time.perf_counter()
    model_path = ROOT / "vendor" / "GENMO" / "inputs" / "checkpoints" / "body_models" / "smplx" / "SMPLX_NEUTRAL.npz"
    params_b, _ = load_motion(output / "baseline_smpl.pt")
    params_a, _ = load_motion(output / "motionagent_smpl.pt")
    verts_b, faces, offset_b = build_vertices(params_b, model_path, "cuda")
    verts_a, _, offset_a = build_vertices(params_a, model_path, "cuda")
    width, height = 960, 720
    pose, camera = camera_for_sequences([verts_b, verts_a], width, height)
    color = [0.02, 0.5, 0.55, 1]
    frames_b, visible_b = render_sequence(verts_b, faces, pose, width, height, color)
    frames_a, visible_a = render_sequence(verts_a, faces, pose, width, height, color)
    encode_video(output / "baseline.mp4", frames_b, FPS)
    encode_video(output / "motionagent.mp4", frames_a, FPS)
    del frames_b, frames_a

    def decode(path):
        reader = cv2.VideoCapture(str(path))
        frames = []
        while True:
            ok, frame = reader.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        reader.release()
        return frames

    encode_video(output / "comparison.mp4", comparison_frames(decode(output / "baseline.mp4"), decode(output / "motionagent.mp4")), FPS)
    videos = {name: validate_video(output / f"{name}.mp4", FRAMES, FPS,
        preview=output / f"{name}_preview.png", sample_dir=output / "observation_samples" / name)
        for name in ["baseline", "motionagent", "comparison"]}
    report = {"camera": camera, "identical_render_settings": True, "body_color": color,
        "rendering_method": "SMPL-X + trimesh + pyrender EGL + FFmpeg H.264", "device": "cuda",
        "render_seconds": time.perf_counter() - start, "videos": videos,
        "baseline_visibility": visible_b, "motionagent_visibility": visible_a,
        "baseline_render_translation": offset_b, "motionagent_render_translation": offset_a,
        "source_motion_edited": False, "frame_padding": False, "retiming": False}
    write_json(output / "render_report.json", report)
    for name in ["baseline", "motionagent"]:
        path = output / f"{name}_metadata.json"
        metadata = json.loads(path.read_text())
        metadata["rendering"] = {"camera": camera, "body_color": color, **videos[name]}
        write_json(path, metadata)
    print(json.dumps({"rendering": "success", "render_seconds": report["render_seconds"], "videos": videos}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["generate", "render"])
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "temporal_reasoning_demo")
    args = parser.parse_args()
    {"generate": generate, "render": render}[args.mode](args.output.resolve())


if __name__ == "__main__":
    main()
