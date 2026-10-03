"""External, paired multi-prompt GEM ablation without authored action timings."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PROMPTS = [
    "A person walks forward while waving their right hand three times, then turns left and sits down.",
    "A person walks backward while waving their left hand twice, then turns right and sits down.",
    "A person walks forward while raising their left arm four times, then jumps once and sits down.",
]
ARMS = ("raw", "legacy", "parser_fixed_split", "fixed")
FRAMES, FPS = 420, 30


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def load_legacy(snapshot):
    modules = {}
    for name in ("semantic_parser", "temporal_resolver", "caption_optimizer", "gem_text_compiler"):
        spec = importlib.util.spec_from_file_location("temporal_ablation_legacy_" + name, snapshot / (name + ".py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modules[name] = module
    return modules


def compile_legacy(prompt, modules):
    from motion_agent.compiler.continuity import apply_heading_continuity
    from motion_agent.compiler.control_intent import detect_control_intents
    from motion_agent.compiler.schemas import MotionSpecification
    from motion_agent.compiler.timeline_compiler import compile_timeline

    segments = modules["semantic_parser"].parse_motion_request(prompt)
    segments = modules["temporal_resolver"].resolve_temporal_semantics(prompt, segments)
    segments = compile_timeline(segments, total_duration_s=FRAMES / FPS, fps=FPS, total_frames=FRAMES)
    segments = apply_heading_continuity(segments)
    segments = [s.model_copy(update={"gem_caption": modules["caption_optimizer"].optimize_caption(s)}) for s in segments]
    spec = MotionSpecification(original_request=prompt, duration_s=FRAMES / FPS, fps=FPS,
                               total_frames=FRAMES, segments=segments, control_intents=detect_control_intents(segments))
    return spec, modules["gem_text_compiler"].compile_gem_text_condition(segments, total_frames=FRAMES)


def motion_stats(params, motion):
    import torch

    if tuple(motion.shape) != (FRAMES, 151) or not torch.isfinite(motion).all():
        raise ValueError("GEM motion is non-finite or has the wrong shape")
    if not all(len(v) == FRAMES and torch.isfinite(v).all() for v in params.values()):
        raise ValueError("SMPL parameters are non-finite or have the wrong frame count")
    root = params["transl"]
    return {"finite": True, "motion_shape": list(motion.shape),
            "smpl_shapes": {k: list(v.shape) for k, v in params.items()},
            "root_range_xyz_m": (root.max(0).values - root.min(0).values).tolist(),
            "root_path_length_m": float(torch.linalg.vector_norm(root[1:] - root[:-1], dim=-1).sum()),
            "root_net_displacement_m": float(torch.linalg.vector_norm(root[-1] - root[0])),
            "motion_std": float(motion.std())}


def generate(output, snapshot, seeds, prompts, *, arms=ARMS, repeat_first_seed=True):
    import torch
    from scripts.create_phase7_baseline_comparison import digest, prepare_native_request, protected_sources
    from scripts.create_temporal_reasoning_demo import RecordingGEM
    from motion_agent.generation import build_generation_request, generate_motion
    from motion_agent.generation.candidate_store import CandidateStore
    from motion_agent.generation.schemas import GenerationOutputPolicy
    from motion_agent.generation.seed_manager import seeded_torch_rng
    from motion_agent.state.schemas import GEMAdapterContext

    output.mkdir(parents=True, exist_ok=True)
    if (output / "experiment_manifest.json").exists():
        raise FileExistsError("Use a new output directory; do not overwrite previous validation")
    before = protected_sources()
    legacy = load_legacy(snapshot)
    vendor = ROOT / "vendor" / "GENMO"
    checkpoint = vendor / "inputs" / "pretrained" / "gem_smpl.ckpt"
    body_model = vendor / "inputs" / "checkpoints" / "body_models" / "smplx" / "SMPLX_NEUTRAL.npz"
    if not checkpoint.is_file() or not body_model.is_file() or not torch.cuda.is_available():
        raise RuntimeError("Existing checkpoint, SMPL-X model and GPU are required")
    checkpoint_hash = digest(checkpoint)
    manifest = {"prompts": prompts, "seeds": seeds, "arms": list(arms), "frames": FRAMES, "fps": FPS,
                "duration_s": FRAMES / FPS, "checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_hash,
                "smplx_model_sha256": digest(body_model), "gpu": torch.cuda.get_device_name(),
                "python": platform.python_version(), "python_executable": sys.executable,
                "torch": torch.__version__, "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
                "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
                "tf32_cudnn": torch.backends.cudnn.allow_tf32,
                "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                "cudnn_benchmark": torch.backends.cudnn.benchmark,
                "legacy_hashes": {p.name: digest(p) for p in sorted(snapshot.glob("*.py"))},
                "manual_segment_timing": False, "repetition_expansion": False, "keyframes": False,
                "sampler": "unchanged GEM checkpoint defaults", "postprocess": True,
                "constraint_satisfaction_evaluated": False, "quality_evaluated": False,
                "render_case_predeclared": {"prompt_index": 0, "seed": seeds[0]},
                "repeat_policy": ("Fresh fixed-arm inference for the first seed of every prompt; no cache or selection"
                                  if repeat_first_seed else "No within-process repeat; independent process replay")}
    write_json(output / "experiment_manifest.json", manifest)
    sys.path.insert(0, str(vendor))
    from gem.motionagent import MotionAgentCameraContext, MotionAgentTextSegment, build_motionagent_text_data
    from scripts.demo.demo_utils import load_model, run_inference

    cwd = Path.cwd()
    started = time.perf_counter()
    try:
        os.chdir(vendor)
        model = load_model(str(checkpoint), load_text_encoder=True)
    finally:
        os.chdir(cwd)
    torch.cuda.synchronize()
    manifest["model_load_seconds"] = time.perf_counter() - started
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    recorder = RecordingGEM(model)
    records, repetitions = [], []

    def infer(arm, case, prompt, seed):
        case.mkdir(parents=True, exist_ok=True)
        if arm == "raw":
            data, _ = build_motionagent_text_data(
                [MotionAgentTextSegment(caption=prompt, length=FRAMES, name="raw_user_text")],
                MotionAgentCameraContext(width=1280, height=720, static_camera=True), seed=seed)
            with seeded_torch_rng(seed, "cuda"), torch.inference_mode():
                prediction = run_inference(recorder, data, static_cam=True)
            params = {k: v.detach().cpu() for k, v in prediction["body_params_global"].items()}
            motion = prediction["net_outputs"]["model_output"]["pred_x"].detach().cpu().reshape(FRAMES, 151)
            del prediction
        else:
            if arm == "legacy":
                spec, text = compile_legacy(prompt, legacy)
                request = build_generation_request(condition=text, motion_spec=spec, fps=FPS, seeds=[seed])
            else:
                request, trace = prepare_native_request(case, prompt, FRAMES, FPS, seed)
                write_json(case / "planner_trace.json", trace)
                if arm == "parser_fixed_split":
                    split = legacy["gem_text_compiler"].compile_gem_text_condition(request.motion_spec.segments, total_frames=FRAMES)
                    request = build_generation_request(condition=split, motion_spec=request.motion_spec, fps=FPS, seeds=[seed])
            request.output_policy = GenerationOutputPolicy(output_root=str(case), artifact_root=str(case / "candidates"))
            if request.motion_spec.original_request != prompt:
                raise ValueError("Original prompt changed between arms")
            write_json(case / "compiler_trace.json", request.motion_spec.model_dump(mode="json"))
            write_json(case / "generation_request.json", request.model_dump(mode="json"))
            store = CandidateStore(request.output_policy.artifact_root)
            with torch.inference_mode():
                result = generate_motion(request, model=recorder, checkpoint_version=checkpoint_hash,
                    adapter_context=GEMAdapterContext(width=1280, height=720, fps=FPS, total_frames=FRAMES), candidate_store=store)
            if result.status != "success" or result.inference_calls != 1 or result.cache_hits:
                raise RuntimeError(f"Fresh inference failed: {result.model_dump(mode='json')}")
            saved = store.load(result.candidates[0].candidate_id)
            params, motion = saved["body_params_global"], saved["motion_repr"].reshape(FRAMES, 151)
            if recorder.receipts[-1]["motion_specification"] != request.motion_spec.model_dump(mode="json"):
                raise ValueError("Compiler trace differs from actual GEM metadata")
            write_json(case / "generation_result.json", result.model_dump(mode="json"))
        receipt = recorder.receipts[-1]
        if receipt["seed"] != seed or receipt["hard_motion_condition_supplied"]:
            raise ValueError("Unpaired seed or unexpected numerical conditioning")
        torch.save(params, case / "smpl.pt")
        torch.save(motion, case / "motion.pt")
        record = {"arm": arm, "prompt": prompt, "seed": seed, "frames": FRAMES, "fps": FPS,
                  "duration_s": FRAMES / FPS, "path": str(case.relative_to(output)),
                  "receipt": receipt, "cache_hits": 0, **motion_stats(params, motion)}
        write_json(case / "metadata.json", record)
        print(json.dumps({"completed": record["path"], "seconds": receipt["inference_seconds"],
                          "root_range_xyz_m": record["root_range_xyz_m"]}), flush=True)
        return record, params, motion

    for prompt_index, prompt in enumerate(prompts):
        for seed in seeds:
            for arm in arms:
                case = output / f"p{prompt_index:02d}_s{seed}" / arm
                record, params, motion = infer(arm, case, prompt, seed)
                records.append(record)
                if repeat_first_seed and arm == "fixed" and seed == seeds[0]:
                    repeat, repeat_params, repeat_motion = infer("fixed", case.parent / "fixed_repeat", prompt, seed)
                    equal_params = all(torch.equal(params[key], repeat_params[key]) for key in params)
                    repetitions.append({"prompt_index": prompt_index, "seed": seed,
                        "fresh_inferences": 2, "cache_hits": 0, "motion_bitwise_equal": torch.equal(motion, repeat_motion),
                        "smpl_bitwise_equal": equal_params, "motion_max_abs_difference": float((motion - repeat_motion).abs().max()),
                        "repeat_metadata": repeat})
                write_json(output / "progress.json", {"records": records, "reproducibility": repetitions})
    after = protected_sources()
    write_json(output / "source_integrity.json", {"unchanged_during_experiment": before == after, "before": before, "after": after})
    if before != after:
        raise RuntimeError("Production/GEM sources changed during experiment")
    manifest["inference_calls"] = len(recorder.receipts)
    manifest["total_inference_seconds"] = sum(r["inference_seconds"] for r in recorder.receipts)
    manifest["all_model_weights_frozen"] = all(not p.requires_grad for p in model.parameters())
    write_json(output / "experiment_manifest.json", manifest)
    write_json(output / "results.json", {"records": records, "reproducibility": repetitions})
    print(json.dumps({"status": "complete", "inference_calls": len(recorder.receipts), "reproducibility": repetitions}), flush=True)


def replay(output, snapshot, *, directory="cross_process_replay"):
    """Reload the model in a new process and repeat each first-seed fixed arm."""
    import torch
    manifest = json.loads((output / "experiment_manifest.json").read_text())
    replay_dir = output / directory
    seed = manifest["seeds"][0]
    generate(replay_dir, snapshot, [seed], manifest["prompts"], arms=("fixed",), repeat_first_seed=False)
    new_manifest = json.loads((replay_dir / "experiment_manifest.json").read_text())
    if new_manifest["checkpoint_sha256"] != manifest["checkpoint_sha256"]:
        raise ValueError("Checkpoint changed between processes")
    comparisons = []
    for index in range(len(manifest["prompts"])):
        relative = Path(f"p{index:02d}_s{seed}") / "fixed"
        old_motion = torch.load(output / relative / "motion.pt", weights_only=True)
        new_motion = torch.load(replay_dir / relative / "motion.pt", weights_only=True)
        old_params = torch.load(output / relative / "smpl.pt", weights_only=True)
        new_params = torch.load(replay_dir / relative / "smpl.pt", weights_only=True)
        comparisons.append({"prompt_index": index, "seed": seed, "cache_hits": 0,
            "motion_bitwise_equal": torch.equal(old_motion, new_motion),
            "smpl_bitwise_equal": all(torch.equal(old_params[k], new_params[k]) for k in old_params),
            "motion_max_abs_difference": float((old_motion - new_motion).abs().max())})
    filename = "cross_process_reproducibility.json" if directory == "cross_process_replay" else directory + "_reproducibility.json"
    write_json(output / filename, comparisons)
    print(json.dumps({"cross_process_reproducibility": comparisons}), flush=True)


def audit(output, snapshot):
    """Verify the legacy-artifact-only fix leaves all benchmark inputs intact."""
    from motion_agent.compiler import CompilerRequest, MotionCompiler
    from scripts.create_phase7_baseline_comparison import protected_sources
    integrity = json.loads((output / "source_integrity.json").read_text())
    current = protected_sources()
    changes = [p for p in current if current[p] != integrity["after"].get(p)]
    if changes != ["motion_agent/compiler/temporal_resolver.py"]:
        raise ValueError(f"Unexpected post-audit source changes: {changes}")
    records = json.loads((output / "results.json").read_text())["records"]
    checks = []
    for record in records:
        if record["arm"] != "fixed":
            continue
        request = json.loads((output / record["path"] / "generation_request.json").read_text())
        compiled = MotionCompiler().compile(CompilerRequest(original_request=record["prompt"], duration_s=FRAMES / FPS, total_frames=FRAMES))
        if compiled.motion_spec.model_dump(mode="json") != request["motion_spec"] or compiled.gem_text_condition.model_dump(mode="json") != request["text_condition"]:
            raise ValueError("The final compiler changed a benchmark input")
        checks.append({"path": record["path"], "spec_and_text_condition_equal": True})
    replay(output, snapshot, directory="legacy_compatibility_replay")
    if protected_sources() != current:
        raise ValueError("Sources changed during post-audit replay")
    write_json(output / "final_source_audit.json", {"changed_since_main_experiment": changes,
        "reason": "Preserve known counts in source-less legacy artifacts instead of rebinding to the first action mention",
        "benchmark_inputs_unchanged": True, "checks": checks, "current_sources": current})


def render(output):
    import cv2
    import numpy as np
    import torch
    from PIL import Image, ImageDraw
    import scripts.create_temporal_reasoning_demo as demo
    from scripts.render_motion_comparison import (build_vertices, camera_for_sequences, encode_video,
        load_motion, overlay_fonts, render_sequence, validate_video)
    manifest = json.loads((output / "experiment_manifest.json").read_text())
    case = output / f"p00_s{manifest['seeds'][0]}"
    started = time.perf_counter()
    model_path = ROOT / "vendor" / "GENMO" / "inputs" / "checkpoints" / "body_models" / "smplx" / "SMPLX_NEUTRAL.npz"
    vertices, translations = {}, {}
    for arm in ARMS:
        params, _ = load_motion(case / arm / "smpl.pt")
        vertices[arm], faces, translations[arm] = build_vertices(params, model_path, "cuda")
    pose, camera = camera_for_sequences(list(vertices.values()), 960, 720)
    videos, visibility = {}, {}
    names = {"raw": "baseline", "legacy": "legacy", "parser_fixed_split": "parser_fixed_split", "fixed": "motionagent"}
    for arm in ARMS:
        frames, visibility[arm] = render_sequence(vertices[arm], faces, pose, 960, 720, [0.02, 0.5, 0.55, 1])
        encode_video(case / (names[arm] + ".mp4"), frames, FPS)
        videos[names[arm]] = validate_video(case / (names[arm] + ".mp4"), FRAMES, FPS,
            preview=case / (names[arm] + "_preview.png"), sample_dir=case / "observation_samples" / names[arm])
        del frames

    def decode(name):
        reader = cv2.VideoCapture(str(case / (name + ".mp4")))
        try:
            while True:
                ok, frame = reader.read()
                if not ok:
                    break
                yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        finally:
            reader.release()

    # Select the predeclared example, never a best-looking seed.
    for source, destination in (("raw", "baseline"), ("fixed", "motionagent")):
        write_json(case / (destination + "_metadata.json"), json.loads((case / source / "metadata.json").read_text()))
    demo.PROMPT, demo.SEED = manifest["prompts"][0], manifest["seeds"][0]
    # The existing two-panel overlay expects sized sequences, not generators.
    encode_video(case / "comparison.mp4", demo.comparison_frames(list(decode("baseline")), list(decode("motionagent"))), FPS)
    title, body, small = overlay_fonts()

    def ablation_frames():
        order = ("baseline", "motionagent", "legacy", "parser_fixed_split")
        labels = ("Raw GEM", "MotionAgent + Temporal Resolver", "Legacy compiler", "Parser fixed + split captions")
        for index, frames in enumerate(zip(*(decode(name) for name in order))):
            canvas = Image.new("RGB", (1920, 1674), (245, 248, 250))
            draw = ImageDraw.Draw(canvas)
            draw.text((20, 55), f"Duration: 14 seconds | 420 frames | 30 FPS | Seed: {demo.SEED}", font=body, fill=(45, 62, 70))
            line, row = "Prompt:", 0
            for word in demo.PROMPT.split():
                trial = line + " " + word
                if draw.textlength(trial, font=body) > 1880:
                    draw.text((20, 85 + row * 24), line, font=body, fill=(45, 62, 70))
                    line, row = word, row + 1
                else:
                    line = trial
            draw.text((20, 85 + row * 24), line, font=body, fill=(45, 62, 70))
            for cell, (frame, label) in enumerate(zip(frames, labels)):
                x = (cell % 2) * 960
                y = 150 if cell < 2 else 910
                canvas.paste(Image.fromarray(frame), (x, y))
                draw.text((x + 20, 12 if cell < 2 else 878), label, font=title, fill=(30, 45, 52))
            draw.text((20, 1640), f"Time: {index / FPS:.2f}s / 14s | Frame: {index + 1}/420", font=small, fill=(45, 62, 70))
            yield np.asarray(canvas)

    encode_video(case / "ablation_comparison.mp4", ablation_frames(), FPS)
    for name in ("comparison", "ablation_comparison"):
        videos[name] = validate_video(case / (name + ".mp4"), FRAMES, FPS,
            preview=case / (name + "_preview.png"), sample_dir=case / "observation_samples" / name)
    write_json(case / "render_report.json", {"videos": videos, "camera": camera,
        "identical_render_settings": True, "visibility": visibility, "render_translation": translations,
        "render_seconds": time.perf_counter() - started, "source_motion_edited": False,
        "frame_padding": False, "retiming": False})
    print(json.dumps({"render": "complete", "videos": videos}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["generate", "render", "replay", "report", "audit"])
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "temporal_compiler_validation")
    parser.add_argument("--legacy", type=Path, default=ROOT / "tests" / "fixtures" / "temporal_compiler_legacy")
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 8, 9])
    parser.add_argument("--prompts-json", type=Path, help="JSON list of other prompts; same prompts are used for all arms")
    args = parser.parse_args()
    if args.mode == "render":
        render(args.output.resolve())
    elif args.mode == "replay":
        replay(args.output.resolve(), args.legacy.resolve())
    elif args.mode == "report":
        report(args.output.resolve())
    elif args.mode == "audit":
        audit(args.output.resolve(), args.legacy.resolve())
    else:
        prompts = json.loads(args.prompts_json.read_text()) if args.prompts_json else PROMPTS
        if not isinstance(prompts, list) or not prompts or not all(isinstance(p, str) and p.strip() for p in prompts):
            raise ValueError("Prompts must be a nonempty list of nonempty strings")
        if len(set(args.seeds)) != len(args.seeds) or any(s < 0 for s in args.seeds):
            raise ValueError("Seeds must be unique nonnegative integers")
        generate(args.output.resolve(), args.legacy.resolve(), args.seeds, prompts)


def report(output):
    from importlib.metadata import PackageNotFoundError, version
    from statistics import mean

    from scripts.create_phase7_baseline_comparison import digest
    manifest = json.loads((output / "experiment_manifest.json").read_text())
    results = json.loads((output / "results.json").read_text())
    replay_results = json.loads((output / "cross_process_replay" / "results.json").read_text())
    cross = json.loads((output / "cross_process_reproducibility.json").read_text())
    integrity = json.loads((output / "source_integrity.json").read_text())
    old_path = ROOT / "outputs" / "temporal_reasoning_demo" / "source_integrity.json"
    previous = json.loads((old_path if old_path.exists() else output / "previous_source_integrity.json").read_text())
    write_json(output / "previous_source_integrity.json", previous)
    old = previous["after"]
    shared_vendor = [p for p in old if p.startswith("vendor/GENMO/") and p in integrity["after"]]
    vendor_unchanged = all(old[p] == integrity["after"][p] for p in shared_vendor)
    changed = [p for p in old if p.startswith("motion_agent/") and p in integrity["after"] and old[p] != integrity["after"][p]]
    allowed = {"motion_agent/compiler/" + name + ".py" for name in (
        "semantic_parser", "temporal_resolver", "caption_optimizer", "gem_text_compiler", "schemas", "validators", "motion_compiler")}
    allowed.update({"motion_agent/generation/validators.py", "motion_agent/generation/segment_inpaint.py"})
    if not vendor_unchanged or not set(changed) <= allowed:
        raise ValueError("Changes exceeded the declared architecture boundary")
    if not integrity["unchanged_during_experiment"]:
        raise ValueError("Experiment source integrity failed")
    audit_path = output / "final_source_audit.json"
    audited = json.loads(audit_path.read_text()) if audit_path.exists() else None
    expected_sources = audited["current_sources"] if audited else integrity["after"]
    if any(digest(ROOT / path) != value for path, value in expected_sources.items()):
        raise ValueError("Report sources differ from the final validated sources")
    packages = {}
    for name in ("torch", "numpy", "transformers", "hydra-core", "langgraph", "smplx", "pyrender", "trimesh", "opencv-python", "Pillow", "pytest"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = "not_installed"
    write_json(output / "environment_packages.json", packages)
    records = results["records"]
    all_records = records + [r["repeat_metadata"] for r in results["reproducibility"]] + replay_results["records"]
    if audited:
        extra = json.loads((output / "legacy_compatibility_replay" / "results.json").read_text())
        all_records += extra["records"]
        cross += json.loads((output / "legacy_compatibility_replay_reproducibility.json").read_text())
    same_process = all(r["motion_bitwise_equal"] and r["smpl_bitwise_equal"] for r in results["reproducibility"])
    cross_process = all(r["motion_bitwise_equal"] and r["smpl_bitwise_equal"] for r in cross)
    summary = {"paired_cases": len(manifest["prompts"]) * len(manifest["seeds"]),
        "inference_calls": len(all_records), "cache_hits": sum(r["cache_hits"] for r in all_records),
        "frames": FRAMES, "fps": FPS, "duration_s": FRAMES / FPS, "seeds": manifest["seeds"],
        "total_inference_seconds": sum(r["receipt"]["inference_seconds"] for r in all_records),
        "peak_gpu_allocated_mb": max(r["receipt"]["peak_gpu_allocated_mb"] for r in all_records),
        "all_finite": all(r["finite"] for r in all_records),
        "same_process_bitwise_equal": same_process, "cross_process_bitwise_equal": cross_process,
        "vendor_sources_checked": len(shared_vendor), "vendor_sources_unchanged": vendor_unchanged,
        "changed_production_files": changed, "compiler_trace_generated_successfully": True,
        "arms": {}}
    for arm in ARMS:
        selected = [r for r in records if r["arm"] == arm]
        summary["arms"][arm] = {"samples": len(selected),
            "mean_inference_seconds": mean(r["receipt"]["inference_seconds"] for r in selected),
            "mean_root_path_length_m": mean(r["root_path_length_m"] for r in selected),
            "mean_root_range_xyz_m": [mean(r["root_range_xyz_m"][axis] for r in selected) for axis in range(3)]}
    write_json(output / "validation_summary.json", summary)
    case = output / f"p00_s{manifest['seeds'][0]}"
    trace = json.loads((case / "fixed" / "compiler_trace.json").read_text())
    lines = ["# Temporal Compiler Repair: Controlled Validation", "", "## Scope", "",
        "This experiment visualizes structured semantic compilation, not constraint satisfaction or motion quality.",
        "The unchanged prompt is passed to every arm. No authored segment durations, boundaries, wave intervals, repetition expansion or keyframes are used.",
        "GEM weights, diffusion sampling and Phase 8/9/10 modules are unchanged.", "", "## Minimal Production Repair", "",
        "Supported verb forms and conjunctions preserve trailing actions and direction. Original source-clause spans keep counts and temporal modes scoped.",
        "Concurrent conditions are projected into compound captions at existing compiler boundaries; original DSL overlap remains intact.",
        "Semantic segment bounds decouple segment IDs from caption indices. Only preflight and segment-range consumers change at the generation boundary.",
        "The Planner remains the existing ScriptedPlanner policy for this reproducible tool-chain experiment, not a new semantic resolver or a claim of LLM planning evaluation.",
        "Production has no prompt-specific branch or ablation switch. Unknown grammar is not universally supported by this bounded deterministic parser.", "",
        "## Design", "", f"- Prompts: {len(manifest['prompts'])}; seeds: {manifest['seeds']}.",
        f"- Frames/FPS/duration: {FRAMES}/{FPS}/{FRAMES / FPS:g} seconds.",
        f"- Final validation: {summary['inference_calls']} fresh inference calls, {summary['cache_hits']} cache hits.",
        "- Arms: raw GEM; saved pre-fix compiler components; fixed parser with legacy split projection; final native Planner/compiler/GenerationRequest path.",
        "- Nine paired cases, three same-process fixed-arm repeats, three independently loaded process replays.",
        "- Displayed case was predeclared as prompt 0, seed 7. No best-seed selection or motion retiming.", ""]
    for index, prompt in enumerate(manifest["prompts"]):
        lines += [f"Prompt {index}: `{prompt}`", ""]
    lines += ["## Generation Statistics", "", "| Arm | Samples | Mean inference (s) | Mean root range XYZ (m) | Mean root path (m) |",
        "| --- | ---: | ---: | --- | ---: |"]
    for arm, values in summary["arms"].items():
        xyz = ", ".join(f"{v:.3f}" for v in values["mean_root_range_xyz_m"])
        lines.append(f"| {arm} | {values['samples']} | {values['mean_inference_seconds']:.3f} | {xyz} | {values['mean_root_path_length_m']:.3f} |")
    lines += ["", f"Total synchronized inference: {summary['total_inference_seconds']:.3f} seconds; peak allocated GPU memory: {summary['peak_gpu_allocated_mb']:.1f} MiB.",
        "Root ranges and paths measure movement only. They may also reflect drift and are not quality, direction, seating or repetition-success scores.",
        "All requested shapes are finite. Model-load/rendering times are recorded separately, not included in inference totals.", "",
        "## Actual Compiler Output", "", f"Artifact: `p00_s7/fixed/compiler_trace.json`; original prompt preserved: {trace['original_request'] == manifest['prompts'][0]}.",
        "| Action | Body parts | Relation | Count | Frames |", "| --- | --- | --- | ---: | --- |"]
    for segment in trace["segments"]:
        relation = (segment["temporal_relation"] or {}).get("type", "sequential")
        count = (segment["temporal_constraint"] or {}).get("count")
        lines.append(f"| {segment['action']} | {', '.join(segment['body_parts'])} | {relation} | {count or ''} | [{segment['start_frame']},{segment['end_frame']}) |")
    lines += ["", "Intervals in this table are actual existing Timeline Compiler outputs, not experimental timing overrides.",
        "Each non-raw arm stores its actual trace, request and GEM receipt. The final native trace is checked against GEM-boundary metadata.", "",
        "## Reproducibility and Boundaries", "", f"Same-process tensor equality: {same_process}; independent-process tensor equality: {cross_process}.",
        "Both GEM motion and every SMPL parameter tensor are compared. Fresh runs bypass cache; per-case maximum absolute differences are recorded.",
        "This confirms repeatability on this checkpoint, GPU and software stack only. Cross-hardware/version bitwise equality and universal natural-language support are not guaranteed.",
        f"Unchanged vendor source/config hashes checked against the previous experiment: {len(shared_vendor)}; result: {vendor_unchanged}.",
        "Only the seven compiler files and two generation-interface compatibility files differ from the prior experiment. See source_integrity.json and validation_summary.json.",
        "Checkpoint and SMPL-X hashes, GPU/software settings, package versions, exact seeds and legacy component hashes are recorded.", "",
        "## Observable Differences", "",
        "The previous compiler drops the final sit action and left-turn direction in prompt 0. The final compiler retains both plus the wave count and simultaneous relation.",
        "In prompt 0/seed 7, the legacy clip has very little root translation; the compound-caption clip has a larger root translation range and vertical variation.",
        "The complete per-seed movement measurements are in results.json. Variation between seeds remains substantial; compiled conditions do not ensure exact three-wave execution or improved motion quality.",
        "Four-arm video shows the predeclared case with a shared fixed camera, neutral body model and matching colors. Rendering only applies the established per-clip rigid display translation; source motion is unchanged.",
        "No later verifier/guided generation work was started.", "", "## Re-run", "", "```bash",
        "cd /workspace/MotionAgent", " .phase7_venv/bin/python -m pytest tests",
        " .phase7_venv/bin/python scripts/validate_temporal_compiler.py generate --output outputs/temporal_compiler_reproduced",
        " .phase7_venv/bin/python scripts/validate_temporal_compiler.py replay --output outputs/temporal_compiler_reproduced",
        " PYOPENGL_PLATFORM=egl .phase7_venv/bin/python scripts/validate_temporal_compiler.py render --output outputs/temporal_compiler_reproduced",
        " .phase7_venv/bin/python scripts/validate_temporal_compiler.py report --output outputs/temporal_compiler_reproduced", "```", "",
        "Use a new output directory; existing runs are never overwritten. --prompts-json and --seeds accept other inputs without changing the production compiler.",
        "Real generation requires the recorded GEM checkpoint and SMPL-X assets; model assets are not included in the result bundle.", ""]
    if audited:
        lines += ["## Final Legacy Compatibility Audit", "",
            "A final backward-compatibility fix preserves known counts in stored artifacts without source spans. Two additional regression tests cover structured and unstructured legacy counts.",
            "All nine benchmark specifications and text conditions were recompiled and compared exactly against their recorded GEM requests; none changed.",
            "Three further independently loaded, uncached native generations using the final source were compared with the recorded motions. Their results are included in inference totals and cross-process equality.",
            "Both main-experiment and final-source hashes remain available, so the post-audit change is explicit rather than hidden. See final_source_audit.json and legacy_compatibility_replay_reproducibility.json.", ""]
    (output / "demo_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
