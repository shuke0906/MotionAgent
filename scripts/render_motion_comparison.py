"""Render saved SMPL motions externally; never import generation modules."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import torch


def load_motion(path: Path):
    if path.is_dir():
        path = path / "smpl_global.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "body_params_global" in payload:
        payload = payload["body_params_global"]
    if not isinstance(payload, dict):
        raise ValueError("Expected saved global SMPL parameters")
    shapes = {"body_pose": 63, "global_orient": 3, "transl": 3, "betas": 10}
    params = {}
    for key, dimension in shapes.items():
        if key not in payload:
            raise ValueError(f"Missing SMPL field {key}")
        tensor = torch.as_tensor(payload[key]).detach().cpu().float()
        if tensor.ndim != 2 or tensor.shape[1] != dimension or not torch.isfinite(tensor).all():
            raise ValueError(f"Invalid {key}: expected finite [frames,{dimension}]")
        params[key] = tensor
    count = len(params["body_pose"])
    if count == 0 or any(len(v) != count for v in params.values()):
        raise ValueError("SMPL parameters have inconsistent or empty frame counts")
    return params, path.resolve()


def read_metadata(path, motion):
    path = path or motion.parent / "metadata.json"
    if not path.exists():
        return {"candidate_id": motion.parent.name, "seed": None, "prompt": "Unspecified"}
    data = json.loads(path.read_text(encoding="utf-8"))
    data.setdefault("candidate_id", motion.parent.name)
    request = data.get("generation_request") or {}
    spec = request.get("motion_spec") or {}
    data.setdefault("prompt", spec.get("original_request", "Unspecified"))
    return data


def build_vertices(params, model_path, device):
    import smplx

    model = smplx.SMPLX(str(model_path), gender="neutral", num_betas=10,
        num_pca_comps=12, flat_hand_mean=False, batch_size=1).to(device).eval()
    vertices = []
    roots = []
    with torch.inference_mode():
        for start in range(0, len(params["body_pose"]), 24):
            batch = {k: v[start:start + 24].to(device) for k, v in params.items()}
            count = len(batch["body_pose"])
            zeros = torch.zeros(count, 3, device=device)
            output = model(**batch, left_hand_pose=torch.zeros(count, 12, device=device),
                right_hand_pose=torch.zeros(count, 12, device=device), jaw_pose=zeros,
                leye_pose=zeros, reye_pose=zeros, expression=torch.zeros(count, 10, device=device))
            vertices.append(output.vertices.cpu().numpy())
            roots.append(output.joints[:, 0].cpu().numpy())
    verts = np.concatenate(vertices)
    root = np.concatenate(roots)[0].copy()
    root[1] = verts[:, :, 1].min()
    # One rigid translation per clip: preserve all motion and orientation changes.
    verts -= root[None, None, :]
    if not np.isfinite(verts).all():
        raise ValueError("SMPL-X produced nonfinite vertices")
    return verts, model.faces.copy(), root.tolist()


def camera_for_sequences(sequences, width, height):
    low = np.min([v.min(axis=(0, 1)) for v in sequences], axis=0)
    high = np.max([v.max(axis=(0, 1)) for v in sequences], axis=0)
    target = (low + high) / 2
    backward = np.array([1.2, 0.5, 1.8], dtype=float)
    backward /= np.linalg.norm(backward)
    right = np.cross(np.array([0., 1., 0.]), backward)
    right /= np.linalg.norm(right)
    up = np.cross(backward, right)
    corners = np.array([[x, y, z] for x in [low[0], high[0]] for y in [low[1], high[1]] for z in [low[2], high[2]]]) - target
    x, y, z = corners @ right, corners @ up, corners @ backward
    tangent = math.tan(math.radians(40) / 2)
    distance = max(np.max(np.abs(y) / tangent + z), np.max(np.abs(x) / (tangent * width / height) + z)) * 1.2
    pose = np.eye(4)
    pose[:3, :3] = np.stack([right, up, backward], axis=1)
    pose[:3, 3] = target + backward * distance
    return pose, {"pose": pose.tolist(), "target": target.tolist(), "vertical_fov_degrees": 40,
        "shared_between_clips": True, "fixed_for_every_frame": True,
        "bounds": {"minimum": low.tolist(), "maximum": high.tolist()}}


def render_sequence(vertices, faces, pose, width, height, color):
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    import pyglet
    pyglet.options["shadow_window"] = False
    import pyrender
    import trimesh

    scene = pyrender.Scene(bg_color=[0.95, 0.97, 0.98, 1.0], ambient_light=[0.5, 0.5, 0.5])
    scene.add(pyrender.PerspectiveCamera(yfov=math.radians(40), znear=0.01, zfar=1000), pose=pose)
    scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=2.0), pose=pose)
    fill = pose.copy()
    fill[:3, 3] *= -1
    scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=1.0), pose=fill)
    ground = trimesh.creation.box(extents=[80, 0.015, 80])
    ground.apply_translation([0, -0.02, 0])
    ground_material = pyrender.MetallicRoughnessMaterial(baseColorFactor=[0.85, 0.88, 0.9, 1], metallicFactor=0, roughnessFactor=1)
    scene.add(pyrender.Mesh.from_trimesh(ground, material=ground_material))
    material = pyrender.MetallicRoughnessMaterial(baseColorFactor=color, metallicFactor=0, roughnessFactor=0.8)
    renderer = pyrender.OffscreenRenderer(width, height)
    frames = []
    visibility = []
    node = None
    try:
        for index, points in enumerate(vertices):
            if node is not None:
                scene.remove_node(node)
            node = scene.add(pyrender.Mesh.from_trimesh(trimesh.Trimesh(vertices=points, faces=faces, process=False), material=material, smooth=True))
            frame, _ = renderer.render(scene)
            # Body colors have chroma; the background and ground are nearly neutral.
            chroma = frame.max(axis=2).astype(int) - frame.min(axis=2).astype(int)
            mask = chroma > 45
            yy, xx = np.where(mask)
            if len(xx) < 250:
                raise RuntimeError(f"Motion not visible in frame {index}")
            visibility.append({"pixels": len(xx), "margin": int(min(xx.min(), yy.min(), width - 1 - xx.max(), height - 1 - yy.max()))})
            frames.append(frame)
            if index % 30 == 0:
                print(f"render frame {index}/{len(vertices)}", flush=True)
    finally:
        renderer.delete()
    return frames, {"minimum_body_pixels": int(min(v["pixels"] for v in visibility)),
        "minimum_frame_margin_pixels": min(v["margin"] for v in visibility),
        "motion_visible_every_frame": True, "unclipped_every_frame": all(v["margin"] > 2 for v in visibility)}


def encode_video(path, frames, fps):
    first = frames[0] if isinstance(frames, list) else next(frames)
    height, width = first.shape[:2]
    command = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pixel_format", "rgb24",
        "-video_size", f"{width}x{height}", "-framerate", str(fps), "-i", "pipe:0", "-an",
        "-c:v", "libx264", "-crf", "18", "-preset", "fast", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        if isinstance(frames, list):
            iterator = iter(frames)
        else:
            process.stdin.write(np.ascontiguousarray(first).tobytes())
            iterator = frames
        for frame in iterator:
            process.stdin.write(np.ascontiguousarray(frame).tobytes())
    finally:
        process.stdin.close()
        error = process.stderr.read().decode("utf-8", errors="replace")
        code = process.wait()
    if code:
        raise RuntimeError(f"ffmpeg failed: {error}")


def validate_video(path, expected_frames, fps, preview=None, sample_dir=None):
    import cv2

    reader = cv2.VideoCapture(str(path))
    if not reader.isOpened():
        raise ValueError(f"Cannot decode {path}")
    actual_fps = reader.get(cv2.CAP_PROP_FPS)
    sample_indices = set(np.linspace(0, expected_frames - 1, min(6, expected_frames), dtype=int).tolist())
    if sample_dir is not None:
        sample_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    shape = None
    signatures = set()
    while True:
        success, frame = reader.read()
        if not success:
            break
        if shape is not None and shape != frame.shape:
            raise ValueError("Video contains inconsistent frames")
        shape = frame.shape
        signatures.add(hashlib.sha256(frame.tobytes()).hexdigest())
        if preview is not None and count == expected_frames // 2:
            cv2.imwrite(str(preview), frame)
        if sample_dir is not None and count in sample_indices:
            if not cv2.imwrite(str(sample_dir / f"frame_{count:04d}.png"), frame):
                raise RuntimeError(f"Could not save observation frame {count}")
        count += 1
    reader.release()
    if count != expected_frames or abs(actual_fps - fps) > 0.01:
        raise ValueError(f"Video timing mismatch: {count} frames at {actual_fps} fps")
    if len(signatures) < 2:
        raise ValueError("Video has no changing frames")
    return {"frames": count, "fps": actual_fps, "duration": count / actual_fps,
        "resolution": [shape[1], shape[0]], "all_frames_decoded": True, "unique_frames": len(signatures),
        "output_video": str(path), "size_bytes": path.stat().st_size}


def active_segments(trace, frame):
    return [segment for segment in trace.get("motion_specification", {}).get("segments", [])
        if segment["start_frame"] <= frame < segment["end_frame"]]


def segment_summary(segment):
    parts = ", ".join(segment.get("body_parts", [])) or "unspecified"
    repeat = segment.get("repetition")
    detail = f"{segment['action'].title()} [{segment['start_frame']},{segment['end_frame']}) | Body part: {parts}"
    if repeat is not None:
        detail += f" | Repetition: {repeat} requested"
    return detail


def overlay_fonts():
    from PIL import ImageFont

    candidates = [Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("C:/Windows/Fonts/arial.ttf")]
    path = next((p for p in candidates if p.exists()), None)
    def font(size):
        return ImageFont.truetype(str(path), size) if path else ImageFont.load_default(size=size)
    return font(28), font(18), font(16)


def draw_timeline(draw, segments, frame, total, x, y, width, font):
    label_width = 100
    bar_x, bar_width = x + label_width, width - label_width
    for row, segment in enumerate(segments):
        top = y + row * 28
        draw.text((x, top), segment["action"].title(), font=font, fill=(45, 62, 70))
        draw.rectangle((bar_x, top + 3, bar_x + bar_width, top + 18), fill=(218, 225, 229))
        start = bar_x + bar_width * segment["start_frame"] / total
        end = bar_x + bar_width * segment["end_frame"] / total
        active = segment["start_frame"] <= frame < segment["end_frame"]
        draw.rectangle((start, top + 3, end, top + 18), fill=(0, 128, 108) if active else (151, 185, 179))
    playhead = bar_x + bar_width * frame / total
    draw.line((playhead, y - 3, playhead, y + max(1, len(segments)) * 28 - 5), fill=(180, 42, 55), width=3)
    bottom = y + len(segments) * 28 + 3
    draw.text((bar_x, bottom), "0", font=font, fill=(65, 80, 87))
    draw.text((bar_x + bar_width - 32, bottom), str(total), font=font, fill=(65, 80, 87))


def simple_comparison_frames(left, right, metadata_a, metadata_b, fps):
    """Display generated motion without interpreting or manufacturing action stages."""
    from PIL import Image, ImageDraw

    if len(left) != len(right) or not left:
        raise ValueError("Current-behavior comparison requires equal nonempty sequences")
    width, height = left[0].shape[1], left[0].shape[0]
    title_font, body_font, small_font = overlay_fonts()
    measure = ImageDraw.Draw(Image.new("RGB", (2 * width, 1)))
    prompt_lines, line = [], ""
    for word in ("Prompt: " + metadata_a.get("prompt", "Unspecified")).split():
        attempt = (line + " " + word).strip()
        if line and measure.textlength(attempt, font=small_font) > 2 * width - 40:
            prompt_lines.append(line)
            line = word
        else:
            line = attempt
    prompt_lines.append(line)
    header = 150 + 22 * len(prompt_lines)
    total = len(left)
    for index, (baseline, agent) in enumerate(zip(left, right)):
        if baseline.shape != left[0].shape or agent.shape != baseline.shape:
            raise ValueError("Comparison render frames must have matching dimensions")
        canvas = Image.new("RGB", (2 * width, height + header + 48), (245, 248, 250))
        canvas.paste(Image.fromarray(baseline), (0, header))
        canvas.paste(Image.fromarray(agent), (width, header))
        draw = ImageDraw.Draw(canvas)
        for x, label, meta in [(20, "Baseline GEM", metadata_b), (width + 20, "MotionAgent", metadata_a)]:
            draw.text((x, 12), label, font=title_font, fill=(30, 45, 52))
            draw.text((x, 48), f"Candidate: {meta.get('candidate_id', 'unknown')} | Seed: {meta.get('seed')}",
                font=small_font, fill=(45, 62, 70))
        draw.text((20, 73), f"{total} frames | {fps} FPS | {total / fps:g} seconds | Same input prompt and frozen GEM checkpoint",
            font=small_font, fill=(65, 80, 87))
        for row, line in enumerate(prompt_lines):
            draw.text((20, 99 + row * 22), line, font=small_font, fill=(45, 62, 70))
        draw.text((20, header - 25), "Current generation behavior. No demo-authored action timing. Generated body appearances are not identity-matched.",
            font=small_font, fill=(65, 80, 87))
        draw.text((20, header + height + 12), f"Frame {index:03d}/{total} | Time {index / fps:.2f}/{total / fps:.2f}s",
            font=body_font, fill=(45, 62, 70))
        draw.text((width + 20, header + height + 12), "No claim of exact repetition, constraint or keyframe satisfaction.",
            font=small_font, fill=(65, 80, 87))
        draw.line((width, header, width, header + height), fill=(180, 190, 196), width=2)
        yield np.asarray(canvas)


def overlap_description(segments):
    names = {s["segment_id"]: s["action"] for s in segments}
    pairs = [f"{s['action']} + {names.get(s['simultaneous_with'], 'unknown')}"
        for s in segments if s.get("simultaneous_with") is not None]
    return "Overlap: " + (", ".join(pairs) if pairs else "none specified")


def comparison_frames(left, right, metadata_a, metadata_b, trace, fps):
    from PIL import Image, ImageDraw

    width, height = left[0].shape[1], left[0].shape[0]
    title_font, body_font, small_font = overlay_fonts()
    segments = trace.get("motion_specification", {}).get("segments", [])
    total = max(len(left), len(right))
    footer_height = max(280, 110 + 54 * len(segments))
    for index in range(total):
        canvas = Image.new("RGB", (width * 2, height + 160 + footer_height), (245, 248, 250))
        canvas.paste(Image.fromarray(left[min(index, len(left) - 1)]), (0, 160))
        canvas.paste(Image.fromarray(right[min(index, len(right) - 1)]), (width, 160))
        draw = ImageDraw.Draw(canvas)
        draw.text((20, 12), "Baseline GEM", font=title_font, fill=(30, 45, 52))
        draw.text((width + 20, 12), "MotionAgent", font=title_font, fill=(30, 45, 52))
        draw.text((20, 48), "Raw text conditioning", font=body_font, fill=(65, 80, 87))
        draw.text((width + 20, 48), "Planner + Compiler + Structured Generation", font=body_font, fill=(65, 80, 87))
        prompt = "Prompt: " + metadata_a.get("prompt", "Unspecified")
        lines = []
        line = ""
        for word in prompt.split():
            attempt = (line + " " + word).strip()
            if draw.textlength(attempt, font=small_font) > 2 * width - 40:
                lines.append(line)
                line = word
            else:
                line = attempt
        lines.append(line)
        for row, line in enumerate(lines):
            draw.text((20, 76 + row * 21), line, font=small_font, fill=(42, 55, 63))
        draw.text((20, 104), "Conditioning strategy comparison, not identity comparison. Generated body appearances are preserved.",
            font=small_font, fill=(65, 80, 87))
        for x, meta in [(20, metadata_b), (width + 20, metadata_a)]:
            draw.text((x, 132), f"Candidate: {meta.get('candidate_id', 'unknown')}   Seed: {meta.get('seed')}", font=small_font, fill=(45, 62, 70))
        y = height + 174
        draw.text((20, y), f"One raw caption | Frames: [0,{len(left)})", font=body_font, fill=(45, 62, 70))
        draw.text((20, y + 29), f"Timeline position: {index:03d}/{total} | {index / fps:.2f}/{total / fps:.2f}s", font=body_font, fill=(45, 62, 70))
        draw.text((20, y + 58), "Same prompt, seed, checkpoint and frame count", font=small_font, fill=(65, 80, 87))
        draw.text((20, y + 86), "No planner, compiler, constraints or keyframe tool", font=small_font, fill=(65, 80, 87))
        draw.text((width + 20, y), "Structured conditioning intent (not observed-action verification)", font=body_font, fill=(45, 62, 70))
        for row, segment in enumerate(segments):
            active = segment["start_frame"] <= index < segment["end_frame"]
            draw.text((width + 20, y + 28 + row * 26), segment_summary(segment), font=small_font,
                fill=(0, 105, 86) if active else (65, 80, 87))
        timeline_y = y + 36 + len(segments) * 26
        draw_timeline(draw, segments, index, len(right), width + 20, timeline_y, width - 40, small_font)
        current = " + ".join(s["action"].title() for s in active_segments(trace, index)) or "None"
        draw.text((width + 20, timeline_y + 28 * len(segments) + 26),
            f"Current segments: {current} | {overlap_description(segments)}", font=small_font, fill=(45, 62, 70))
        timing_note = ("Timing: external visualization preset; native compiler output retained in JSON."
            if trace.get("visualization_timing_preset") else "Timing: native compiler output, forwarded unchanged; no per-action demo overrides.")
        draw.text((20, canvas.height - 46), timing_note,
            font=small_font, fill=(65, 80, 87))
        draw.text((20, canvas.height - 24), "Semantic decomposition and structured conditioning only. Constraint satisfaction and motion quality are not evaluated.",
            font=small_font, fill=(65, 80, 87))
        draw.line([(width, 0), (width, canvas.height)], fill=(180, 190, 196), width=2)
        yield np.asarray(canvas)


def trace_frames(frames, trace, fps):
    from PIL import Image, ImageDraw

    width, height = frames[0].shape[1], frames[0].shape[0]
    title_font, body_font, small_font = overlay_fonts()
    segments = trace.get("motion_specification", {}).get("segments", [])
    footer_height = max(260, 130 + 54 * len(segments))
    for index, frame in enumerate(frames):
        canvas = Image.new("RGB", (width, height + 120 + footer_height), (245, 248, 250))
        canvas.paste(Image.fromarray(frame), (0, 120))
        draw = ImageDraw.Draw(canvas)
        draw.text((20, 12), "MotionAgent | Semantic Trace", font=title_font, fill=(30, 45, 52))
        draw.text((20, 49), "Conditioning intent, not measured constraint satisfaction", font=body_font, fill=(65, 80, 87))
        draw.text((20, 78), trace.get("prompt", ""), font=small_font, fill=(45, 62, 70))
        active = active_segments(trace, index)
        current = " + ".join(f"{s['segment_id']}: {s['action'].title()}" for s in active) or "None"
        y = height + 136
        draw.text((20, y), f"Current segment: {current}", font=body_font, fill=(0, 105, 86))
        for row, segment in enumerate(active):
            repeat = segment.get("repetition")
            detail = (f"Action: {segment['action']} | Body part: {', '.join(segment.get('body_parts', []))}"
                f" | Repetition: {repeat if repeat is not None else 'unspecified'}")
            draw.text((20, y + 28 + row * 26), detail, font=body_font, fill=(45, 62, 70))
        timeline_y = y + 34 + len(segments) * 26
        draw_timeline(draw, segments, index, len(frames), 20, timeline_y, width - 40, small_font)
        draw.text((20, timeline_y + len(segments) * 28 + 27),
            f"Timeline position: {index:03d}/{len(frames)} | {index / fps:.2f}/{len(frames) / fps:.2f}s | {overlap_description(segments)}",
            font=body_font, fill=(45, 62, 70))
        draw.text((20, canvas.height - 43), "Action counts and highlighted windows are conditioning metadata, not measured results.",
            font=small_font, fill=(65, 80, 87))
        timing_note = "Timing: visualization preset." if trace.get("visualization_timing_preset") else "Timing: unchanged compiler output."
        draw.text((20, canvas.height - 22), timing_note,
            font=small_font, fill=(65, 80, 87))
        yield np.asarray(canvas)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion_a", type=Path, required=True)
    parser.add_argument("--motion_b", type=Path, required=True)
    parser.add_argument("--smplx_model", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--metadata_a", type=Path)
    parser.add_argument("--metadata_b", type=Path)
    parser.add_argument("--fps", type=int, choices=[24, 30], default=30)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--generated_name", default="generated.mp4")
    parser.add_argument("--overlay_style", choices=["simple", "semantic"], default="simple")
    parser.add_argument("--trace_video", action="store_true", help="Also render actual conditioning metadata as a trace video")
    args = parser.parse_args()
    if args.width < 800 or args.height < 480 or args.width % 2 or args.height % 2:
        raise ValueError("Even dimensions >= 800x480 are required for legible comparison metadata")
    if Path(args.generated_name).name != args.generated_name or not args.generated_name.endswith(".mp4"):
        raise ValueError("generated_name must be an MP4 filename")
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg is required")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    params_a, path_a = load_motion(args.motion_a)
    params_b, path_b = load_motion(args.motion_b)
    metadata_a, metadata_b = read_metadata(args.metadata_a, path_a), read_metadata(args.metadata_b, path_b)
    verts_a, faces, offset_a = build_vertices(params_a, args.smplx_model, args.device)
    verts_b, _, offset_b = build_vertices(params_b, args.smplx_model, args.device)
    pose, camera = camera_for_sequences([verts_a, verts_b], args.width, args.height)
    frames_a, visible_a = render_sequence(verts_a, faces, pose, args.width, args.height, [0.02, 0.5, 0.55, 1])
    frames_b, visible_b = render_sequence(verts_b, faces, pose, args.width, args.height, [0.72, 0.24, 0.16, 1])
    generated_path = output / args.generated_name
    baseline_path = output / "baseline.mp4"
    comparison_path = output / "comparison.mp4"
    trace_path = output / "motionagent_trace.mp4"
    encode_video(generated_path, frames_a, args.fps)
    encode_video(baseline_path, frames_b, args.fps)
    comparison_source = (simple_comparison_frames(frames_b, frames_a, metadata_a, metadata_b, args.fps)
        if args.overlay_style == "simple" else comparison_frames(frames_b, frames_a, metadata_a, metadata_b, metadata_a, args.fps))
    encode_video(comparison_path, comparison_source, args.fps)
    if args.trace_video:
        encode_video(trace_path, trace_frames(frames_a, metadata_a, args.fps), args.fps)
    entries = []
    for label, path, params, metadata, visible, offset in [
        ("motionagent", generated_path, params_a, metadata_a, visible_a, offset_a),
        ("baseline", baseline_path, params_b, metadata_b, visible_b, offset_b),
    ]:
        entries.append({"label": label, "candidate_id": metadata.get("candidate_id"), "seed": metadata.get("seed"),
            "source_motion": str(path_a if label == "motionagent" else path_b),
            **validate_video(path, len(params["body_pose"]), args.fps), **visible, "render_translation": offset})
    comparison = validate_video(comparison_path, max(len(frames_a), len(frames_b)), args.fps, output / "comparison_preview.png")
    trace_video = (validate_video(trace_path, len(frames_a), args.fps, output / "motionagent_trace_preview.png")
        if args.trace_video else None)
    report = {"fps": args.fps, "frames": len(frames_a), "duration": len(frames_a) / args.fps,
        "candidate_id": metadata_a.get("candidate_id"), "seed": metadata_a.get("seed"), "source_motion": str(path_a),
        "output_video": str(generated_path), "rendering_method": "SMPL-X forward + trimesh + pyrender EGL + ffmpeg H.264",
        "device": args.device, "camera": camera, "videos": entries, "comparison": comparison, "trace_video": trace_video,
        "comparison_description": "Original raw-text GEM versus MotionAgent structured generation; no Phase 4 reference",
        "comparison_scope": "conditioning strategy, not identity", "body_identity_matched": False,
        "visualization_timing_preset": metadata_a.get("visualization_timing_preset"),
        "manual_timing_override": metadata_a.get("manual_timing_override", bool(metadata_a.get("visualization_timing_preset"))),
        "overlay_style": args.overlay_style,
        "constraint_satisfaction_evaluated": False,
        "frame_hold_policy": "none" if len(frames_a) == len(frames_b) else "Hold final frame only if input lengths differ",
        "semantic_quality_evaluated": False}
    (output / "render_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"render_report": str(output / "render_report.json"), "validation": "PASS"}), flush=True)


if __name__ == "__main__":
    main()
