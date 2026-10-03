from __future__ import annotations

import numpy as np
import json
import pytest
import torch

from scripts.render_motion_comparison import (
    active_segments, camera_for_sequences, comparison_frames, load_motion,
    overlay_fonts, read_metadata, segment_summary, simple_comparison_frames, trace_frames,
)
from scripts.create_phase7_baseline_comparison import DEMO_PROMPT, prepare_native_request
from motion_agent.compiler.motion_compiler import CompilerRequest, MotionCompiler


def params(frames=3):
    return {key: torch.zeros(frames, size) for key, size in {"body_pose": 63, "global_orient": 3, "transl": 3, "betas": 10}.items()}


def test_load_standalone_and_nested_smpl(tmp_path):
    for index, value in enumerate([params(), {"body_params_global": params()}]):
        path = tmp_path / f"motion_{index}.pt"
        torch.save(value, path)
        loaded, source = load_motion(path)
        assert source == path.resolve()
        assert loaded["body_pose"].shape == (3, 63)


def test_rejects_corrupt_smpl_values(tmp_path):
    values = params()
    values["transl"][0, 0] = float("nan")
    path = tmp_path / "corrupt.pt"
    torch.save(values, path)
    with pytest.raises(ValueError, match="Invalid transl"):
        load_motion(path)


def test_rejects_mismatched_sequence_lengths(tmp_path):
    values = params()
    values["body_pose"] = torch.zeros(4, 63)
    path = tmp_path / "mismatch.pt"
    torch.save(values, path)
    with pytest.raises(ValueError, match="inconsistent"):
        load_motion(path)


def test_metadata_without_motion_spec_keeps_seed_and_candidate(tmp_path):
    path = tmp_path / "metadata.json"
    path.write_text(json.dumps({"seed": 7, "generation_request": {"motion_spec": None}}), encoding="utf-8")
    loaded = read_metadata(path, tmp_path / "smpl_global.pt")
    assert loaded["seed"] == 7 and loaded["prompt"] == "Unspecified"
    assert loaded["candidate_id"] == tmp_path.name


def test_shared_camera_fits_both_complete_motion_bounds():
    points = np.array([[-3., 0., -2.], [4., 2.2, 6.], [1., 1., 0.]])
    first = points[None]
    second = (points + np.array([0.5, 0., 1.]))[None]
    pose, metadata = camera_for_sequences([first, second], 960, 720)
    assert metadata["shared_between_clips"] and metadata["fixed_for_every_frame"]
    tangent = np.tan(np.deg2rad(metadata["vertical_fov_degrees"]) / 2)
    for sequence in [first, second]:
        view = (sequence.reshape(-1, 3) - pose[:3, 3]) @ pose[:3, :3]
        assert (view[:, 2] < 0).all()
        assert (np.abs(view[:, 1] / -view[:, 2]) < tangent).all()
        assert (np.abs(view[:, 0] / -view[:, 2]) < tangent * 960 / 720).all()


def demo_trace():
    native = MotionCompiler().compile(CompilerRequest(original_request=DEMO_PROMPT, duration_s=14, total_frames=420))
    return native, {"motion_specification": native.motion_spec.model_dump(mode="json"), "prompt": DEMO_PROMPT}


@pytest.mark.parametrize("prompt,frames", [(DEMO_PROMPT, 420),
    ("A person walks forward slowly while continuing to wave their right hand twice.", 360)])
def test_actual_planner_compiler_output_forwarded_unchanged(tmp_path, prompt, frames):
    request, trace = prepare_native_request(tmp_path, prompt, frames, 30, 7)
    native = MotionCompiler().compile(CompilerRequest(original_request=prompt,
        duration_s=frames / 30, total_frames=frames))
    assert request.motion_spec.model_dump(mode="json") == native.motion_spec.model_dump(mode="json")
    assert request.text_condition.model_dump(mode="json") == native.gem_text_condition.model_dump(mode="json")
    assert trace["native_compiler_output"]["motion_spec"] == trace["motion_specification"]
    assert trace["native_compiler_output"]["gem_text_condition"] == trace["gem_text_condition"]
    assert trace["manual_timing_override"] is False
    assert trace["compiler_output_forwarded_unchanged"] is True
    assert "visualization_timing_preset" not in trace
    assert request.seeds == [7]


def test_trace_half_open_boundaries_come_from_actual_compiler():
    native, trace = demo_trace()
    for segment in native.motion_spec.segments:
        start_ids = [s["segment_id"] for s in active_segments(trace, segment.start_frame)]
        end_ids = [s["segment_id"] for s in active_segments(trace, segment.end_frame)]
        assert segment.segment_id in start_ids
        assert segment.segment_id not in end_ids
    assert active_segments(trace, -1) == active_segments(trace, native.motion_spec.total_frames) == []


@pytest.mark.parametrize("width,height", [(800, 480), (960, 720)])
def test_overlay_sizes_stable_and_metadata_fits(width, height):
    from PIL import Image, ImageDraw

    native, trace = demo_trace()
    spec = native.motion_spec
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frames = [frame, frame]
    comparisons = list(comparison_frames(frames, frames, trace, trace, trace, 30))
    overlays = list(trace_frames(frames, trace, 30))
    count = len(spec.segments)
    assert comparisons[0].shape == comparisons[1].shape == (height + 160 + max(280, 110 + 54 * count), width * 2, 3)
    assert overlays[0].shape == overlays[1].shape == (height + 120 + max(260, 130 + 54 * count), width, 3)
    draw = ImageDraw.Draw(Image.new("RGB", (width, height)))
    _, _, font = overlay_fonts()
    for segment in spec.model_dump(mode="json")["segments"]:
        assert draw.textlength(segment_summary(segment), font=font) <= width - 40


@pytest.mark.parametrize("width,height", [(800, 480), (960, 720)])
def test_simple_comparison_never_reads_action_stages_and_preserves_render_pixels(width, height):
    class Metadata(dict):
        def get(self, key, default=None):
            if key in {"motion_specification", "native_compiler_output", "visualization_timing_preset"}:
                raise AssertionError("Simple comparison must not interpret action timelines")
            return super().get(key, default)

    meta = Metadata(prompt=DEMO_PROMPT, seed=7, candidate_id="demo")
    left = np.full((height, width, 3), (60, 80, 100), dtype=np.uint8)
    right = np.full((height, width, 3), (120, 140, 160), dtype=np.uint8)
    output = list(simple_comparison_frames([left, left], [right, right], meta, meta, 30))
    assert len(output) == 2 and output[0].shape == output[1].shape
    header = output[0].shape[0] - height - 48
    np.testing.assert_array_equal(output[0][header:header + height, 4:width - 4], left[:, 4:width - 4])
    np.testing.assert_array_equal(output[0][header:header + height, width + 4:2 * width - 4], right[:, 4:width - 4])


def test_simple_comparison_rejects_frame_padding():
    frame = np.zeros((480, 800, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="equal nonempty"):
        list(simple_comparison_frames([frame], [frame, frame], {}, {}, 30))
