import numpy as np
import pytest

from motion_agent.compiler import CompilerRequest, MotionCompiler
from scripts.create_phase7_baseline_comparison import prepare_native_request
from scripts.create_temporal_reasoning_demo import FRAMES, FPS, PROMPT, SEED, comparison_frames


def test_experiment_forwards_exact_prompt_and_native_temporal_spec(tmp_path):
    request, trace = prepare_native_request(tmp_path, PROMPT, FRAMES, FPS, SEED)
    native = MotionCompiler().compile(CompilerRequest(
        original_request=PROMPT, duration_s=FRAMES / FPS, total_frames=FRAMES, fps=FPS))
    assert request.motion_spec == native.motion_spec
    assert request.text_condition == native.gem_text_condition
    assert request.motion_spec.original_request == PROMPT
    assert request.seeds == [SEED]
    wave = next(s for s in request.motion_spec.segments if s.action == "wave")
    assert wave.temporal_constraint.count == 3
    assert "right_hand" in wave.body_parts
    assert wave.simultaneous_with is not None
    assert trace["compiler_output_forwarded_unchanged"]
    assert not request.condition_bundle.hard_motion_condition_handle
    assert not request.condition_bundle.keyframe_specs


def test_comparison_preserves_pixels_without_using_action_windows():
    baseline = np.full((720, 960, 3), 25, dtype=np.uint8)
    agent = np.full((720, 960, 3), 200, dtype=np.uint8)
    result = next(comparison_frames([baseline] * FRAMES, [agent] * FRAMES))
    assert result.shape == (914, 1920, 3)
    assert np.array_equal(result[150:870, :959], baseline[:, :959])
    assert np.array_equal(result[150:870, 962:], agent[:, 2:])
    with pytest.raises(ValueError, match="exactly"):
        next(comparison_frames([baseline], [agent]))
