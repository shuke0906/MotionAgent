from pathlib import Path

import pytest

from motion_agent.compiler import CompilerRequest, MotionCompiler
from scripts.validate_temporal_compiler import ARMS, PROMPTS, compile_legacy, load_legacy


@pytest.mark.parametrize("prompt", PROMPTS)
def test_predeclared_ablation_inputs_compile_without_authored_timing(prompt):
    root = Path(__file__).resolve().parents[2]
    modules = load_legacy(root / "tests" / "fixtures" / "temporal_compiler_legacy")
    legacy, legacy_text = compile_legacy(prompt, modules)
    fixed = MotionCompiler().compile(CompilerRequest(original_request=prompt, duration_s=14, total_frames=420))
    assert legacy.original_request == fixed.motion_spec.original_request == prompt
    assert legacy.total_frames == fixed.motion_spec.total_frames == 420
    split = modules["gem_text_compiler"].compile_gem_text_condition(fixed.motion_spec.segments, total_frames=420)
    assert split.captions == [s.gem_caption for s in fixed.motion_spec.segments]
    assert split.window_start == [s.normalized_start for s in fixed.motion_spec.segments]
    assert fixed.gem_text_condition.bounds_for_segment(1) == (fixed.motion_spec.segments[1].start_frame,
                                                           fixed.motion_spec.segments[1].end_frame)
    assert ARMS == ("raw", "legacy", "parser_fixed_split", "fixed")
    assert legacy_text.window_start[0] == fixed.gem_text_condition.window_start[0] == 0
