from __future__ import annotations

import pytest

from motion_agent.compiler import CompilerRequest, MotionCompiler
from motion_agent.compiler.schemas import GEMTextCondition
from motion_agent.compiler.validators import validate_motion_semantics, validate_request_coverage
from motion_agent.generation import build_generation_request
from motion_agent.generation.segment_inpaint import segment_frame_ranges
from motion_agent.generation.validators import validate_generation_request


def compile_prompt(prompt):
    return MotionCompiler().compile(CompilerRequest(original_request=prompt, duration_s=14, total_frames=420))


@pytest.mark.parametrize("verbs", [("walk", "wave", "turn", "sit"),
                                  ("walks", "waves", "turns", "sits"),
                                  ("walking", "waving", "turning", "sitting"),
                                  ("walked", "waved", "turned", "sat")])
@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("expression,count", [("once", 1), ("twice", 2), ("three times", 3), ("four times", 4)])
def test_composite_verb_body_and_count_matrix(verbs, side, expression, count):
    walk, wave, turn, sit = verbs
    prompt = f"A person {walk} forward while {wave} their {side} hand {expression}, then {turn} left and {sit} down."
    result = compile_prompt(prompt)
    first, repeated, turning, seated = result.motion_spec.segments
    assert [s.action for s in result.motion_spec.segments] == ["walk", "wave", "turn", "sit_down"]
    assert repeated.body_parts == [f"{side}_hand"]
    assert repeated.temporal_constraint.count == count
    assert repeated.temporal_constraint.source_text == expression
    assert repeated.temporal_relation.type == "simultaneous"
    assert repeated.temporal_relation.related_action == "walk"
    assert turning.direction == "left" and seated.end_frame == 420
    assert all(s.temporal_constraint is None for s in (first, turning, seated))
    assert result.gem_text_condition.window_start[1:] == result.gem_text_condition.window_end[:-1]
    assert len(result.gem_text_condition.captions) == 3
    assert "while waving" in result.gem_text_condition.captions[0]
    for segment in result.motion_spec.segments:
        assert prompt[segment.source_start:segment.source_end] == segment.source_text
    assert compile_prompt(prompt).model_dump(mode="json") == result.model_dump(mode="json")


@pytest.mark.parametrize("prompt,actions", [
    ("Run backward while raising your left arm twice, then jump once and sit down.", ["run", "raise", "jump", "sit_down"]),
    ("Jump twice, wave your left hand four times, then turn right and sit down.", ["jump", "wave", "turn", "sit_down"]),
    ("Wave your right hand twice then wave your left hand once.", ["wave", "wave"]),
    ("Wave your left hand repeatedly then sit down.", ["wave", "sit_down"]),
    ("Wave left and right hands alternately then turn counterclockwise.", ["wave", "turn"]),
])
def test_different_actions_and_connectors(prompt, actions):
    result = compile_prompt(prompt)
    assert [s.action for s in result.motion_spec.segments] == actions
    validate_request_coverage(prompt, result.motion_spec.segments)
    validate_motion_semantics(result.motion_spec, result.gem_text_condition)


def test_same_action_counts_are_clause_scoped():
    result = compile_prompt("Wave your right hand twice then wave your left hand once.")
    assert [s.temporal_constraint.count for s in result.motion_spec.segments] == [2, 1]


@pytest.mark.parametrize("continuous", ["keep walking forward", "walk continuously", "continuously walk forward"])
def test_continuous_mode_does_not_leak_to_next_action(continuous):
    result = compile_prompt(continuous + " then jump twice")
    first, second = result.motion_spec.segments
    assert first.temporal_mode == "continuous" and second.temporal_mode is None
    assert "continuously" in result.gem_text_condition.captions[0]
    assert second.temporal_constraint.count == 2


def test_alternation_is_scoped_and_both_hands_survive():
    result = compile_prompt("wave left and right hands alternately then sit down")
    first, second = result.motion_spec.segments
    assert first.temporal_relation.type == "alternating" and second.temporal_relation is None
    assert set(first.body_parts) == {"left_hand", "right_hand"}
    assert "left and right hands" in result.gem_text_condition.captions[0]


@pytest.mark.parametrize("missing", ["action", "count", "body", "direction"])
def test_coverage_rejects_dropped_source_semantics(missing):
    prompt = "Walk forward while waving your right hand three times, then turns left and sits down."
    segments = compile_prompt(prompt).motion_spec.segments
    if missing == "action":
        segments = segments[:-1]
    elif missing == "count":
        segments[1].temporal_constraint = None
    elif missing == "body":
        segments[1].body_parts = ["full_body"]
    else:
        segments[2].direction = None
    with pytest.raises(ValueError, match="missing"):
        validate_request_coverage(prompt, segments)


def test_compound_condition_preserves_semantic_ids_after_json_roundtrip():
    result = compile_prompt("walk forward while waving your right hand three times then turn left and sit down")
    condition = GEMTextCondition.model_validate_json(result.gem_text_condition.model_dump_json())
    assert len(condition.captions) == 3 and len(condition.segment_bounds) == 4
    request = build_generation_request(condition=condition, motion_spec=result.motion_spec, seed=7, fps=30,
                                       scope="segment", target_segments=[1, 3], previous_candidate_id="previous")
    assert validate_generation_request(request).status == "ready"
    assert segment_frame_ranges(request) == [condition.segment_bounds[1], condition.segment_bounds[3]]
    assert condition.segment_bounds[0] == condition.segment_bounds[1]
    request.target_segments = [4]
    assert "invalid_target_segment" in validate_generation_request(request).errors
    with pytest.raises(ValueError, match="unknown semantic"):
        segment_frame_ranges(request)


def test_legacy_condition_segment_indices_remain_supported():
    condition = GEMTextCondition(captions=["walk", "wave"], window_start=[0, 0.5], window_end=[0.5, 1], total_frames=180)
    assert condition.bounds_for_segment(1) == (90, 180)


def test_partial_overlap_preserves_native_timing_and_all_active_intent():
    prompt = "Walk forward for 2 seconds while continuing to walk, wave your right hand twice. Then turn left and sit down."
    result = compile_prompt(prompt)
    first, second, *_ = result.motion_spec.segments
    assert second.start_frame == 60 and first.start_frame == 0
    assert first.end_frame == second.end_frame
    condition = result.gem_text_condition
    assert condition.bounds_for_segment(0) == (first.start_frame, first.end_frame)
    assert condition.bounds_for_segment(1) == (second.start_frame, second.end_frame)
    assert "wave" not in condition.captions[0] and "while waving" in condition.captions[1]


def test_validator_rejects_a_caption_that_drops_a_concurrent_action():
    result = compile_prompt("walk forward while raising your left arm twice then sit down")
    result.gem_text_condition.captions[0] = "a person walks forward"
    with pytest.raises(ValueError, match="missing required"):
        validate_motion_semantics(result.motion_spec, result.gem_text_condition)


@pytest.mark.parametrize("expression", ["once", "twice", "three times", "four times", "repeatedly", "continuously"])
def test_unspecified_wave_limb_stays_consistent_with_existing_dsl_default(expression):
    result = compile_prompt("wave " + expression)
    assert result.motion_spec.segments[0].body_parts == ["right_arm"]
    assert "right arm" in result.gem_text_condition.captions[0]


@pytest.mark.parametrize("body", ["left hand", "right hand", "left arm", "right arm", "left and right hands", "left and right arms"])
def test_raise_caption_keeps_requested_limb(body):
    result = compile_prompt("raise your " + body + " twice")
    assert body in result.gem_text_condition.captions[0]


@pytest.mark.parametrize("action,gerund", [("raise", "raising"), ("sit_down", "sitting"), ("rotate", "rotating")])
def test_keep_action_uses_the_shared_verb_lexicon(action, gerund):
    result = compile_prompt("keep " + gerund)
    assert result.motion_spec.segments[0].action == action
    assert result.motion_spec.segments[0].temporal_mode == "continuous"


def test_primary_and_secondary_repetition_counts_are_not_conflated():
    result = compile_prompt("jump twice while waving your left hand once")
    assert [s.temporal_constraint.count for s in result.motion_spec.segments] == [2, 1]


def test_continuous_secondary_action_does_not_change_primary_mode():
    result = compile_prompt("walk forward while continuously waving your right hand")
    first, second = result.motion_spec.segments
    assert first.temporal_mode is None and second.temporal_mode == "continuous"
    assert second.temporal_relation.type == "simultaneous"
    assert "waving the right hand continuously" in result.gem_text_condition.captions[0]


@pytest.mark.parametrize("structured", [True, False])
def test_legacy_artifact_counts_do_not_rebind_to_first_action_mention(structured):
    from motion_agent.compiler.temporal_resolver import resolve_temporal_semantics
    prompt = "wave your right hand twice then wave your left hand once"
    segments = compile_prompt(prompt).motion_spec.segments
    for segment in segments:
        segment.source_text = segment.source_start = segment.source_end = None
        if not structured:
            segment.temporal_constraint = None
    resolved = resolve_temporal_semantics(prompt, segments)
    assert [s.temporal_constraint.count for s in resolved] == [2, 1]
