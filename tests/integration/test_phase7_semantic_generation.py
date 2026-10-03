from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from motion_agent.compiler import CompilerRequest, MotionCompiler
from motion_agent.compiler.validators import validate_timeline
from motion_agent.constraints.schemas import HardMotionCondition, VerificationSpec
from motion_agent.constraints.store import ConditionStore
from motion_agent.generation import assemble_generation_conditions, build_generation_request, generate_motion
from motion_agent.generation.candidate_store import CandidateStore
from motion_agent.generation.gem_adapter import prepare_pure_text_input
from motion_agent.keyframes.schemas import KeyframeSpec, KeyframeVerificationSpec
from scripts.validate_phase7_runpod import (
    PROMPT, RecordingGEM, condition_interface_evidence, prepare_semantic_run, satisfaction_evidence,
)
from tests.unit.test_phase4_generation import FakeGEMModel


def test_required_composite_timeline_and_captions():
    result = MotionCompiler().compile(CompilerRequest(original_request=PROMPT))
    walk, wave, turn, sit = result.motion_spec.segments
    assert [(s.action, s.start_frame, s.end_frame) for s in (walk, wave, turn, sit)] == [
        ("walk", 0, 120), ("wave", 60, 120), ("turn", 120, 150), ("sit_down", 150, 180),
    ]
    assert walk.explicit_duration_s == 2 and walk.duration_s == 4
    assert wave.parent_segment_id == wave.continuation_of == wave.simultaneous_with == 0
    assert wave.body_parts == ["right_hand"] and wave.repetition == 2
    assert turn.direction == "left" and turn.angle_deg == 90 and turn.transition == "pause"
    assert "right hand" in wave.gem_caption and "2 times" in wave.gem_caption
    assert "90 degrees" in turn.gem_caption and "left" in turn.gem_caption
    assert result.gem_text_condition.window_end[:2] == [0.333333, 0.666667]
    assert "while waving" in result.gem_text_condition.captions[1]
    assert result.gem_text_condition.segment_bounds[1] == (60, 120)


def test_composite_parser_is_not_specific_to_required_wording():
    prompt = "Walk backward for 1 second. While continuing to walk, wave your left hand thrice. Then turn 45 degrees to the right, and sit down."
    result = MotionCompiler().compile(CompilerRequest(original_request=prompt))
    walk, wave, turn, sit = result.motion_spec.segments
    assert walk.direction == "backward" and wave.start_frame == 30
    assert wave.repetition == 3 and wave.body_parts == ["left_hand"]
    assert wave.end_frame == walk.end_frame == turn.start_frame
    assert turn.direction == "right" and turn.angle_deg == 45
    assert sit.end_frame == 180


def test_concurrent_stage_followed_by_sequence_overlaps_parent():
    result = MotionCompiler().compile(CompilerRequest(original_request="walk forward while waving the right hand then sit down"))
    walk, wave, sit = result.motion_spec.segments
    assert wave.continuation_of is None and wave.simultaneous_with == 0
    assert (walk.start_frame, walk.end_frame) == (wave.start_frame, wave.end_frame)
    assert sit.start_frame == walk.end_frame


def test_timeline_rejects_invalid_overlap_relationship():
    result = MotionCompiler().compile(CompilerRequest(original_request=PROMPT))
    result.motion_spec.segments[1].simultaneous_with = 3
    with pytest.raises(ValueError, match="earlier segment"):
        validate_timeline(result.motion_spec.segments, total_frames=180)


def test_adapter_keeps_semantic_overlap_with_compound_text_windows():
    compiled = MotionCompiler().compile(CompilerRequest(original_request=PROMPT))
    request = build_generation_request(condition=compiled.gem_text_condition, motion_spec=compiled.motion_spec, fps=30, seed=7)
    data, info = prepare_pure_text_input(request)
    assert int(data["length"]) == data["kp2d"].shape[0] == 180
    assert [(s["start"], s["end"]) for s in info] == [(0, 60), (60, 120), (120, 150), (150, 180)]
    assert compiled.gem_text_condition.segment_bounds[0] == (0, 120)
    assert data["meta"][0]["motion_specification"] == compiled.motion_spec.model_dump(mode="json")


def test_scripted_planner_compiler_generator_candidate_store_and_cache():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        state, decision, spec, text, _ = prepare_semantic_run(root / "planner")
        assert decision.action == "GENERATE" and state.plan.status == "ready"
        request = build_generation_request(condition=text, motion_spec=spec, fps=30, seeds=[7, 8], num_candidates=decision.payload["num_candidates"])
        model = RecordingGEM(FakeGEMModel())
        store = CandidateStore(root / "candidates")
        first = generate_motion(request, model=model, checkpoint_version="fake", candidate_store=store)
        repeated = generate_motion(request, model=model, checkpoint_version="fake", candidate_store=store)
        assert first.status == "success" and len(first.candidates) == 2
        assert repeated.inference_calls == 0 and repeated.cache_hits == 2
        assert model.receipt["motion_specification"] == spec.model_dump(mode="json")
        for candidate in first.candidates:
            saved = store.load(candidate.candidate_id)
            assert saved["motion_repr"].shape == (180, 151)
            assert saved["candidate"].metadata.generation_request == request


def test_condition_interface_passes_when_model_does_not_satisfy_target():
    compiled = MotionCompiler().compile(CompilerRequest(original_request=PROMPT))
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        conditions = ConditionStore(root / "conditions")
        store = CandidateStore(root / "candidates")
        values = torch.zeros(180, 151)
        mask = torch.zeros_like(values)
        values[90, :6], mask[90, :6] = 100, 1
        handle = conditions.save_hard_condition(HardMotionCondition(values=values, mask=mask, source_constraint_id="constraint"))
        verification = VerificationSpec(verification_spec_id="verification", constraint_id="constraint", metric_type="feature_linf_error", target_segments=[1], frame_indices=[90], pass_threshold=1e-4, units="normalized_features")
        key_verification = KeyframeVerificationSpec(keyframe_id="key", target_frame=90, temporal_tolerance_frames=3, pose_handle="pose_fixture", metrics=["joint_rotation_error"])
        key = KeyframeSpec(keyframe_id="key", target_segment=1, target_time_s=3, target_frame=90, temporal_tolerance_frames=3, description="Fixture pose", source_type="explicit_pose", pose_handle="pose_fixture", hard_condition_handle=handle, verification_spec=key_verification)
        assembled = assemble_generation_conditions(text_condition_id="text", text_condition_payload=compiled.gem_text_condition.model_dump(mode="json"), total_frames=180, hard_condition_handles=[handle], active_constraint_ids=["constraint"], active_keyframe_ids=["key"], verification_specs=[verification], keyframe_specs=[key], store=conditions)
        request = build_generation_request(condition=compiled.gem_text_condition, motion_spec=compiled.motion_spec, fps=30, seed=7, condition_bundle=assembled.bundle)
        model = RecordingGEM(FakeGEMModel(), conditions)
        result = generate_motion(request, model=model, checkpoint_version="fake", candidate_store=store, condition_store=conditions)
        evidence = condition_interface_evidence(request, result, store, model.receipt)
        assert evidence["status"] == "PASS"
        saved_request = store.load(result.candidates[0].candidate_id)["candidate"].metadata.generation_request
        assert saved_request.condition_bundle.verification_specs == [verification]
        assert saved_request.condition_bundle.keyframe_specs == [key]
        motion = store.load(result.candidates[0].candidate_id)["motion_repr"]
        error = float((motion[90, :6] - 100).abs().max())
        assert error > 1e-4
        satisfaction = satisfaction_evidence({"error": error}, False)
        assert satisfaction["status"] == "DEFERRED" and not satisfaction["tolerance_met"]


def test_invalid_condition_mask_is_an_interface_failure():
    compiled = MotionCompiler().compile(CompilerRequest(original_request=PROMPT))
    with tempfile.TemporaryDirectory() as tmp:
        conditions = ConditionStore(Path(tmp) / "conditions")
        handle = conditions.save_hard_condition(HardMotionCondition(values=torch.zeros(180, 151), mask=torch.full((180, 151), 0.5), source_constraint_id="invalid"))
        request = build_generation_request(condition=compiled.gem_text_condition, fps=30, seed=7)
        request.condition_bundle.hard_motion_condition_handle = handle
        with pytest.raises(ValueError, match="binary"):
            prepare_pure_text_input(request, condition_store=conditions)
