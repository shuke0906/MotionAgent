from __future__ import annotations

import json
import unittest
from pathlib import Path

from motion_agent.compiler import CompilerRequest, MotionCompiler, MotionSpecification
from motion_agent.compiler.gem_text_compiler import build_multi_text_data
from motion_agent.compiler.validators import validate_heading_continuity, validate_motion_semantics, validate_timeline


FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "phase3_compiler" / "golden_cases.json"


def compile_prompt(prompt: str):
    return MotionCompiler().compile(
        CompilerRequest(original_request=prompt, duration_s=6.0, fps=30, total_frames=180)
    )


def heading_modes(result):
    return [segment.heading_continuity.mode for segment in result.motion_spec.segments]


def heading_anchors(result):
    return [segment.heading_continuity.anchor_segment_id for segment in result.motion_spec.segments]


class Phase3MotionCompilerTests(unittest.TestCase):
    def test_golden_dsl_fixtures(self):
        cases = json.loads(FIXTURES.read_text(encoding="utf-8"))
        for case in cases:
            with self.subTest(case=case["name"]):
                result = compile_prompt(case["prompt"])
                self.assertEqual(len(result.motion_spec.segments), len(case["segments"]))
                for index, expected in enumerate(case["segments"]):
                    segment = result.motion_spec.segments[index]
                    self.assertEqual(segment.action, expected["action"])
                    if "direction" in expected:
                        self.assertEqual(segment.direction, expected["direction"])
                    if "repetition" in expected:
                        self.assertEqual(segment.repetition, expected["repetition"])
                    for body_part in expected.get("body_parts", []):
                        self.assertIn(body_part, segment.body_parts)
                    for style in expected.get("style", []):
                        self.assertIn(style, segment.style)
                    if "explicit_event_time_s" in expected:
                        self.assertEqual(segment.explicit_event_time_s, expected["explicit_event_time_s"])
                    if "secondary_contains" in expected:
                        self.assertIn(expected["secondary_contains"], segment.secondary_actions)

                hints = [(hint.intent_type, hint.reason_code) for hint in result.routing_hints]
                for hint in case["routing_hints"]:
                    self.assertIn((hint["intent_type"], hint["reason_code"]), hints)

    def test_required_example_wave_three_times_then_sit(self):
        result = compile_prompt("wave the right hand three times and sit down")
        self.assertEqual(len(result.motion_spec.segments), 2)
        first, second = result.motion_spec.segments
        self.assertEqual(first.action, "wave")
        self.assertIn("right_hand", first.body_parts)
        self.assertEqual(first.repetition, 3)
        self.assertEqual(first.temporal_constraint.type, "repetition")
        self.assertEqual(first.temporal_constraint.mode, "cycle")
        self.assertEqual(first.temporal_constraint.count, 3)
        self.assertEqual(second.action, "sit_down")
        self.assertLess(first.end_frame, second.end_frame)
        self.assertEqual(first.end_frame, second.start_frame)

    def test_temporal_resolver_structures_wave_three_times(self):
        result = compile_prompt("wave your right hand three times")
        segment = result.motion_spec.segments[0]
        self.assertEqual(segment.action, "wave")
        self.assertIn("right_hand", segment.body_parts)
        self.assertEqual(segment.temporal_constraint.model_dump(mode="json"), {
            "type": "repetition",
            "mode": "cycle",
            "count": 3,
            "quantifier": None,
            "source_text": "three times",
        })

    def test_temporal_resolver_marks_keep_walking_continuous(self):
        result = compile_prompt("keep walking forward")
        segment = result.motion_spec.segments[0]
        self.assertEqual(segment.action, "walk")
        self.assertEqual(segment.direction, "forward")
        self.assertEqual(segment.temporal_mode, "continuous")

    def test_temporal_resolver_supports_twice(self):
        result = compile_prompt("jump twice")
        segment = result.motion_spec.segments[0]
        self.assertEqual(segment.action, "jump")
        self.assertEqual(segment.temporal_constraint.count, 2)

    def test_temporal_relation_for_walk_while_waving_three_times(self):
        result = compile_prompt("Walk forward while waving your right hand three times.")
        segment = result.motion_spec.segments[0]
        self.assertEqual(segment.action, "walk")
        self.assertIn("right_hand", segment.body_parts)
        self.assertEqual(segment.temporal_relation.type, "simultaneous")
        self.assertEqual(segment.temporal_relation.marker, "while")
        self.assertEqual(segment.temporal_constraint.count, 3)

    def test_timeline_invariants(self):
        result = compile_prompt("walk forward then wave the right hand three times then sit down")
        validate_timeline(result.motion_spec.segments, total_frames=result.motion_spec.total_frames)
        self.assertEqual(result.motion_spec.segments[0].start_s, 0.0)
        self.assertEqual(result.motion_spec.segments[-1].end_frame, result.motion_spec.total_frames)
        for left, right in zip(result.motion_spec.segments, result.motion_spec.segments[1:]):
            self.assertEqual(left.end_frame, right.start_frame)
            self.assertLessEqual(left.normalized_end, right.normalized_start)

    def test_comma_then_sequence_preserves_all_actions_at_300_frames(self):
        result = MotionCompiler().compile(
            CompilerRequest(
                original_request="walk forward, wave the right hand, then sit down",
                duration_s=10.0,
                fps=30,
                total_frames=300,
            )
        )
        self.assertEqual(
            [segment.action for segment in result.motion_spec.segments],
            ["walk", "wave", "sit_down"],
        )
        self.assertEqual(result.gem_text_condition.total_frames, 300)
        self.assertEqual(len(result.gem_text_condition.captions), 3)
        self.assertEqual(result.gem_text_condition.window_start[0], 0.0)
        self.assertEqual(result.gem_text_condition.window_end[-1], 1.0)

    def test_heading_inherits_across_walk_wave_sit(self):
        result = compile_prompt("walk forward, wave the right hand, then sit down")
        self.assertEqual([segment.action for segment in result.motion_spec.segments], ["walk", "wave", "sit_down"])
        self.assertEqual(heading_modes(result), ["explicit", "inherit_previous", "inherit_previous"])
        self.assertEqual(heading_anchors(result), [None, 0, 1])
        self.assertEqual(result.motion_spec.segments[0].heading_continuity.explicit_facing, "forward")
        self.assertNotIn("turn", " ".join(result.gem_text_condition.captions).lower())

    def test_explicit_turn_resets_heading_anchor(self):
        result = compile_prompt("walk forward, turn right, then sit down")
        self.assertEqual([segment.action for segment in result.motion_spec.segments], ["walk", "turn", "sit_down"])
        self.assertEqual(heading_modes(result), ["explicit", "explicit", "inherit_previous"])
        self.assertEqual(heading_anchors(result), [None, 0, 1])
        self.assertEqual(result.motion_spec.segments[1].heading_continuity.explicit_facing, "right")
        self.assertEqual(result.motion_spec.segments[2].heading_continuity.anchor_segment_id, 1)

    def test_no_prior_heading_is_free(self):
        result = compile_prompt("wave the right hand")
        self.assertEqual(heading_modes(result), ["free"])
        self.assertIsNone(result.motion_spec.segments[0].heading_continuity.anchor_segment_id)

    def test_trajectory_heading_policy_follows_trajectory(self):
        result = compile_prompt("walk along a circle")
        self.assertEqual(heading_modes(result), ["follow_trajectory"])
        self.assertEqual(result.motion_spec.segments[0].heading_continuity.source, "trajectory")

    def test_segment_revision_preserves_unrelated_heading_metadata(self):
        initial = compile_prompt("walk forward, wave the right hand, then sit down")
        original_first = initial.motion_spec.segments[0].heading_continuity
        original_second = initial.motion_spec.segments[1].heading_continuity
        revised = MotionCompiler().compile(
            CompilerRequest(
                original_request="raise the right arm",
                mode="revise",
                target_segments=[2],
                focus=["semantic", "body_part", "gem_caption"],
                duration_s=6.0,
                fps=30,
                total_frames=180,
                existing_motion_spec=MotionSpecification.model_validate(initial.motion_spec.model_dump(mode="python")),
            )
        )
        self.assertEqual(revised.changed_segments, [2])
        self.assertEqual(revised.motion_spec.segments[0].heading_continuity, original_first)
        self.assertEqual(revised.motion_spec.segments[1].heading_continuity, original_second)
        self.assertEqual(revised.motion_spec.segments[2].heading_continuity.mode, "inherit_previous")

    def test_explicit_facing_in_simultaneous_motion_is_preserved(self):
        result = compile_prompt("walk forward while facing left")
        self.assertEqual(len(result.motion_spec.segments), 1)
        segment = result.motion_spec.segments[0]
        self.assertEqual(segment.action, "walk")
        self.assertEqual(segment.orientation, "left")
        self.assertEqual(segment.heading_continuity.mode, "explicit")
        self.assertEqual(segment.heading_continuity.explicit_facing, "left")

    def test_simultaneous_action_stays_in_one_segment(self):
        result = compile_prompt("walk forward while waving the right hand")
        self.assertEqual(len(result.motion_spec.segments), 1)
        segment = result.motion_spec.segments[0]
        self.assertEqual(segment.action, "walk")
        self.assertIn("wave right hand", segment.secondary_actions)
        self.assertIn("right_hand", segment.body_parts)
        self.assertEqual(segment.heading_continuity.mode, "explicit")
        self.assertEqual(segment.heading_continuity.explicit_facing, "forward")

    def test_explicit_timing_is_preserved_without_llm_frame_math(self):
        result = compile_prompt("at 3 seconds raise the right arm")
        segment = result.motion_spec.segments[0]
        self.assertEqual(segment.explicit_event_time_s, 3.0)
        self.assertEqual(segment.start_frame, 0)
        self.assertEqual(segment.end_frame, 180)

    def test_routing_hint_boundaries_do_not_create_downstream_artifacts(self):
        result = compile_prompt("touch the table with the right hand")
        self.assertEqual(len(result.routing_hints), 1)
        hint = result.routing_hints[0]
        self.assertEqual(hint.intent_type, "constraint_candidate")
        self.assertEqual(hint.reason_code, "EXPLICIT_CONTACT")
        dumped = result.model_dump(mode="json")
        self.assertNotIn("CompiledConstraint", json.dumps(dumped))
        self.assertNotIn("RetrievedReference", json.dumps(dumped))

    def test_rare_style_and_keyframe_hints_stay_recommendations(self):
        rare = compile_prompt("walk with an unusual asymmetric limping gait")
        self.assertIn(("retrieval_candidate", "RARE_STYLE"), [(h.intent_type, h.reason_code) for h in rare.routing_hints])
        keyframe = compile_prompt("finish in a stable seated whole-body pose")
        self.assertIn(
            ("keyframe_candidate", "WHOLE_BODY_END_STATE"),
            [(h.intent_type, h.reason_code) for h in keyframe.routing_hints],
        )

    def test_caption_semantic_preservation_for_all_fixtures(self):
        for case in json.loads(FIXTURES.read_text(encoding="utf-8")):
            with self.subTest(case=case["name"]):
                result = compile_prompt(case["prompt"])
                validate_heading_continuity(result.motion_spec.segments)
                validate_motion_semantics(result.motion_spec, result.gem_text_condition)

    def test_gem_text_condition_contract_and_multi_text_mapping(self):
        result = compile_prompt("walk forward then sit down")
        condition = result.gem_text_condition
        self.assertEqual(len(condition.captions), len(condition.window_start))
        self.assertEqual(len(condition.captions), len(condition.window_end))
        for start, end in zip(condition.window_start, condition.window_end):
            self.assertGreaterEqual(start, 0.0)
            self.assertLessEqual(end, 1.0)
            self.assertLess(start, end)
        multi_text = build_multi_text_data(condition)
        self.assertEqual(multi_text["caption"], condition.captions)
        self.assertEqual(multi_text["text_ind"], [0, 1])

    def test_segment_revision_preserves_unrelated_validated_segments(self):
        initial = compile_prompt("walk forward then wave the right hand three times then sit down")
        original_first = initial.motion_spec.segments[0]
        original_third = initial.motion_spec.segments[2]
        revised = MotionCompiler().compile(
            CompilerRequest(
                original_request="raise the right arm",
                mode="revise",
                target_segments=[1],
                focus=["semantic", "body_part", "gem_caption"],
                duration_s=6.0,
                fps=30,
                total_frames=180,
                existing_motion_spec=MotionSpecification.model_validate(initial.motion_spec.model_dump(mode="python")),
            )
        )
        self.assertEqual(revised.changed_segments, [1])
        self.assertEqual(revised.motion_spec.segments[1].action, "raise")
        self.assertEqual(revised.motion_spec.segments[0].action, original_first.action)
        self.assertEqual(revised.motion_spec.segments[0].body_parts, original_first.body_parts)
        self.assertEqual(revised.motion_spec.segments[2].action, original_third.action)
        self.assertEqual(revised.motion_spec.segments[2].body_parts, original_third.body_parts)


if __name__ == "__main__":
    unittest.main()
