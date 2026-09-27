"""Compiler-owned routing hint detection."""

from __future__ import annotations

from motion_agent.compiler.schemas import ControlIntent, MotionSegment


def detect_control_intents(segments: list[MotionSegment]) -> list[ControlIntent]:
    hints: list[ControlIntent] = []
    for segment in segments:
        if segment.rare_motion_hints:
            hints.append(
                ControlIntent(
                    intent_type="retrieval_candidate",
                    segment_id=segment.segment_id,
                    reason_code="RARE_STYLE",
                    details={
                        "query_hint": " ".join(segment.rare_motion_hints + [segment.action.replace("_", " ")]).strip(),
                        "preferred_type": "motion",
                    },
                )
            )
        if segment.contact_intent:
            hints.append(
                ControlIntent(
                    intent_type="constraint_candidate",
                    segment_id=segment.segment_id,
                    reason_code="EXPLICIT_CONTACT",
                    details={
                        "constraint_type": "contact",
                        "body_part": segment.contact_intent.get("body_part"),
                        "target": segment.contact_intent.get("target"),
                        "time_s": segment.explicit_event_time_s,
                    },
                )
            )
        if segment.trajectory_intent:
            hints.append(
                ControlIntent(
                    intent_type="constraint_candidate",
                    segment_id=segment.segment_id,
                    reason_code="ROOT_TRAJECTORY",
                    details=segment.trajectory_intent,
                )
            )
        if segment.whole_body_keyframe_intent:
            hints.append(
                ControlIntent(
                    intent_type="keyframe_candidate",
                    segment_id=segment.segment_id,
                    reason_code="WHOLE_BODY_END_STATE",
                    details={
                        "time_s": segment.end_s,
                        "description": segment.whole_body_keyframe_intent.get("description", segment.action),
                    },
                )
            )
    return hints
