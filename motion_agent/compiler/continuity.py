"""Deterministic cross-segment continuity rules for Phase 3."""

from __future__ import annotations

from motion_agent.compiler.schemas import HeadingContinuitySpec, MotionSegment


LOCOMOTION_ACTIONS = {"walk", "run", "move", "follow_trajectory"}
EXPLICIT_ORIENTATION_ACTIONS = {"turn", "spin", "rotate"}


def apply_heading_continuity(segments: list[MotionSegment]) -> list[MotionSegment]:
    """Attach semantic heading/facing expectations without numeric yaw values."""

    updated: list[MotionSegment] = []
    prior_heading_segment_id: int | None = None

    for segment in segments:
        policy = infer_heading_continuity(segment, prior_heading_segment_id)
        revised = segment.model_copy(update={"heading_continuity": policy})
        updated.append(revised)

        if policy.mode in {"explicit", "inherit_previous", "follow_trajectory"}:
            prior_heading_segment_id = revised.segment_id

    return updated


def infer_heading_continuity(
    segment: MotionSegment,
    prior_heading_segment_id: int | None,
) -> HeadingContinuitySpec:
    if _uses_trajectory_orientation(segment):
        return HeadingContinuitySpec(
            mode="follow_trajectory",
            anchor_segment_id=prior_heading_segment_id,
            preserve_facing=True,
            explicit_facing=_trajectory_facing(segment),
            source="trajectory",
        )

    explicit_facing = _explicit_facing(segment)
    if explicit_facing is not None:
        return HeadingContinuitySpec(
            mode="explicit",
            anchor_segment_id=prior_heading_segment_id,
            preserve_facing=True,
            explicit_facing=explicit_facing,
            source="user_explicit",
        )

    if _establishes_heading(segment):
        return HeadingContinuitySpec(
            mode="explicit",
            anchor_segment_id=prior_heading_segment_id,
            preserve_facing=True,
            explicit_facing=segment.direction,
            source="user_explicit",
        )

    if prior_heading_segment_id is not None:
        return HeadingContinuitySpec(
            mode="inherit_previous",
            anchor_segment_id=prior_heading_segment_id,
            preserve_facing=True,
            source="continuity_default",
        )

    return HeadingContinuitySpec(mode="free", source="unknown")


def _uses_trajectory_orientation(segment: MotionSegment) -> bool:
    if not segment.trajectory_intent:
        return False
    orientation = (segment.orientation or "").lower()
    shape = str(segment.trajectory_intent.get("shape", "")).lower()
    return shape in {"circle", "circular"} or "tangent" in orientation or "trajectory" in orientation


def _trajectory_facing(segment: MotionSegment) -> str | None:
    orientation = (segment.orientation or "").lower()
    if "tangent" in orientation:
        return "tangent"
    if segment.trajectory_intent:
        return f"{segment.trajectory_intent.get('shape', 'trajectory')}_path"
    return None


def _explicit_facing(segment: MotionSegment) -> str | None:
    orientation = (segment.orientation or "").strip().lower()
    if orientation:
        return orientation
    if segment.action in EXPLICIT_ORIENTATION_ACTIONS:
        return segment.direction or segment.action
    return None


def _establishes_heading(segment: MotionSegment) -> bool:
    return (
        segment.action in LOCOMOTION_ACTIONS
        and segment.direction is not None
        and segment.direction != "none"
    )
