"""Rule validators for Phase 3 Motion Compiler outputs."""

from __future__ import annotations

from motion_agent.compiler.schemas import GEMTextCondition, MotionSegment, MotionSpecification


KNOWN_ACTION_TERMS = {
    "walk",
    "wave",
    "sit",
    "raise",
    "reach",
    "touch",
    "move",
    "jump",
    "turn",
    "rotate",
    "spin",
    "finish",
}


def validate_timeline(segments: list[MotionSegment], *, total_frames: int) -> None:
    if segments[0].start_s != 0.0 or segments[0].start_frame != 0:
        raise ValueError("first segment must start at 0")
    if segments[-1].end_frame != total_frames:
        raise ValueError("last segment must end at total_frames")
    previous_end = 0
    for segment in segments:
        if segment.start_frame != previous_end:
            raise ValueError("frame boundaries must be continuous")
        if segment.start_frame is None or segment.end_frame is None or segment.start_frame >= segment.end_frame:
            raise ValueError("invalid segment frame interval")
        if segment.normalized_start is None or segment.normalized_end is None:
            raise ValueError("normalized windows are required")
        if not (0 <= segment.normalized_start < segment.normalized_end <= 1):
            raise ValueError("invalid normalized window")
        previous_end = segment.end_frame


def validate_motion_semantics(spec: MotionSpecification, condition: GEMTextCondition) -> None:
    for segment, caption in zip(spec.segments, condition.captions):
        lowered = caption.lower()
        required_terms = _required_terms(segment)
        missing = [term for term in required_terms if term not in lowered]
        if missing:
            raise ValueError(f"caption missing required terms for segment {segment.segment_id}: {missing}")
        disallowed = _extra_action_terms(segment, lowered)
        if disallowed:
            raise ValueError(f"caption adds unsupported actions for segment {segment.segment_id}: {disallowed}")


def validate_caption_length(condition: GEMTextCondition, *, max_words: int = 50) -> None:
    for caption in condition.captions:
        if len(caption.split()) > max_words:
            raise ValueError("caption exceeds deterministic Phase 3 word limit")


def validate_gem_text_condition(condition: GEMTextCondition) -> None:
    GEMTextCondition.model_validate(condition.model_dump(mode="python"))


def validate_heading_continuity(segments: list[MotionSegment]) -> None:
    ids = {segment.segment_id for segment in segments}
    trajectory_segments = {segment.segment_id for segment in segments if segment.trajectory_intent}
    for segment in segments:
        policy = segment.heading_continuity
        if policy.mode == "inherit_previous":
            if policy.anchor_segment_id is None:
                raise ValueError(f"segment {segment.segment_id} inherits heading without an anchor")
            if policy.anchor_segment_id not in ids or policy.anchor_segment_id >= segment.segment_id:
                raise ValueError(f"segment {segment.segment_id} heading anchor must be an earlier segment")
            if segment.action in {"turn", "rotate", "spin"} or segment.orientation:
                raise ValueError(f"segment {segment.segment_id} has explicit orientation but inherits heading")
        elif policy.mode == "follow_trajectory":
            if segment.segment_id not in trajectory_segments:
                raise ValueError(f"segment {segment.segment_id} follows trajectory heading without trajectory intent")
        elif policy.mode == "explicit":
            if policy.source != "user_explicit" or not policy.explicit_facing:
                raise ValueError(f"segment {segment.segment_id} explicit heading requires user-facing metadata")
        elif policy.mode == "free":
            caption = (segment.gem_caption or "").lower()
            if "same facing direction" in caption or "facing forward" in caption:
                raise ValueError(f"segment {segment.segment_id} free heading caption implies fixed facing")


def _required_terms(segment: MotionSegment) -> list[str]:
    terms = []
    if segment.action == "sit_down":
        terms.append("sit")
    elif segment.action == "reach_and_contact":
        terms.append("touch" if segment.contact_intent else "reach")
    elif segment.action == "finish_stable_seated_pose":
        terms.extend(["stable", "seated"])
    elif segment.action == "follow_trajectory":
        terms.append("path")
    else:
        terms.append(segment.action.split("_")[0])
    if segment.repetition:
        terms.append(str(segment.repetition))
    if segment.direction and segment.direction != "none":
        terms.append(segment.direction)
    if "right_hand" in segment.body_parts:
        terms.extend(["right", "hand"])
    if "right_arm" in segment.body_parts:
        terms.extend(["right", "arm"])
    for style in segment.style:
        if style != "stable":
            terms.append(style)
    return terms


def _extra_action_terms(segment: MotionSegment, caption: str) -> list[str]:
    allowed = set(_required_terms(segment))
    allowed.update(segment.action.split("_"))
    for secondary in segment.secondary_actions:
        allowed.update(secondary.split())
    if segment.contact_intent:
        allowed.update({"touch", "touching", "reach", "reaches"})
    if segment.trajectory_intent:
        allowed.update({"move", "moves", "following", "path"})
    if segment.action == "finish_stable_seated_pose":
        allowed.update({"finish", "finishes", "sit", "sits", "seated"})
    extras = []
    for term in KNOWN_ACTION_TERMS:
        if term in caption and term not in allowed:
            if not (term == "sit" and segment.action in {"sit_down", "finish_stable_seated_pose"}):
                extras.append(term)
    return extras
