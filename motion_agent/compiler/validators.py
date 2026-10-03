"""Rule validators for Phase 3 Motion Compiler outputs."""

from __future__ import annotations

import re

from motion_agent.compiler.schemas import GEMTextCondition, MotionSegment, MotionSpecification
from motion_agent.compiler.semantic_parser import ACTION_FORMS, action_mentions
from motion_agent.compiler.temporal_resolver import _detect_count


KNOWN_ACTION_TERMS = {
    "walk",
    "run",
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
    by_id = {segment.segment_id: segment for segment in segments}
    if len(by_id) != len(segments):
        raise ValueError("segment IDs must be unique")
    previous_end = 0
    for segment in segments:
        if segment.start_frame is None or segment.end_frame is None or segment.start_frame >= segment.end_frame:
            raise ValueError("invalid segment frame interval")
        if segment.normalized_start is None or segment.normalized_end is None:
            raise ValueError("normalized windows are required")
        if not (0 <= segment.normalized_start < segment.normalized_end <= 1):
            raise ValueError("invalid normalized window")
        if not (0 <= segment.start_frame < segment.end_frame <= total_frames):
            raise ValueError("segment outside timeline")
        if segment.simultaneous_with is not None:
            parent = by_id.get(segment.simultaneous_with)
            if parent is None or parent.segment_id >= segment.segment_id:
                raise ValueError("simultaneous parent must be an earlier segment")
            if not (parent.start_frame <= segment.start_frame < segment.end_frame <= parent.end_frame):
                raise ValueError("simultaneous segment must overlap within parent bounds")
            if segment.parent_segment_id != parent.segment_id:
                raise ValueError("simultaneous parent relationship is inconsistent")
            if segment.continuation_of not in {None, parent.segment_id}:
                raise ValueError("continuation relationship is inconsistent")
        else:
            if segment.start_frame != previous_end:
                raise ValueError("frame boundaries must be continuous")
            previous_end = segment.end_frame
    if previous_end != total_frames:
        raise ValueError("sequential timeline must cover all frames")


def validate_request_coverage(request: str, segments: list[MotionSegment]) -> None:
    """Reject supported action mentions lost between the source and the DSL."""
    for action, start, end in action_mentions(request):
        candidates = [segment for segment in segments
                      if segment.source_start is not None and segment.source_end is not None
                      and segment.source_start <= start and end <= segment.source_end]
        if any(action == segment.action or any(action.replace("_down", "") in secondary.split()
               for secondary in segment.secondary_actions) for segment in candidates):
            continue
        if re.search(r"(?:continuing\s+to|still)\s*$", request[:start].lower()) and any(
            segment.action == action for segment in segments
        ):
            continue
        raise ValueError(f"source action {action!r} at [{start},{end}) is missing from MotionSpecification")
    for segment in segments:
        if segment.source_start is None or segment.source_end is None:
            continue
        source = request[segment.source_start:segment.source_end].lower()
        if source != (segment.source_text or "").lower():
            raise ValueError("source span does not match the original request")
        count, _ = _detect_count(source)
        if count is not None and (not segment.temporal_constraint or segment.temporal_constraint.count != count):
            raise ValueError("source repetition count is missing from MotionSpecification")
        for side, body in re.findall(r"\b(left|right)\s+(hand|arm|foot|leg)s?\b", source):
            if f"{side}_{body}" not in segment.body_parts:
                raise ValueError("source body part is missing from MotionSpecification")
        direction = re.search(r"\b(?:turn(?:s|ing|ed)?|rotate(?:s|d)?|rotating)\s+(left|right)\b", source)
        if direction and segment.direction != direction.group(1):
            raise ValueError("source direction is missing from MotionSpecification")


def validate_motion_semantics(spec: MotionSpecification, condition: GEMTextCondition) -> None:
    expected_bounds = {s.segment_id: (s.start_frame, s.end_frame) for s in spec.segments}
    if condition.segment_bounds and condition.segment_bounds != expected_bounds:
        raise ValueError("semantic segment bounds differ from MotionSpecification")
    windows = [(round(start * spec.total_frames), round(end * spec.total_frames), caption)
               for start, end, caption in zip(condition.window_start, condition.window_end, condition.captions)]
    for start, end, caption in windows:
        active = [segment for segment in spec.segments if segment.start_frame <= start and end <= segment.end_frame]
        if not active:
            raise ValueError("caption window has no corresponding semantic segment")
        lowered = caption.lower()
        required_terms = [term for segment in active for term in _required_terms(segment)]
        missing = [term for term in required_terms if not _contains_term(lowered, term)]
        if missing:
            raise ValueError(f"caption missing required terms for segments {[s.segment_id for s in active]}: {missing}")
        disallowed = set.intersection(*(set(_extra_action_terms(segment, lowered)) for segment in active))
        if disallowed:
            raise ValueError(f"caption adds unsupported actions for segments {[s.segment_id for s in active]}: {disallowed}")
    for segment in spec.segments:
        covered = sorted((max(start, segment.start_frame), min(end, segment.end_frame))
                         for start, end, _ in windows if start < segment.end_frame and segment.start_frame < end)
        cursor = segment.start_frame
        for start, end in covered:
            if start > cursor:
                break
            cursor = max(cursor, end)
        if cursor != segment.end_frame:
            raise ValueError(f"text conditions do not cover semantic segment {segment.segment_id}")


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
    for secondary in segment.secondary_actions:
        terms.extend("sit" if action == "sit_down" else action
                     for action, _, _ in action_mentions(secondary))
    if segment.repetition:
        terms.append(str(segment.repetition))
    elif segment.temporal_constraint and segment.temporal_constraint.count:
        terms.append(str(segment.temporal_constraint.count))
    if segment.angle_deg:
        terms.append(f"{segment.angle_deg:g}")
    if segment.direction and segment.direction != "none":
        terms.append(segment.direction)
    if "right_hand" in segment.body_parts:
        terms.extend(["right", "hand"])
    if "right_arm" in segment.body_parts:
        terms.extend(["right", "arm"])
    if "left_hand" in segment.body_parts:
        terms.extend(["left", "hand"])
    if "left_arm" in segment.body_parts:
        terms.extend(["left", "arm"])
    if segment.temporal_mode == "continuous":
        terms.append("continuously")
    if segment.frequency == "repeatedly":
        terms.append("repeatedly")
    if segment.temporal_relation and segment.temporal_relation.type == "alternating":
        terms.append("alternately")
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
        if _contains_term(caption, term) and term not in allowed:
            if not (term == "sit" and segment.action in {"sit_down", "finish_stable_seated_pose"}):
                extras.append(term)
    return extras


def _contains_term(caption: str, term: str) -> bool:
    canonical = "sit_down" if term == "sit" else term
    if canonical in ACTION_FORMS:
        return any(action == canonical for action, _, _ in action_mentions(caption))
    return bool(re.search(rf"\b{re.escape(term)}(?:s|es|ing)?\b", caption))
