"""Deterministic GEM caption optimizer."""

from __future__ import annotations

from motion_agent.compiler.schemas import MotionSegment


def optimize_caption(segment: MotionSegment) -> str:
    parts = ["a person"]
    if segment.speed and segment.speed not in {"normal"}:
        parts.append(_speed_word(segment.speed))
    parts.append(_action_phrase(segment))
    if segment.direction and segment.direction != "none" and segment.direction not in parts[-1]:
        parts.append(segment.direction)
    if segment.style:
        style = " ".join(s for s in segment.style if s != "stable")
        if style:
            parts.append(f"with a {style} style" if "gait" not in style else f"with a {style}")
    if segment.repetition:
        parts.append(f"{segment.repetition} times")
    if segment.secondary_actions:
        parts.append("while " + " and ".join(_secondary_phrase(action) for action in segment.secondary_actions))
    if segment.contact_intent:
        target = segment.contact_intent.get("target", "target")
        body_part = _humanize(segment.contact_intent.get("body_part", "hand"))
        parts.append(f"touching the {target} with the {body_part}")
    if segment.trajectory_intent:
        shape = segment.trajectory_intent.get("shape", "specified path")
        parts.append(f"following a {shape} path")
    if _should_materialize_inherited_heading(segment):
        parts.append("keeping the same facing direction")
    elif _should_materialize_explicit_facing(segment):
        parts.append(f"facing {_humanize(segment.heading_continuity.explicit_facing)}")
    caption = " ".join(part for part in parts if part).strip()
    return _compress(caption)


def _action_phrase(segment: MotionSegment) -> str:
    action = segment.action
    if action == "walk":
        if "limping" in segment.style or "asymmetric" in segment.style:
            return "walks"
        return "walks"
    if action == "wave":
        body = "right hand" if "right_hand" in segment.body_parts else "left hand" if "left_hand" in segment.body_parts else "hand"
        return f"waves the {body}"
    if action == "sit_down":
        return "sits down"
    if action == "raise":
        body = "right arm" if "right_arm" in segment.body_parts else "left arm" if "left_arm" in segment.body_parts else "arm"
        return f"raises the {body}"
    if action == "reach_and_contact":
        return "reaches"
    if action == "follow_trajectory":
        return "moves"
    if action == "turn":
        return "turns"
    if action == "rotate":
        return "rotates"
    if action == "spin":
        return "spins"
    if action == "orientation_hold":
        return "holds orientation"
    if action == "finish_stable_seated_pose":
        return "finishes in a stable seated whole body pose"
    return _humanize(action)


def _speed_word(speed: str) -> str:
    return {
        "very_slow": "very slowly",
        "slow": "slowly",
        "fast": "quickly",
        "very_fast": "very quickly",
    }.get(speed, "")


def _humanize(value: str) -> str:
    return value.replace("_", " ").replace("-", " ")


def _secondary_phrase(value: str) -> str:
    if value == "wave right hand":
        return "waving the right hand"
    if value == "wave left hand":
        return "waving the left hand"
    return _humanize(value)


def _should_materialize_inherited_heading(segment: MotionSegment) -> bool:
    if segment.heading_continuity.mode != "inherit_previous":
        return False
    return segment.action in {"wave", "raise", "sit_down", "reach_and_contact", "finish_stable_seated_pose"}


def _should_materialize_explicit_facing(segment: MotionSegment) -> bool:
    policy = segment.heading_continuity
    if policy.mode != "explicit" or not policy.explicit_facing:
        return False
    if segment.action in {"turn", "rotate", "spin"}:
        return False
    return segment.orientation is not None


def _compress(caption: str, max_words: int = 35) -> str:
    words = caption.split()
    return " ".join(words[:max_words])
