"""Deterministic GEM caption optimizer."""

from __future__ import annotations

import re

from motion_agent.compiler.schemas import MotionSegment


def optimize_caption(segment: MotionSegment, retrieved_captions: list[str] | None = None) -> str:
    parts = ["a person"]
    if segment.speed and segment.speed not in {"normal"}:
        parts.append(_speed_word(segment.speed))
    parts.append(_action_phrase(segment))
    if segment.angle_deg:
        parts.append(f"{segment.angle_deg:g} degrees")
    if segment.direction and segment.direction != "none" and segment.direction not in parts[-1]:
        parts.append(segment.direction)
    if segment.style:
        style = " ".join(s for s in segment.style if s != "stable")
        if style:
            parts.append(f"with a {style} style" if "gait" not in style else f"with a {style}")
    if segment.repetition:
        parts.append(f"{segment.repetition} times")
    if segment.frequency == "repeatedly" and not segment.secondary_actions:
        parts.append("repeatedly")
    if segment.temporal_mode == "continuous":
        parts.append("continuously")
    if segment.secondary_actions:
        secondary = " and ".join(_secondary_phrase(action) for action in segment.secondary_actions)
        if segment.temporal_constraint and segment.temporal_constraint.count and not segment.repetition:
            secondary = f"{secondary} {segment.temporal_constraint.count} times"
        if segment.frequency == "repeatedly":
            secondary += " repeatedly"
        parts.append("while " + secondary)
    if segment.temporal_relation and segment.temporal_relation.type == "alternating":
        parts.append("alternately")
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
    if retrieved_captions:
        parts = _apply_retrieved_caption_style(parts, segment, retrieved_captions)
    caption = " ".join(part for part in parts if part).strip()
    return _compress(caption)


def _action_phrase(segment: MotionSegment) -> str:
    action = segment.action
    if action == "walk":
        if "limping" in segment.style or "asymmetric" in segment.style:
            return "walks"
        return "walks"
    if action == "wave":
        return f"waves the {_limb_phrase(segment, 'hand')}"
    if action == "sit_down":
        return "sits down"
    if action == "raise":
        return f"raises the {_limb_phrase(segment, 'arm')}"
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
    if action == "run":
        return "runs"
    if action == "jump":
        return "jumps"
    if action == "orientation_hold":
        return "holds orientation"
    if action == "finish_stable_seated_pose":
        return "finishes in a stable seated whole body pose"
    return _humanize(action)


def _limb_phrase(segment: MotionSegment, default: str) -> str:
    parts = [part for part in segment.body_parts if part in {"left_hand", "right_hand", "left_arm", "right_arm"}]
    for body in ("hand", "arm"):
        if set(parts) == {f"left_{body}", f"right_{body}"}:
            return f"left and right {body}s"
    return " and ".join(_humanize(part) for part in parts) or default


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
    for action, gerund in (("wave", "waving"), ("raise", "raising")):
        if value.startswith(action + " "):
            return gerund + " the " + value[len(action) + 1:]
    return _humanize(value)


def compose_concurrent_caption(segments: list[MotionSegment]) -> str:
    """Express concurrent intent once; leave every DSL segment unchanged."""
    parent, *children = segments
    caption = parent.gem_caption or optimize_caption(parent)
    gerunds = {"walks": "walking", "waves": "waving", "turns": "turning",
               "rotates": "rotating", "spins": "spinning", "sits": "sitting",
               "raises": "raising", "jumps": "jumping", "runs": "running",
               "reaches": "reaching", "moves": "moving", "finishes": "finishing"}
    clauses = []
    for child in children:
        phrase = (child.gem_caption or optimize_caption(child)).removeprefix("a person ")
        if child.heading_continuity.mode == "inherit_previous":
            phrase = phrase.removesuffix(" keeping the same facing direction")
        phrase = re.sub(r"\b(?:" + "|".join(gerunds) + r")\b",
                        lambda match: gerunds[match.group()], phrase, count=1)
        clauses.append(phrase)
    return caption + (" while " + " and ".join(clauses) if clauses else "")


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


def _apply_retrieved_caption_style(parts: list[str], segment: MotionSegment, retrieved_captions: list[str]) -> list[str]:
    """Use retrieved captions as untrusted wording examples only."""

    lowered = " ".join(retrieved_captions).lower()
    enriched = list(parts)
    if segment.action == "walk" and ("limp" in lowered or "uneven" in lowered):
        style_text = " ".join(enriched)
        if "limping gait" not in style_text and "with a limping style" not in style_text:
            if "asymmetric" in segment.style or "asymmetric" in lowered:
                enriched.append("with an asymmetric limping gait")
            else:
                enriched.append("with a limping gait")
    if segment.action == "sit_down" and "seated" in lowered and "seated" not in " ".join(enriched):
        enriched.append("into a seated pose")
    if segment.action == "wave" and "right hand" in lowered and "right_hand" in segment.body_parts:
        # The base caption already carries the body part; no semantic expansion.
        return enriched
    return enriched
