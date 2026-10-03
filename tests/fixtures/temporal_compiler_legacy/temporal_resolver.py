"""Temporal semantic normalization for Motion Compiler outputs."""

from __future__ import annotations

import re

from motion_agent.compiler.schemas import MotionSegment, TemporalConstraint, TemporalRelation


_NUMBER_WORDS = {
    "one": 1,
    "once": 1,
    "two": 2,
    "twice": 2,
    "three": 3,
    "thrice": 3,
    "four": 4,
}


def resolve_temporal_semantics(request: str, segments: list[MotionSegment]) -> list[MotionSegment]:
    """Attach structured temporal semantics without changing generation behavior."""

    text = _normalize(request)
    while_clause = _while_clause(text)
    resolved: list[MotionSegment] = []
    for segment in segments:
        context = _segment_context(text, segment, while_clause)
        constraint = _resolve_repetition(context, segment)
        relation = _resolve_relation(text, segment, while_clause)
        mode = "continuous" if _is_continuous(context, segment) or _is_continuous(text, segment) else None
        updates = {}
        if constraint:
            updates["temporal_constraint"] = constraint
            if constraint.count and not segment.secondary_actions and segment.repetition is None:
                updates["repetition"] = constraint.count
            if constraint.quantifier == "unbounded":
                updates["frequency"] = "repeatedly"
        if relation:
            updates["temporal_relation"] = relation
        if mode:
            updates["temporal_mode"] = mode
            updates["transition"] = "continuous"
        resolved.append(segment.model_copy(update=updates))
    return resolved


def _normalize(text: str) -> str:
    return " ".join(text.lower().replace(",", " , ").replace(".", " ").split())


def _while_clause(text: str) -> str | None:
    if " while " not in text:
        return None
    return text.split(" while ", 1)[1]


def _segment_context(text: str, segment: MotionSegment, while_clause: str | None) -> str:
    if while_clause and segment.secondary_actions:
        return while_clause
    action = segment.action.split("_")[0]
    match = re.search(rf"\b{re.escape(action)}\w*\b(?:\s+\w+){{0,5}}", text)
    return match.group(0) if match else text


def _resolve_repetition(context: str, segment: MotionSegment) -> TemporalConstraint | None:
    if segment.repetition:
        return TemporalConstraint(
            type="repetition",
            mode="cycle",
            count=segment.repetition,
            source_text=str(segment.repetition),
        )
    count, source = _detect_count(context)
    if count is not None:
        return TemporalConstraint(type="repetition", mode="cycle", count=count, source_text=source)
    if re.search(r"\brepeatedly\b", context):
        return TemporalConstraint(
            type="repetition",
            mode="cycle",
            quantifier="unbounded",
            source_text="repeatedly",
        )
    return None


def _detect_count(text: str) -> tuple[int | None, str | None]:
    for word in ("once", "twice", "thrice"):
        if re.search(rf"\b{word}\b", text):
            return _NUMBER_WORDS[word], word
    match = re.search(r"\b(\d+)\s+times\b", text)
    if match:
        return int(match.group(1)), match.group(0)
    for word, value in _NUMBER_WORDS.items():
        if re.search(rf"\b{word}\s+times\b", text):
            return value, f"{word} times"
    return None, None


def _resolve_relation(text: str, segment: MotionSegment, while_clause: str | None) -> TemporalRelation | None:
    if re.search(r"\balternat(?:e|ely|ing)\b", text):
        return TemporalRelation(
            type="alternating",
            marker="alternately",
            related_action=segment.action,
            body_parts=list(segment.body_parts),
        )
    if while_clause and segment.secondary_actions:
        return TemporalRelation(
            type="simultaneous",
            marker="while",
            related_action=segment.secondary_actions[0],
            body_parts=_body_parts_from_text(while_clause),
        )
    return None


def _is_continuous(context: str, segment: MotionSegment) -> bool:
    action = segment.action.split("_")[0]
    gerund = {
        "walk": "walking",
        "wave": "waving",
        "jump": "jumping",
        "run": "running",
    }.get(action, f"{action}ing")
    action_pattern = rf"(?:{re.escape(action)}\w*|{re.escape(gerund)})"
    return bool(
        re.search(rf"\bkeep\s+{re.escape(gerund)}\b", context)
        or re.search(rf"\bcontinu(?:e|ing)\s+(?:to\s+)?{action_pattern}\b", context)
        or re.search(rf"\b{action_pattern}\s+continuously\b", context)
    )


def _body_parts_from_text(text: str) -> list[str]:
    parts: list[str] = []
    for side in ("left", "right"):
        if f"{side} hand" in text:
            parts.append(f"{side}_hand")
        if f"{side} arm" in text:
            parts.append(f"{side}_arm")
    return parts
