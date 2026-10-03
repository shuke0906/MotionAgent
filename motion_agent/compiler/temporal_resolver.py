"""Temporal semantic normalization for Motion Compiler outputs."""

from __future__ import annotations

import re

from motion_agent.compiler.schemas import MotionSegment, TemporalConstraint, TemporalRelation
from motion_agent.compiler.semantic_parser import ACTION_FORMS


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
    by_id = {segment.segment_id: segment for segment in segments}
    for segment in segments:
        context = _segment_context(text, segment, while_clause)
        constraint = _resolve_repetition(context, segment)
        relation = _resolve_relation(context, segment, by_id)
        mode = "continuous" if _is_continuous(context, segment) else None
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
    if segment.source_text:
        return _normalize(segment.source_text)
    if while_clause and segment.secondary_actions:
        return while_clause
    action = segment.action.split("_")[0]
    match = re.search(rf"\b{re.escape(action)}\w*\b(?:\s+\w+){{0,5}}", text)
    return match.group(0) if match else text


def _resolve_repetition(context: str, segment: MotionSegment) -> TemporalConstraint | None:
    if segment.secondary_actions:
        context = _while_clause(context) or context
    # Legacy artifacts lack clause spans; their stored counts are authoritative.
    if not segment.source_text and segment.temporal_constraint:
        return segment.temporal_constraint
    count, source = _detect_count(context)
    if not segment.source_text and segment.repetition:
        count, source = segment.repetition, str(segment.repetition)
    if count is not None:
        return TemporalConstraint(type="repetition", mode="cycle", count=count, source_text=source)
    if segment.repetition:
        return TemporalConstraint(
            type="repetition",
            mode="cycle",
            count=segment.repetition,
            source_text=str(segment.repetition),
        )
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


def _resolve_relation(text: str, segment: MotionSegment, by_id: dict[int, MotionSegment]) -> TemporalRelation | None:
    if re.search(r"\balternat(?:e|ely|ing)\b", text):
        return TemporalRelation(
            type="alternating",
            marker="alternately",
            related_action=segment.action,
            body_parts=list(segment.body_parts),
        )
    if segment.simultaneous_with is not None:
        parent = by_id.get(segment.simultaneous_with)
        if parent is None:
            raise ValueError("temporal relation refers to an unknown parent")
        return TemporalRelation(
            type="simultaneous", marker="while", related_action=parent.action,
            body_parts=list(segment.body_parts),
        )
    while_clause = _while_clause(text)
    if while_clause and segment.secondary_actions:
        return TemporalRelation(
            type="simultaneous",
            marker="while",
            related_action=segment.secondary_actions[0],
            body_parts=[part for part in segment.body_parts if part != "full_body"],
        )
    return None


def _is_continuous(context: str, segment: MotionSegment) -> bool:
    action = segment.action.split("_")[0]
    forms = ACTION_FORMS.get(segment.action, (action, f"{action}ing"))
    gerund = next((form for form in forms if form.endswith("ing")), f"{action}ing")
    action_pattern = "(?:" + "|".join(re.escape(form) for form in forms) + ")"
    return bool(
        re.search(rf"\bkeep\s+{re.escape(gerund)}\b", context)
        or re.search(rf"\bcontinu(?:e|ing)\s+(?:to\s+)?{action_pattern}\b", context)
        or re.search(rf"\b{action_pattern}\s+continuously\b", context)
        or re.search(rf"\bcontinuously\s+{action_pattern}\b", context)
    )
