"""Deterministic retrieval query construction."""

from __future__ import annotations

import re
from typing import Any

from motion_agent.retrieval.schemas import RetrievalPurpose, RetrievalQuery


ACTION_ALIASES = {
    "walk": "walk",
    "walking": "walk",
    "limp": "walk",
    "limping": "walk",
    "sit": "sit_down",
    "sitting": "sit_down",
    "seated": "sit_down",
    "jump": "jump",
    "jumping": "jump",
    "wave": "wave",
    "waving": "wave",
    "turn": "turn",
    "turning": "turn",
}


def build_retrieval_query(segment: Any, query_hint: str, purpose: RetrievalPurpose) -> RetrievalQuery:
    tokens = set(re.findall(r"[a-zA-Z_]+", query_hint.lower()))
    hinted_action = _first_match(tokens, ACTION_ALIASES)
    action = hinted_action or getattr(segment, "action", None)
    direction = _direction(tokens) or (None if hinted_action else getattr(segment, "direction", None))
    speed = _speed(tokens) or (None if hinted_action else getattr(segment, "speed", None))
    body_parts = [] if hinted_action else list(getattr(segment, "body_parts", []) or [])
    body_parts = _merge_unique(body_parts, _body_parts(tokens))
    style = [] if hinted_action else list(getattr(segment, "style", []) or [])
    style = _merge_unique(style, _styles(tokens))
    repetition = getattr(segment, "repetition", None)

    text = _compose_text(
        action=action,
        direction=direction,
        speed=speed,
        body_parts=body_parts,
        style=style,
        repetition=repetition,
        fallback=query_hint,
    )
    return RetrievalQuery(
        text=text,
        action=action,
        body_parts=body_parts,
        direction=direction,
        speed=speed,
        style=style,
        repetition=repetition,
        purpose=purpose,
        metadata={"query_hint": query_hint},
    )


def _first_match(tokens: set[str], aliases: dict[str, str]) -> str | None:
    for token in tokens:
        if token in aliases:
            return aliases[token]
    return None


def _direction(tokens: set[str]) -> str | None:
    for value in ["forward", "backward", "left", "right", "clockwise", "counterclockwise"]:
        if value in tokens:
            return value
    return None


def _speed(tokens: set[str]) -> str | None:
    if "slow" in tokens or "slowly" in tokens:
        return "slow"
    if "fast" in tokens or "quickly" in tokens:
        return "fast"
    return None


def _body_parts(tokens: set[str]) -> list[str]:
    parts: list[str] = []
    if "right" in tokens and "hand" in tokens:
        parts.append("right_hand")
    elif "left" in tokens and "hand" in tokens:
        parts.append("left_hand")
    elif "hand" in tokens:
        parts.append("right_hand")
    if "arm" in tokens:
        parts.append("right_arm" if "right" in tokens else "left_arm" if "left" in tokens else "right_arm")
    if "leg" in tokens:
        parts.append("right_leg" if "right" in tokens else "left_leg" if "left" in tokens else "right_leg")
    return parts


def _styles(tokens: set[str]) -> list[str]:
    styles: list[str] = []
    for value in ["limping", "asymmetric", "injured", "uneven", "drunk", "careful"]:
        if value in tokens:
            styles.append("limping" if value in {"injured", "uneven"} else value)
    if "gait" in tokens and "limping" in styles:
        styles.append("gait")
    return styles


def _merge_unique(left: list[str], right: list[str]) -> list[str]:
    merged: list[str] = []
    for value in [*left, *right]:
        if value and value not in merged:
            merged.append(value)
    return merged


def _compose_text(
    *,
    action: str | None,
    direction: str | None,
    speed: str | None,
    body_parts: list[str],
    style: list[str],
    repetition: int | None,
    fallback: str,
) -> str:
    action_phrase = {
        "walk": "walks",
        "sit_down": "sits down",
        "jump": "jumps",
        "wave": "waves",
        "turn": "turns",
    }.get(action or "", action or "")
    if not action_phrase:
        return fallback.strip()
    parts = ["a person"]
    if speed and speed != "normal":
        parts.append("slowly" if speed == "slow" else "quickly" if speed == "fast" else speed)
    parts.append(action_phrase)
    if body_parts and action in {"wave", "raise"}:
        parts.append("the " + body_parts[0].replace("_", " "))
    if direction:
        parts.append(direction)
    if style:
        style_text = " ".join(value for value in style if value != "gait")
        if "limping" in style and "asymmetric" in style:
            parts.append("with an asymmetric limping gait")
        elif style_text:
            parts.append(f"with a {style_text} style")
    if repetition:
        parts.append(f"{repetition} times")
    return " ".join(parts).strip()
