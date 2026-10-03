"""Deterministic semantic parser backend for Phase 3 golden coverage."""

from __future__ import annotations

import re

from motion_agent.compiler.schemas import MotionSegment


NUMBER_WORDS = {
    "one": 1,
    "once": 1,
    "two": 2,
    "twice": 2,
    "three": 3,
    "thrice": 3,
    "four": 4,
    "five": 5,
}

ACTION_FORMS = {
    "walk": ("walk", "walks", "walking", "walked"),
    "wave": ("wave", "waves", "waving", "waved"),
    "turn": ("turn", "turns", "turning", "turned"),
    "rotate": ("rotate", "rotates", "rotating", "rotated"),
    "spin": ("spin", "spins", "spinning", "spun"),
    "sit_down": ("sit", "sits", "sitting", "sat"),
    "raise": ("raise", "raises", "raising", "raised", "lift", "lifts", "lifting", "lifted"),
    "jump": ("jump", "jumps", "jumping", "jumped"),
    "run": ("run", "runs", "running", "ran"),
}
ACTION_BY_FORM = {form: action for action, forms in ACTION_FORMS.items() for form in forms}
ACTION_PATTERN = r"\b(?:" + "|".join(ACTION_BY_FORM) + r")\b"


def action_mentions(text: str):
    return [(ACTION_BY_FORM[match.group()], match.start(), match.end())
            for match in re.finditer(ACTION_PATTERN, text.lower())]


def _canonicalize_actions(text: str) -> str:
    return re.sub(ACTION_PATTERN, lambda m: ACTION_BY_FORM[m.group()].replace("sit_down", "sit"), text)


def normalize_text(text: str) -> str:
    text = re.sub(r"(?<!\d)\.|\.(?!\d)", " ", text.lower())
    return " ".join(text.replace(",", " , ").split())


def parse_motion_request(request: str) -> list[MotionSegment]:
    segments = _parse_motion_request(request)
    tokens = list(re.finditer(r"\d+(?:\.\d+)?|\w+(?:-\w+)*", request.lower()))
    cursor = 0
    for segment in segments:
        clause = segment.source_text or ""
        words = re.findall(r"\d+(?:\.\d+)?|\w+(?:-\w+)*", clause.lower())
        for index in range(cursor, len(tokens) - len(words) + 1):
            if words and [token.group() for token in tokens[index:index + len(words)]] == words:
                segment.source_start = tokens[index].start()
                segment.source_end = tokens[index + len(words) - 1].end()
                segment.source_text = request[segment.source_start:segment.source_end]
                cursor = index + len(words)
                break
    return segments


def _parse_motion_request(request: str) -> list[MotionSegment]:
    text = normalize_text(request)
    if not text:
        return [_segment_from_clause("motion")]

    if " while " in text and (
        "continuing to" in text or "still " in text or " then " in text
        or "continuously" in text or "keep " in text
        or _detect_repetition(text.split(" while ", 1)[0]) is not None
    ):
        return _parse_composite(text)

    if " while " in text:
        primary, secondary = text.split(" while ", 1)
        segment = _segment_from_clause(primary)
        secondary_segment = _segment_from_clause(secondary)
        if secondary_segment.action != "orientation_hold":
            segment.secondary_actions = [_secondary_caption_fragment(secondary_segment)]
        segment.body_parts = _merge_unique(segment.body_parts + secondary_segment.body_parts)
        if secondary_segment.orientation:
            segment.orientation = secondary_segment.orientation
        segment.source_text = text
        return [segment]

    clauses = _split_sequential(text)
    return [_segment_from_clause(clause, segment_id=index) for index, clause in enumerate(clauses)]


def _parse_composite(text: str) -> list[MotionSegment]:
    """Keep concurrent actions separate when a request also has timeline stages."""
    primary_text, concurrent_text = text.split(" while ", 1)
    segments = [_segment_from_clause(clause, index) for index, clause in enumerate(_split_sequential(primary_text))]
    parent = segments[-1]
    continuation = re.match(r"(?:continuing to|still)\s+(\w+)\s*,\s*", concurrent_text)
    if continuation:
        continued_action = _detect_action(continuation.group(1))
        if continued_action != parent.action:
            raise ValueError("continuation must refer to the preceding action")
        concurrent_text = concurrent_text[continuation.end():]
    stages = re.split(r"\s+(?:and\s+)?then\s+|\s+afterwards?\s+", concurrent_text)
    concurrent = _segment_from_clause(stages[0], len(segments))
    concurrent.parent_segment_id = parent.segment_id
    concurrent.simultaneous_with = parent.segment_id
    concurrent.continuation_of = parent.segment_id if continuation else None
    segments.append(concurrent)
    for stage in stages[1:]:
        clauses = _split_coordinated(stage)
        stopped = False
        for clause in clauses:
            clause = clause.strip()
            if not clause:
                continue
            if clause == "stop":
                stopped = True
                continue
            segment = _segment_from_clause(clause, len(segments))
            # Unspecified terminal transitions share the remaining time equally.
            if segment.explicit_duration_s is None:
                segment.duration_weight = 1.0
            if stopped:
                segment.transition = "pause"
                stopped = False
            segments.append(segment)
    return segments


def _split_sequential(text: str) -> list[str]:
    comma_then = re.split(r"\s*,\s*(?:and\s+)?then\s+", text, maxsplit=1)
    if len(comma_then) == 2:
        leading = [part.strip() for part in comma_then[0].split(",") if part.strip()]
        return [*leading, *_split_sequential(comma_then[1].strip())]
    if " before " in text:
        left, right = text.split(" before ", 1)
        return [left.strip(), right.strip()]
    markers = [" and then ", " then ", " afterward ", " afterwards "]
    for marker in markers:
        if marker in text:
            return [clause for part in text.split(marker) if part.strip() for clause in _split_sequential(part.strip())]
    return _split_coordinated(text)


def _split_coordinated(text: str) -> list[str]:
    subject = r"(?:(?:a|the)\s+person\s+|(?:he|she|they)\s+)?"
    return [part.strip(" ,") for part in re.split(
        rf"\s*,\s*(?:and\s+)?|\s+and\s+(?={subject}{ACTION_PATTERN})", text
    ) if part.strip(" ,")]


def _segment_from_clause(clause: str, segment_id: int = 0) -> MotionSegment:
    source_text = clause.strip(" ,")
    clause = _canonicalize_actions(source_text)
    action = _detect_action(clause)
    body_parts = _detect_body_parts(clause, action)
    repetition = _detect_repetition(clause)
    direction = _detect_direction(clause)
    speed = _detect_speed(clause)
    style = _detect_style(clause)
    orientation = _detect_orientation(clause)
    event_time = _detect_event_time(clause)
    duration = _detect_duration(clause)
    interaction = _detect_interaction(clause)
    trajectory = _detect_trajectory(clause)
    keyframe = _detect_keyframe(clause)
    rare_hints = [hint for hint in style if hint in {"unusual", "asymmetric", "limping", "staggering"}]
    return MotionSegment(
        segment_id=segment_id,
        action=action,
        source_text=source_text,
        body_parts=body_parts,
        direction=direction,
        speed=speed,
        style=style,
        orientation=orientation,
        repetition=repetition,
        angle_deg=_detect_angle(clause),
        duration_weight=duration or _duration_weight(action),
        explicit_duration_s=duration,
        explicit_event_time_s=event_time,
        interaction=interaction,
        contact_intent=interaction if interaction and interaction.get("type") == "contact" else None,
        trajectory_intent=trajectory,
        whole_body_keyframe_intent=keyframe,
        rare_motion_hints=rare_hints,
    )


def _detect_action(clause: str) -> str:
    if any(word in clause for word in ["turn", "rotate", "spin"]):
        if "spin" in clause:
            return "spin"
        if "rotate" in clause:
            return "rotate"
        return "turn"
    if _detect_orientation(clause) and not any(word in clause for word in ["walk", "run", "move", "wave", "raise", "sit"]):
        return "orientation_hold"
    if any(word in clause for word in ["sit", "seated"]):
        if "finish" in clause or "stable" in clause or "pose" in clause:
            return "finish_stable_seated_pose"
        return "sit_down"
    if "wave" in clause or "waving" in clause:
        return "wave"
    if "walk" in clause or "limp" in clause:
        return "walk"
    if "raise" in clause or "lift" in clause:
        return "raise"
    if any(word in clause for word in ["touch", "contact", "place", "support"]):
        return "reach_and_contact"
    if any(word in clause for word in ["circle", "trajectory", "path"]):
        return "follow_trajectory"
    if "jump" in clause:
        return "jump"
    if re.search(r"\brun\b", clause):
        return "run"
    return clause or "unknown_motion"


def _detect_body_parts(clause: str, action: str) -> list:
    parts: list[str] = []
    coordinated = re.search(r"\b(?:left and right|right and left)\s+(hands?|arms?|feet|legs?)\b", clause)
    if coordinated:
        body = {"hands": "hand", "arms": "arm", "feet": "foot", "legs": "leg"}.get(
            coordinated.group(1), coordinated.group(1))
        parts.extend([f"left_{body}", f"right_{body}"])
    for side in ["left", "right"]:
        if f"{side} hand" in clause:
            parts.append(f"{side}_hand")
        if f"{side} arm" in clause:
            parts.append(f"{side}_arm")
        if f"{side} foot" in clause:
            parts.append(f"{side}_foot")
        if f"{side} leg" in clause:
            parts.append(f"{side}_leg")
    if "whole-body" in clause or "whole body" in clause or action in {"walk", "run", "sit_down", "finish_stable_seated_pose", "jump", "turn", "rotate", "spin", "orientation_hold"}:
        parts.insert(0, "full_body")
    if not parts and action in {"raise", "wave", "reach_and_contact"}:
        parts.append("right_arm" if "right" in clause else "left_arm" if "left" in clause else "right_arm")
    return _merge_unique(parts or ["full_body"])


def _detect_repetition(clause: str) -> int | None:
    for word in ("once", "twice", "thrice"):
        if re.search(rf"\b{word}\b", clause):
            return NUMBER_WORDS[word]
    match = re.search(r"\b(\d+)\s+times\b", clause)
    if match:
        return int(match.group(1))
    for word, value in NUMBER_WORDS.items():
        if re.search(rf"\b{word}\s+times\b|\b{word}\b", clause) and "time" in clause:
            return value
    return None


def _detect_angle(clause: str) -> float | None:
    match = re.search(r"\b(\d+(?:\.\d+)?)\s+degrees?\b", clause)
    return float(match.group(1)) if match else None


def _detect_direction(clause: str):
    for direction in ["forward", "backward", "clockwise", "counterclockwise"]:
        if re.search(rf"\b{direction}\b", clause):
            return direction
    for direction in ["left", "right"]:
        if (
            f"to the {direction}" in clause
            or f"toward the {direction}" in clause
            or f"{direction}ward" in clause
            or f"turn {direction}" in clause
            or f"rotate {direction}" in clause
            or f"face {direction}" in clause
            or f"facing {direction}" in clause
        ):
            return direction
    if "move up" in clause or "moves up" in clause:
        return "up"
    if "move down" in clause or "moves down" in clause:
        return "down"
    if "circle" in clause:
        return "clockwise" if "clockwise" in clause else "none"
    return None


def _detect_orientation(clause: str) -> str | None:
    for direction in ["left", "right", "forward", "backward"]:
        if f"face {direction}" in clause or f"facing {direction}" in clause:
            return direction
    if "turn around" in clause:
        return "backward"
    if "rotate 90" in clause:
        if "right" in clause or "clockwise" in clause:
            return "rotate_90_right"
        if "left" in clause or "counterclockwise" in clause:
            return "rotate_90_left"
        return "rotate_90"
    if "turn right" in clause:
        return "right"
    if "turn left" in clause:
        return "left"
    if "spin" in clause:
        return "spin"
    return None


def _detect_speed(clause: str):
    if "very slow" in clause:
        return "very_slow"
    if "slow" in clause or "slowly" in clause:
        return "slow"
    if "very fast" in clause:
        return "very_fast"
    if "fast" in clause or "quick" in clause:
        return "fast"
    return "normal"


def _detect_style(clause: str) -> list[str]:
    styles = []
    for style in ["unusual", "asymmetric", "limping", "staggering", "stable"]:
        if style in clause or (style == "limping" and "limp" in clause):
            styles.append(style)
    return styles


def _detect_event_time(clause: str) -> float | None:
    match = re.search(r"(?:at|after)\s+(\d+(?:\.\d+)?)\s*(?:seconds|second|s)\b", clause)
    return float(match.group(1)) if match else None


def _detect_duration(clause: str) -> float | None:
    match = re.search(r"for\s+(\d+(?:\.\d+)?)\s*(?:seconds|second|s)\b", clause)
    return float(match.group(1)) if match else None


def _detect_interaction(clause: str) -> dict | None:
    if any(word in clause for word in ["touch", "contact", "place", "support"]):
        target = "table" if "table" in clause else "surface" if "surface" in clause else "unknown"
        body_part = "right_hand" if "right" in clause else "left_hand" if "left" in clause else "hand"
        return {"type": "contact", "target": target, "body_part": body_part}
    return None


def _detect_trajectory(clause: str) -> dict | None:
    if any(word in clause for word in ["trajectory", "path", "circle", "circular"]):
        shape = "circle" if "circle" in clause or "circular" in clause else "specified_path"
        return {"type": "root_trajectory", "shape": shape}
    return None


def _detect_keyframe(clause: str) -> dict | None:
    if ("finish" in clause or "final" in clause or "end" in clause) and ("pose" in clause or "seated" in clause or "whole-body" in clause):
        return {"type": "whole_body_end_state", "description": clause}
    return None


def _duration_weight(action: str) -> float:
    return {
        "walk": 3.0,
        "wave": 2.0,
        "sit_down": 2.0,
        "raise": 1.0,
        "reach_and_contact": 1.0,
        "follow_trajectory": 3.0,
        "finish_stable_seated_pose": 1.5,
    }.get(action, 1.0)


def _secondary_caption_fragment(segment: MotionSegment) -> str:
    if segment.action in {"wave", "raise"}:
        parts = [part.replace("_", " ") for part in segment.body_parts if part != "full_body"]
        return segment.action + (" " + " and ".join(parts) if parts else "")
    return segment.action.replace("_", " ")


def _merge_unique(values: list[str]) -> list[str]:
    result = []
    for value in values:
        if value not in result:
            result.append(value)
    return result
