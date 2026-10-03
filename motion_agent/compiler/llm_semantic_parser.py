"""Real LLM semantic parser backend for Phase 11 local readiness."""

from __future__ import annotations

import json
import re

from pydantic import Field

from motion_agent.compiler.schemas import BodyPart, Direction, MotionSegment, Speed
from motion_agent.llm import LLMClient, LLMConfig, PromptEnvelope
from motion_agent.state.schemas import StrictModel


class LLMSemanticSegment(StrictModel):
    source_text: str = Field(description="Exact source clause from the user request.")
    action: str = Field(description="Canonical motion action, e.g. walk, wave, turn, sit_down.")
    secondary_actions: list[str] = Field(default_factory=list)
    body_parts: list[BodyPart] = Field(default_factory=lambda: ["full_body"])
    direction: Direction | None = None
    speed: Speed | None = "normal"
    style: list[str] = Field(default_factory=list)
    repetition: int | None = None
    orientation: str | None = None
    angle_deg: float | None = Field(default=None, gt=0, le=360)
    duration_weight: float = Field(default=1.0, gt=0)
    explicit_duration_s: float | None = Field(default=None, gt=0)
    explicit_event_time_s: float | None = None
    parent_segment_id: int | None = None
    simultaneous_with: int | None = None
    continuation_of: int | None = None
    interaction: str | None = Field(default=None, description="JSON object string or null.")
    contact_intent: str | None = Field(default=None, description="JSON object string or null.")
    trajectory_intent: str | None = Field(default=None, description="JSON object string or null.")
    whole_body_keyframe_intent: str | None = Field(default=None, description="JSON object string or null.")
    rare_motion_hints: list[str] = Field(default_factory=list)


class LLMSemanticParseResult(StrictModel):
    segments: list[LLMSemanticSegment] = Field(min_length=1)


SEMANTIC_PARSER_SYSTEM_PROMPT = """You are the MotionAgent semantic parser.

Convert the user request into a bounded structured motion plan. Do not generate SMPL values,
coordinates, tool calls, captions, or repairs.

Rules:
- Preserve only actions, body parts, direction, style, timing, repetition, and temporal relationships stated by the user.
- Use canonical actions where possible: walk, wave, turn, rotate, spin, sit_down, raise, jump, run, reach_and_contact, finish_stable_seated_pose, follow_trajectory.
- For simultaneous actions such as "walk while waving", prefer a parent segment for the base action and a child segment with simultaneous_with set to the parent segment id.
- Put repetition counts on the repeated action segment.
- Segment IDs are zero-based list indices. Set parent_segment_id and simultaneous_with on a simultaneous child.
- Use null for unspecified intent fields; any stated complex intent is a JSON object string.
- source_text must be an exact contiguous clause copied from the user request.
- Return only the required structured schema."""


class LLMSemanticParserBackend:
    def __init__(self, client: LLMClient | None = None, *, prompt_version: str | None = None) -> None:
        self.client = client or LLMClient(LLMConfig.from_env())
        self.prompt_version = prompt_version or self.client.config.prompt_version
        self.last_metadata = None

    def parse(self, request: str) -> list[MotionSegment]:
        envelope = PromptEnvelope(
            prompt_version=self.prompt_version,
            system=SEMANTIC_PARSER_SYSTEM_PROMPT,
            user=json.dumps({"user_request": request}, sort_keys=True),
            metadata={"component": "motion_compiler.semantic_parser"},
        )
        parsed, metadata = self.client.complete_structured(envelope, LLMSemanticParseResult)
        self.last_metadata = metadata
        segments = [
            MotionSegment(
                segment_id=index,
                source_text=segment.source_text,
                action=segment.action,
                secondary_actions=segment.secondary_actions,
                body_parts=segment.body_parts,
                direction=segment.direction,
                speed=segment.speed,
                style=segment.style,
                repetition=segment.repetition,
                orientation=segment.orientation,
                angle_deg=segment.angle_deg,
                duration_weight=segment.duration_weight,
                explicit_duration_s=segment.explicit_duration_s,
                explicit_event_time_s=segment.explicit_event_time_s,
                parent_segment_id=segment.parent_segment_id,
                simultaneous_with=segment.simultaneous_with,
                continuation_of=segment.continuation_of,
                interaction=_intent(segment.interaction),
                contact_intent=_intent(segment.contact_intent),
                trajectory_intent=_intent(segment.trajectory_intent),
                whole_body_keyframe_intent=_intent(segment.whole_body_keyframe_intent),
                rare_motion_hints=segment.rare_motion_hints,
            )
            for index, segment in enumerate(parsed.segments)
        ]
        return _attach_source_spans(request, segments)


def _intent(value: str | None) -> dict | None:
    if value is None:
        return None
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("semantic intent must encode a JSON object")
    return parsed


def _attach_source_spans(request: str, segments: list[MotionSegment]) -> list[MotionSegment]:
    lowered = request.lower()
    cursor = 0
    bound: list[MotionSegment] = []
    for segment in segments:
        source_text = (segment.source_text or "").strip()
        start = lowered.find(source_text.lower(), cursor) if source_text else -1
        if start < 0 and source_text:
            start = lowered.find(source_text.lower())
        if start >= 0:
            end = start + len(source_text)
            cursor = end
            bound.append(
                segment.model_copy(
                    update={
                        "source_start": start,
                        "source_end": end,
                        "source_text": request[start:end],
                    }
                )
            )
            continue
        # Last-resort conservative binding to the first clause containing the action.
        pattern = re.compile(rf"[^,.]*\b{re.escape(segment.action.split('_')[0])}\w*\b[^,.]*", re.IGNORECASE)
        match = pattern.search(request)
        if match:
            bound.append(
                segment.model_copy(
                    update={
                        "source_start": match.start(),
                        "source_end": match.end(),
                        "source_text": request[match.start():match.end()].strip(),
                    }
                )
            )
        else:
            bound.append(segment)
    return bound
