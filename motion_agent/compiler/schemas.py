"""Canonical Motion Compiler schemas for Phase 3."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, model_validator

from motion_agent.state.schemas import StrictModel


BodyPart = Literal[
    "full_body",
    "head",
    "torso",
    "pelvis",
    "left_arm",
    "right_arm",
    "left_hand",
    "right_hand",
    "left_leg",
    "right_leg",
    "left_foot",
    "right_foot",
]

Direction = Literal[
    "forward",
    "backward",
    "left",
    "right",
    "up",
    "down",
    "clockwise",
    "counterclockwise",
    "none",
]

Speed = Literal["very_slow", "slow", "normal", "fast", "very_fast"]
Transition = Literal["continuous", "abrupt", "hold", "pause"]
HeadingContinuityMode = Literal["free", "inherit_previous", "explicit", "follow_trajectory"]
HeadingContinuitySource = Literal["user_explicit", "continuity_default", "trajectory", "unknown"]
TemporalMode = Literal["continuous"]
TemporalConstraintType = Literal["repetition"]
TemporalConstraintMode = Literal["cycle"]
TemporalRelationType = Literal["simultaneous", "alternating"]


class TemporalConstraint(StrictModel):
    type: TemporalConstraintType
    mode: TemporalConstraintMode = "cycle"
    count: int | None = None
    quantifier: str | None = None
    source_text: str | None = None

    @model_validator(mode="after")
    def validate_constraint(self) -> "TemporalConstraint":
        if self.count is not None and self.count <= 0:
            raise ValueError("temporal repetition count must be positive")
        if self.count is None and self.quantifier is None:
            raise ValueError("temporal repetition requires count or quantifier")
        return self


class TemporalRelation(StrictModel):
    type: TemporalRelationType
    marker: str | None = None
    related_action: str | None = None
    body_parts: list[BodyPart] = Field(default_factory=list)


class HeadingContinuitySpec(StrictModel):
    mode: HeadingContinuityMode = "free"
    anchor_segment_id: int | None = None
    preserve_facing: bool = False
    explicit_facing: str | None = None
    source: HeadingContinuitySource = "unknown"


class MotionSegment(StrictModel):
    segment_id: int
    action: str
    source_text: str | None = None
    source_start: int | None = None
    source_end: int | None = None
    secondary_actions: list[str] = Field(default_factory=list)
    body_parts: list[BodyPart] = Field(default_factory=lambda: ["full_body"])
    direction: Direction | None = None
    speed: Speed | None = "normal"
    style: list[str] = Field(default_factory=list)
    repetition: int | None = None
    temporal_constraint: TemporalConstraint | None = None
    temporal_mode: TemporalMode | None = None
    temporal_relation: TemporalRelation | None = None
    angle_deg: float | None = Field(default=None, gt=0, le=360)
    parent_segment_id: int | None = None
    continuation_of: int | None = None
    simultaneous_with: int | None = None
    frequency: str | None = None
    orientation: str | None = None
    transition: Transition | None = "continuous"
    interaction: dict[str, Any] | None = None
    duration_weight: float = 1.0
    explicit_duration_s: float | None = Field(default=None, gt=0)
    duration_s: float | None = Field(default=None, gt=0)
    explicit_start_s: float | None = None
    explicit_end_s: float | None = None
    explicit_event_time_s: float | None = None
    start_s: float | None = None
    end_s: float | None = None
    start_frame: int | None = None
    end_frame: int | None = None
    normalized_start: float | None = None
    normalized_end: float | None = None
    contact_intent: dict[str, Any] | None = None
    trajectory_intent: dict[str, Any] | None = None
    whole_body_keyframe_intent: dict[str, Any] | None = None
    rare_motion_hints: list[str] = Field(default_factory=list)
    heading_continuity: HeadingContinuitySpec = Field(default_factory=HeadingContinuitySpec)
    gem_caption: str | None = None
    confidence: float = 1.0
    validated: bool = True

    @model_validator(mode="after")
    def validate_segment(self) -> "MotionSegment":
        if self.repetition is not None and self.repetition <= 0:
            raise ValueError("repetition must be positive")
        if self.duration_weight <= 0:
            raise ValueError("duration_weight must be positive")
        if self.start_s is not None and self.end_s is not None and self.start_s >= self.end_s:
            raise ValueError("segment start_s must be before end_s")
        if self.start_frame is not None and self.end_frame is not None and self.start_frame >= self.end_frame:
            raise ValueError("segment start_frame must be before end_frame")
        return self


class ControlIntent(StrictModel):
    intent_type: Literal["retrieval_candidate", "constraint_candidate", "keyframe_candidate"]
    segment_id: int
    reason_code: str
    details: dict[str, Any] = Field(default_factory=dict)


class MotionSpecification(StrictModel):
    original_request: str
    duration_s: float
    fps: int
    total_frames: int
    segments: list[MotionSegment]
    control_intents: list[ControlIntent] = Field(default_factory=list)


class GEMTextCondition(StrictModel):
    captions: list[str]
    window_start: list[float]
    window_end: list[float]
    total_frames: int
    segment_bounds: dict[int, tuple[int, int]] = Field(default_factory=dict)

    def bounds_for_segment(self, segment_id: int) -> tuple[int, int]:
        if self.segment_bounds:
            if segment_id not in self.segment_bounds:
                raise ValueError(f"unknown semantic segment {segment_id}")
            return self.segment_bounds[segment_id]
        if not 0 <= segment_id < len(self.captions):
            raise ValueError(f"unknown legacy text segment {segment_id}")
        return (round(self.window_start[segment_id] * self.total_frames),
                round(self.window_end[segment_id] * self.total_frames))

    @model_validator(mode="after")
    def validate_windows(self) -> "GEMTextCondition":
        if not (len(self.captions) == len(self.window_start) == len(self.window_end)):
            raise ValueError("captions/window_start/window_end counts must match")
        for start, end in zip(self.window_start, self.window_end):
            if not (0 <= start <= 1 and 0 <= end <= 1 and start < end):
                raise ValueError("GEM windows must satisfy 0 <= start < end <= 1")
        for segment_id, (start, end) in self.segment_bounds.items():
            if segment_id < 0 or not 0 <= start < end <= self.total_frames:
                raise ValueError("invalid semantic segment bounds")
        return self


class CompilerResult(StrictModel):
    motion_spec: MotionSpecification
    gem_text_condition: GEMTextCondition
    routing_hints: list[ControlIntent] = Field(default_factory=list)
    changed_segments: list[int] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
