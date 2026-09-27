"""Canonical Phase 2 planner actions and decision schema."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import Field, PositiveInt, model_validator

from motion_agent.state.schemas import StrictModel


class PlannerAction(str, Enum):
    COMPILE_MOTION = "COMPILE_MOTION"
    RETRIEVE_REFERENCE = "RETRIEVE_REFERENCE"
    BUILD_CONSTRAINT = "BUILD_CONSTRAINT"
    BUILD_KEYFRAME = "BUILD_KEYFRAME"
    GENERATE = "GENERATE"
    ACCEPT = "ACCEPT"
    STOP_FAILED = "STOP_FAILED"


EXECUTABLE_ACTIONS = {
    PlannerAction.COMPILE_MOTION,
    PlannerAction.RETRIEVE_REFERENCE,
    PlannerAction.BUILD_CONSTRAINT,
    PlannerAction.BUILD_KEYFRAME,
    PlannerAction.GENERATE,
}

TERMINAL_ACTIONS = {PlannerAction.ACCEPT, PlannerAction.STOP_FAILED}

REASON_CODES = {
    "INITIAL_COMPILE",
    "SEMANTIC_UNCLEAR",
    "TEMPORAL_UNCLEAR",
    "RARE_MOTION",
    "MISSING_REFERENCE",
    "GEOMETRIC_REQUIREMENT",
    "CONTACT_REQUIREMENT",
    "TRAJECTORY_REQUIREMENT",
    "WHOLE_BODY_STATE_REQUIRED",
    "CONDITIONS_READY",
    "SEMANTIC_FAILURE",
    "TEMPORAL_FAILURE",
    "CONSTRAINT_FAILURE",
    "NATURALNESS_FAILURE",
    "PERSISTENT_CONSTRAINT_FAILURE",
    "PERSISTENT_FAILURE",
    "ALL_REQUIRED_CHECKS_PASSED",
    "BUDGET_EXHAUSTED",
    "UNRECOVERABLE_TOOL_FAILURE",
}


class CompileMotionPayload(StrictModel):
    mode: Literal["initial", "revise"]
    focus: list[Literal["semantic", "timeline", "frequency", "body_part", "style", "gem_caption"]]


class RetrieveReferencePayload(StrictModel):
    query: str = Field(min_length=1)
    retrieval_type: Literal["caption", "motion", "pose", "trajectory"]
    purpose: Literal["prompt_grounding", "motion_prior", "keyframe_source", "constraint_source"]
    top_k: PositiveInt = 5


class BuildConstraintPayload(StrictModel):
    mode: Literal["add", "replace", "remove"]
    constraint_type: Literal["joint_target", "body_part_pose", "root_trajectory", "contact", "fixed_joint"]
    body_part: str | None = None
    time_range: list[float] | None = None
    target: str | dict[str, Any] | None = None
    strength: Literal["hard", "soft"] = "soft"


class BuildKeyframePayload(StrictModel):
    mode: Literal["add", "replace", "remove"]
    time: float
    description: str = Field(min_length=1)
    source_preference: Literal["retrieval", "ik", "generation", "auto"]


class GeneratePayload(StrictModel):
    strategy: Literal["normal", "guided"]
    scope: Literal["full", "segment"]
    num_candidates: PositiveInt = 4
    reward_targets: list[str] = Field(default_factory=list)


class EmptyPayload(StrictModel):
    pass


PAYLOAD_SCHEMAS = {
    PlannerAction.COMPILE_MOTION: CompileMotionPayload,
    PlannerAction.RETRIEVE_REFERENCE: RetrieveReferencePayload,
    PlannerAction.BUILD_CONSTRAINT: BuildConstraintPayload,
    PlannerAction.BUILD_KEYFRAME: BuildKeyframePayload,
    PlannerAction.GENERATE: GeneratePayload,
    PlannerAction.ACCEPT: EmptyPayload,
    PlannerAction.STOP_FAILED: EmptyPayload,
}


class PlannerDecision(StrictModel):
    action: PlannerAction
    reason_code: str
    target_segments: list[int] | None = None
    reason_summary: str = Field(min_length=1)
    payload: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_reason_and_payload(self) -> "PlannerDecision":
        if self.reason_code not in REASON_CODES:
            raise ValueError(f"unknown reason_code: {self.reason_code}")
        validate_action_payload(self.action, self.payload)
        return self

    @property
    def typed_payload(self) -> StrictModel:
        return validate_action_payload(self.action, self.payload)


def validate_action_payload(action: PlannerAction | str, payload: dict[str, Any]) -> StrictModel:
    planner_action = PlannerAction(action)
    schema = PAYLOAD_SCHEMAS[planner_action]
    return schema.model_validate(payload)


def parse_planner_decision(value: Any) -> PlannerDecision:
    """Accept one structured decision and reject multi-action responses."""

    if isinstance(value, list):
        raise ValueError("Planner must return exactly one PlannerDecision, not a list")
    if isinstance(value, dict) and "actions" in value:
        raise ValueError("Planner must return exactly one PlannerDecision, not an actions collection")
    return PlannerDecision.model_validate(value)


ACTION_TO_NODE = {
    PlannerAction.COMPILE_MOTION: "compile_motion",
    PlannerAction.RETRIEVE_REFERENCE: "retrieval",
    PlannerAction.BUILD_CONSTRAINT: "constraint",
    PlannerAction.BUILD_KEYFRAME: "keyframe",
    PlannerAction.GENERATE: "generation",
    PlannerAction.ACCEPT: "accept",
    PlannerAction.STOP_FAILED: "stop_failed",
}

