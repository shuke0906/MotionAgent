"""Bounded LLM transport schema; canonical action validators remain authoritative."""

import json
from enum import Enum
from typing import Literal

from motion_agent.agent.actions import PlannerAction, PlannerDecision, REASON_CODES
from motion_agent.state.schemas import StrictModel

ReasonCode = Enum("ReasonCode", {code: code for code in sorted(REASON_CODES)}, type=str)


class LLMPayload(StrictModel):
    mode: Literal["initial", "revise", "add", "replace", "remove"] | None = None
    focus: list[Literal["semantic", "timeline", "frequency", "body_part", "style", "gem_caption"]] | None = None
    query: str | None = None
    retrieval_type: Literal["caption", "motion", "pose", "trajectory"] | None = None
    purpose: Literal["prompt_grounding", "motion_prior", "keyframe_source", "constraint_source"] | None = None
    top_k: int | None = None
    constraint_type: Literal["joint_target", "body_part_pose", "root_trajectory", "contact", "fixed_joint"] | None = None
    body_part: str | None = None
    time_range: list[float] | None = None
    target: str | None = None
    target_json: str | None = None
    strength: Literal["hard", "soft"] | None = None
    time: float | None = None
    description: str | None = None
    source_preference: Literal["retrieval", "ik", "generation", "auto"] | None = None
    strategy: Literal["normal", "guided"] | None = None
    scope: Literal["full", "segment"] | None = None
    num_candidates: int | None = None
    reward_targets: list[str] | None = None


class LLMPlannerDecision(StrictModel):
    action: PlannerAction
    reason_code: ReasonCode
    target_segments: list[int] | None = None
    reason_summary: str
    payload: LLMPayload

    def canonical(self) -> PlannerDecision:
        payload = self.payload.model_dump(exclude_none=True)
        target_json = payload.pop("target_json", None)
        if target_json is not None:
            if "target" in payload:
                raise ValueError("target and target_json are mutually exclusive")
            target = json.loads(target_json)
            if not isinstance(target, dict):
                raise ValueError("target_json must encode an object")
            payload["target"] = target
        return PlannerDecision(
            action=self.action, reason_code=self.reason_code.value,
            target_segments=self.target_segments, reason_summary=self.reason_summary,
            payload=payload,
        )
