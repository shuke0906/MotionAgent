"""Schemas for Phase 6 constraint compilation and shared conditions."""

from __future__ import annotations

from typing import Any, Literal

import torch
from pydantic import ConfigDict, Field, model_validator

from motion_agent.state.schemas import StrictModel


ConstraintType = Literal["joint_target", "body_part_pose", "root_trajectory", "contact", "fixed_joint"]
ConstraintStatus = Literal[
    "compiled",
    "compiled_soft",
    "needs_reference",
    "needs_geometry",
    "conflict",
    "unsupported",
    "error",
]
EffectiveMode = Literal["hard_condition", "selection_reward", "guided_reward"]


class TensorModel(StrictModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)


class TimeSpec(StrictModel):
    type: Literal["point", "window", "segment"]
    time_s: float | None = None
    start_s: float | None = None
    end_s: float | None = None
    segment_relative: bool = False


class ResolvedTimeSpec(StrictModel):
    frame_indices: list[int]
    start_frame: int
    end_frame: int

    @model_validator(mode="after")
    def validate_frames(self) -> "ResolvedTimeSpec":
        if not self.frame_indices:
            raise ValueError("frame_indices must not be empty")
        if self.start_frame > self.end_frame:
            raise ValueError("start_frame must be <= end_frame")
        return self


class ConstraintTarget(StrictModel):
    target_type: Literal[
        "point",
        "trajectory",
        "plane",
        "reference_pose",
        "reference_motion",
        "relative_anchor",
        "self_anchor",
        "scene_anchor",
    ]
    coordinate_frame: Literal["world", "root_local", "gravity_aligned", "scene_local"] = "world"
    values: dict[str, Any] = Field(default_factory=dict)
    geometry_handle: str | None = None
    reference_id: str | None = None


class ConstraintTolerance(StrictModel):
    position_m: float = 0.05
    rotation_deg: float = 8.0
    trajectory_rmse_m: float = 0.10
    contact_distance_m: float = 0.06
    drift_m: float = 0.04


class ConstraintRequest(StrictModel):
    mode: Literal["add", "replace", "remove"] = "add"
    constraint_id: str | None = None
    target_segment: int
    constraint_type: ConstraintType
    requested_strength: Literal["hard", "soft"] = "soft"
    time_spec: TimeSpec
    body_parts: list[str] = Field(default_factory=list)
    target: ConstraintTarget | None = None
    reference_ids: list[str] = Field(default_factory=list)
    tolerance: ConstraintTolerance = Field(default_factory=ConstraintTolerance)
    priority: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class RewardSpec(StrictModel):
    reward_id: str
    constraint_id: str
    reward_type: str
    target: dict[str, Any] = Field(default_factory=dict)
    frame_indices: list[int] = Field(default_factory=list)
    body_parts: list[str] = Field(default_factory=list)
    weight: float = 1.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class VerificationSpec(StrictModel):
    verification_spec_id: str
    constraint_id: str
    metric_type: str
    target_segments: list[int]
    body_parts: list[str] = Field(default_factory=list)
    frame_indices: list[int] = Field(default_factory=list)
    target: dict[str, Any] = Field(default_factory=dict)
    pass_threshold: float
    units: str


class HardMotionCondition(TensorModel):
    values: torch.Tensor
    mask: torch.Tensor
    source_constraint_id: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_tensor_shapes(self) -> "HardMotionCondition":
        if tuple(self.values.shape) != tuple(self.mask.shape):
            raise ValueError("values and mask must have identical shape")
        if self.values.ndim != 2:
            raise ValueError("values and mask must be [frames, features]")
        return self


class CompiledConstraint(StrictModel):
    constraint_id: str
    target_segment: int
    constraint_type: str
    requested_strength: str
    effective_mode: EffectiveMode
    resolved_time: ResolvedTimeSpec
    target_spec: ConstraintTarget | None
    hard_condition_handle: str | None = None
    reward_spec: RewardSpec | None = None
    verification_spec: VerificationSpec
    source_reference_ids: list[str] = Field(default_factory=list)
    tolerance: ConstraintTolerance = Field(default_factory=ConstraintTolerance)
    priority: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class ConstraintConflict(StrictModel):
    constraint_ids: list[str]
    frame_indices: list[int]
    feature_indices: list[int]
    max_value_difference: float
    conflict_type: str


class ConstraintBundle(StrictModel):
    hard_condition_handle: str | None = None
    reward_specs: list[RewardSpec] = Field(default_factory=list)
    verification_specs: list[VerificationSpec] = Field(default_factory=list)
    active_constraint_ids: list[str] = Field(default_factory=list)
    active_keyframe_ids: list[str] = Field(default_factory=list)
    mask_density: float = 0.0
    conflicts: list[ConstraintConflict] = Field(default_factory=list)
    generation_blocked: bool = False


class ConstraintCompileResult(StrictModel):
    status: ConstraintStatus
    constraint: CompiledConstraint | None = None
    warnings: list[str] = Field(default_factory=list)
    routing_hints: list[str] = Field(default_factory=list)
