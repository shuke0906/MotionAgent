"""Schemas for Phase 6 keyframes."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from motion_agent.state.schemas import StrictModel


class KeyframePoseTarget(StrictModel):
    body_part: str
    target_type: Literal["position", "rotation", "pose_reference"]
    values: dict[str, Any] = Field(default_factory=dict)
    coordinate_frame: str | None = None


class KeyframeRequest(StrictModel):
    mode: Literal["add", "replace", "remove"] = "add"
    keyframe_id: str | None = None
    target_segment: int
    time: float
    description: str
    source_preference: Literal["retrieval", "ik", "generation", "auto", "explicit"] = "auto"
    reference_ids: list[str] = Field(default_factory=list)
    source_candidate_id: str | None = None
    pose_targets: list[KeyframePoseTarget] = Field(default_factory=list)
    control_root_orientation: bool = True
    control_root_translation: bool = False
    temporal_tolerance_frames: int | None = None
    explicit_pose: dict[str, Any] | None = None
    retrieved_references: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class KeyframeVerificationSpec(StrictModel):
    keyframe_id: str
    target_frame: int
    temporal_tolerance_frames: int
    pose_handle: str
    metrics: list[
        Literal[
            "joint_rotation_error",
            "joint_position_error",
            "root_orientation_error",
            "root_position_error",
        ]
    ]
    rotation_threshold_deg: float | None = 8.0
    position_threshold_m: float | None = 0.08
    control_root_orientation: bool = True
    control_root_translation: bool = False


class KeyframeSpec(StrictModel):
    keyframe_id: str
    target_segment: int
    target_time_s: float
    target_frame: int
    temporal_tolerance_frames: int
    description: str
    source_type: Literal["retrieval_pose", "retrieval_motion", "candidate", "ik_refined", "explicit_pose"]
    source_reference_ids: list[str] = Field(default_factory=list)
    source_candidate_id: str | None = None
    pose_handle: str
    control_body_pose: bool = True
    control_root_orientation: bool = True
    control_root_translation: bool = False
    hard_condition_handle: str
    verification_spec: KeyframeVerificationSpec
    status: Literal["active", "superseded", "removed"] = "active"
    metadata: dict[str, Any] = Field(default_factory=dict)


class KeyframeBuildResult(StrictModel):
    status: Literal["compiled", "needs_reference", "needs_candidate", "conflict", "unsupported", "error"]
    keyframe: KeyframeSpec | None = None
    warnings: list[str] = Field(default_factory=list)
    routing_hints: list[str] = Field(default_factory=list)
