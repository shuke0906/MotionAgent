"""Deterministic Phase 6 Constraint Compiler."""

from __future__ import annotations

from typing import Any

import torch

from motion_agent.common.ids import new_id
from motion_agent.constraints.coordinates import CoordinateResolver
from motion_agent.constraints.feature_mask import FeatureMaskBuilder
from motion_agent.constraints.gem_encoder import GEMConstraintEncoder
from motion_agent.constraints.kinematics import SimpleKinematics
from motion_agent.constraints.schemas import (
    CompiledConstraint,
    ConstraintCompileResult,
    ConstraintRequest,
    ConstraintTarget,
    HardMotionCondition,
    RewardSpec,
    ResolvedTimeSpec,
    VerificationSpec,
)
from motion_agent.constraints.store import ConditionStore
from motion_agent.constraints.time_resolver import TimeResolver
from motion_agent.state.schemas import MotionAgentState


class ConstraintCompiler:
    def __init__(
        self,
        *,
        store: ConditionStore | None = None,
        encoder: GEMConstraintEncoder | None = None,
    ) -> None:
        self.store = store or ConditionStore()
        self.encoder = encoder or GEMConstraintEncoder()
        self.kinematics = SimpleKinematics(self.encoder.mapper.joints)
        self.coordinates = CoordinateResolver()
        self.time_resolver = TimeResolver()

    def compile(self, state: MotionAgentState, request: ConstraintRequest) -> ConstraintCompileResult:
        if request.mode == "remove":
            return ConstraintCompileResult(status="compiled", warnings=["constraint removal requested"])
        try:
            resolved_time = self.time_resolver.resolve(state, request.target_segment, request.time_spec)
            target_resolution = self.coordinates.resolve_target(request.target)
        except ValueError as exc:
            return ConstraintCompileResult(status="error", warnings=[str(exc)])

        if target_resolution.status == "needs_geometry":
            return ConstraintCompileResult(
                status="needs_geometry",
                warnings=target_resolution.warnings,
                routing_hints=["PROVIDE_SCENE_GEOMETRY", "RETRIEVE_REFERENCE"],
            )
        if target_resolution.status == "needs_reference" and request.constraint_type in {"joint_target", "body_part_pose"}:
            return ConstraintCompileResult(
                status="needs_reference",
                warnings=target_resolution.warnings,
                routing_hints=["RETRIEVE_REFERENCE"],
            )

        if request.constraint_type == "joint_target":
            return self._compile_joint_target(state, request, resolved_time, target_resolution.point)
        if request.constraint_type == "body_part_pose":
            return self._compile_body_part_pose(state, request, resolved_time)
        if request.constraint_type == "root_trajectory":
            return self._compile_root_trajectory(request, resolved_time)
        if request.constraint_type == "contact":
            return self._compile_contact(request, resolved_time)
        if request.constraint_type == "fixed_joint":
            return self._compile_fixed_joint(request, resolved_time)
        return ConstraintCompileResult(status="unsupported", warnings=[f"unsupported constraint_type={request.constraint_type}"])

    def _compile_joint_target(
        self,
        state: MotionAgentState,
        request: ConstraintRequest,
        resolved_time: ResolvedTimeSpec,
        target_point: torch.Tensor | None,
    ) -> ConstraintCompileResult:
        if target_point is None:
            return ConstraintCompileResult(status="needs_geometry", warnings=["joint_target requires a resolved XYZ point"])
        joint_name = (request.body_parts or ["right_wrist"])[0]
        constraint_id = request.constraint_id or new_id("constraint")
        base_pose = self._pose_from_request(request, state.task.total_frames)[resolved_time.start_frame].reshape(21, 6)
        ik = self.kinematics.solve_ik(
            base_pose=base_pose,
            target_joint=joint_name,
            target_position=target_point,
            tolerance_m=request.tolerance.position_m,
        )
        reward = self._reward(
            constraint_id,
            "joint_position_error",
            request,
            resolved_time,
            target={"target_xyz": target_point.tolist(), "ik_residual_m": ik.residual_m},
        )
        verification = self._verification(
            constraint_id,
            "joint_position_error",
            request,
            resolved_time,
            pass_threshold=request.tolerance.position_m,
            units="m",
            target={"target_xyz": target_point.tolist()},
        )
        if request.requested_strength == "hard" and ik.success:
            sequence = self.encoder.canonical_sequence(state.task.total_frames)
            sequence["body_pose"][resolved_time.frame_indices] = ik.body_pose.reshape(1, 126)
            encoded = self.encoder.encode_sequence(sequence, total_frames=state.task.total_frames)
            mask = FeatureMaskBuilder(state.task.total_frames, self.encoder.mapper).for_body_joints(
                ik.affected_joint_names,
                resolved_time.frame_indices,
            )
            handle = self.store.save_hard_condition(
                HardMotionCondition(
                    values=encoded,
                    mask=mask,
                    source_constraint_id=constraint_id,
                    metadata={"constraint_type": "joint_target", "ik_residual_m": ik.residual_m},
                )
            )
            compiled = self._compiled(
                constraint_id,
                request,
                resolved_time,
                "hard_condition",
                verification,
                target_spec=request.target,
                hard_condition_handle=handle,
                reward_spec=reward,
                metadata={"ik_success": True, "ik_residual_m": ik.residual_m},
            )
            return ConstraintCompileResult(status="compiled", constraint=compiled)
        compiled = self._compiled(
            constraint_id,
            request,
            resolved_time,
            "selection_reward",
            verification,
            target_spec=request.target,
            reward_spec=reward,
            metadata={"ik_success": ik.success, "ik_residual_m": ik.residual_m, "ik_message": ik.message},
        )
        return ConstraintCompileResult(status="compiled_soft", constraint=compiled, warnings=[ik.message] if ik.message else [])

    def _compile_body_part_pose(
        self,
        state: MotionAgentState,
        request: ConstraintRequest,
        resolved_time: ResolvedTimeSpec,
    ) -> ConstraintCompileResult:
        constraint_id = request.constraint_id or new_id("constraint")
        joint_names = self.encoder.mapper.joints.expand_body_parts(request.body_parts or ["right_arm"])
        body_pose = self._pose_from_request(request, state.task.total_frames)
        sequence = self.encoder.canonical_sequence(state.task.total_frames)
        sequence["body_pose"] = body_pose.reshape(state.task.total_frames, 126)
        encoded = self.encoder.encode_sequence(sequence, total_frames=state.task.total_frames)
        mask = FeatureMaskBuilder(state.task.total_frames, self.encoder.mapper).for_body_joints(
            joint_names,
            resolved_time.frame_indices,
        )
        handle = self.store.save_hard_condition(
            HardMotionCondition(
                values=encoded,
                mask=mask,
                source_constraint_id=constraint_id,
                metadata={"constraint_type": "body_part_pose", "body_parts": request.body_parts},
            )
        )
        verification = self._verification(
            constraint_id,
            "pose_rotation_error",
            request,
            resolved_time,
            pass_threshold=request.tolerance.rotation_deg,
            units="deg",
            target={"body_parts": joint_names},
        )
        compiled = self._compiled(
            constraint_id,
            request,
            resolved_time,
            "hard_condition",
            verification,
            target_spec=request.target,
            hard_condition_handle=handle,
            metadata={"masked_joints": joint_names},
        )
        return ConstraintCompileResult(status="compiled", constraint=compiled)

    def _compile_root_trajectory(self, request: ConstraintRequest, resolved_time: ResolvedTimeSpec) -> ConstraintCompileResult:
        constraint_id = request.constraint_id or new_id("constraint")
        values = request.target.values if request.target else {}
        has_orientation = bool(values.get("orientations") or values.get("orientation_policy") == "follow_tangent")
        reward = self._reward(constraint_id, "trajectory_rmse", request, resolved_time, target=values)
        verification = self._verification(
            constraint_id,
            "trajectory_rmse",
            request,
            resolved_time,
            pass_threshold=request.tolerance.trajectory_rmse_m,
            units="m",
            target=values,
        )
        compiled = self._compiled(
            constraint_id,
            request,
            resolved_time,
            "selection_reward",
            verification,
            target_spec=request.target,
            reward_spec=reward,
            metadata={"orientation_known": has_orientation},
        )
        return ConstraintCompileResult(status="compiled_soft", constraint=compiled)

    def _compile_contact(self, request: ConstraintRequest, resolved_time: ResolvedTimeSpec) -> ConstraintCompileResult:
        if request.target and request.target.target_type == "scene_anchor" and not request.target.geometry_handle:
            return ConstraintCompileResult(
                status="needs_geometry",
                warnings=["contact target references scene geometry that is not available"],
                routing_hints=["PROVIDE_SCENE_GEOMETRY", "RETRIEVE_REFERENCE"],
            )
        constraint_id = request.constraint_id or new_id("constraint")
        reward = self._reward(constraint_id, "contact_distance_velocity", request, resolved_time)
        verification = self._verification(
            constraint_id,
            "contact_distance_velocity",
            request,
            resolved_time,
            pass_threshold=request.tolerance.contact_distance_m,
            units="m",
            target=(request.target.model_dump(mode="json") if request.target else {}),
        )
        compiled = self._compiled(
            constraint_id,
            request,
            resolved_time,
            "selection_reward",
            verification,
            target_spec=request.target,
            reward_spec=reward,
        )
        return ConstraintCompileResult(status="compiled_soft", constraint=compiled)

    def _compile_fixed_joint(self, request: ConstraintRequest, resolved_time: ResolvedTimeSpec) -> ConstraintCompileResult:
        constraint_id = request.constraint_id or new_id("constraint")
        reward = self._reward(constraint_id, "joint_drift", request, resolved_time)
        verification = self._verification(
            constraint_id,
            "joint_drift",
            request,
            resolved_time,
            pass_threshold=request.tolerance.drift_m,
            units="m",
            target={"fixed_joint": (request.body_parts or ["left_foot"])[0]},
        )
        compiled = self._compiled(
            constraint_id,
            request,
            resolved_time,
            "selection_reward",
            verification,
            target_spec=request.target,
            reward_spec=reward,
        )
        return ConstraintCompileResult(status="compiled_soft", constraint=compiled)

    def _pose_from_request(self, request: ConstraintRequest, total_frames: int) -> torch.Tensor:
        pose_values: Any | None = None
        if request.target:
            pose_values = request.target.values.get("body_pose")
        if pose_values is None:
            pose_values = request.metadata.get("body_pose")
        if pose_values is None:
            return torch.zeros(total_frames, 21, 6, dtype=torch.float32)
        pose = torch.as_tensor(pose_values, dtype=torch.float32)
        if pose.ndim == 2 and pose.shape == (21, 6):
            pose = pose[None].repeat(total_frames, 1, 1)
        elif pose.ndim == 2 and pose.shape[1] == 126:
            pose = pose.reshape(pose.shape[0], 21, 6)
        elif pose.ndim == 1 and pose.numel() == 126:
            pose = pose.reshape(1, 21, 6).repeat(total_frames, 1, 1)
        if pose.shape[0] == 1 and total_frames > 1:
            pose = pose.repeat(total_frames, 1, 1)
        if pose.shape != (total_frames, 21, 6):
            raise ValueError(f"body_pose must resolve to [{total_frames}, 21, 6]")
        return pose

    def _reward(
        self,
        constraint_id: str,
        reward_type: str,
        request: ConstraintRequest,
        resolved_time: ResolvedTimeSpec,
        *,
        target: dict[str, Any] | None = None,
    ) -> RewardSpec:
        return RewardSpec(
            reward_id=new_id("reward"),
            constraint_id=constraint_id,
            reward_type=reward_type,
            target=target or (request.target.model_dump(mode="json") if request.target else {}),
            frame_indices=resolved_time.frame_indices,
            body_parts=request.body_parts,
        )

    def _verification(
        self,
        constraint_id: str,
        metric_type: str,
        request: ConstraintRequest,
        resolved_time: ResolvedTimeSpec,
        *,
        pass_threshold: float,
        units: str,
        target: dict[str, Any],
    ) -> VerificationSpec:
        return VerificationSpec(
            verification_spec_id=new_id("verification_spec"),
            constraint_id=constraint_id,
            metric_type=metric_type,
            target_segments=[request.target_segment],
            body_parts=request.body_parts,
            frame_indices=resolved_time.frame_indices,
            target=target,
            pass_threshold=pass_threshold,
            units=units,
        )

    def _compiled(
        self,
        constraint_id: str,
        request: ConstraintRequest,
        resolved_time: ResolvedTimeSpec,
        effective_mode: str,
        verification: VerificationSpec,
        *,
        target_spec: ConstraintTarget | None,
        hard_condition_handle: str | None = None,
        reward_spec: RewardSpec | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> CompiledConstraint:
        return CompiledConstraint(
            constraint_id=constraint_id,
            target_segment=request.target_segment,
            constraint_type=request.constraint_type,
            requested_strength=request.requested_strength,
            effective_mode=effective_mode,
            resolved_time=resolved_time,
            target_spec=target_spec,
            hard_condition_handle=hard_condition_handle,
            reward_spec=reward_spec,
            verification_spec=verification,
            source_reference_ids=request.reference_ids,
            tolerance=request.tolerance,
            priority=request.priority,
            metadata=metadata or {},
        )


def compile_constraint(
    state: MotionAgentState,
    request: ConstraintRequest,
    *,
    store: ConditionStore | None = None,
) -> ConstraintCompileResult:
    return ConstraintCompiler(store=store).compile(state, request)
