"""Deterministic Phase 6 Keyframe Tool."""

from __future__ import annotations

from pathlib import Path

import torch

from motion_agent.common.ids import new_id
from motion_agent.constraints.feature_mask import FeatureMaskBuilder
from motion_agent.constraints.gem_encoder import GEMConstraintEncoder
from motion_agent.constraints.kinematics import SimpleKinematics
from motion_agent.constraints.schemas import HardMotionCondition
from motion_agent.constraints.store import ConditionStore
from motion_agent.generation.candidate_store import CandidateStore
from motion_agent.keyframes.schemas import (
    KeyframeBuildResult,
    KeyframeRequest,
    KeyframeSpec,
    KeyframeVerificationSpec,
)
from motion_agent.keyframes.source_resolver import KeyframeSourceResolver
from motion_agent.keyframes.store import KeyframeStore
from motion_agent.state.schemas import MotionAgentState


class KeyframeTool:
    def __init__(
        self,
        *,
        condition_store: ConditionStore | None = None,
        keyframe_store: KeyframeStore | None = None,
        candidate_store: CandidateStore | None = None,
        encoder: GEMConstraintEncoder | None = None,
    ) -> None:
        self.condition_store = condition_store or ConditionStore()
        self.keyframe_store = keyframe_store or KeyframeStore()
        self.encoder = encoder or GEMConstraintEncoder()
        self.resolver = KeyframeSourceResolver(candidate_store)
        self.kinematics = SimpleKinematics(self.encoder.mapper.joints)

    def build(self, state: MotionAgentState, request: KeyframeRequest) -> KeyframeBuildResult:
        if request.mode == "remove":
            return KeyframeBuildResult(status="compiled", warnings=["keyframe removal requested"])
        target_frame = round(request.time * state.task.fps)
        if target_frame < 0 or target_frame >= state.task.total_frames:
            return KeyframeBuildResult(status="error", warnings=["keyframe target frame is outside motion bounds"])
        source = self.resolver.resolve(state, request, target_frame)
        if source.status == "needs_reference":
            return KeyframeBuildResult(status="needs_reference", warnings=source.warnings, routing_hints=["RETRIEVE_REFERENCE"])
        if source.status == "needs_candidate":
            return KeyframeBuildResult(status="needs_candidate", warnings=source.warnings, routing_hints=["GENERATE"])
        if source.status != "resolved" or source.pose is None or source.source_type is None:
            return KeyframeBuildResult(status="unsupported", warnings=source.warnings)

        pose = source.pose
        ik_used = False
        ik_residual = None
        if request.pose_targets:
            refined = self._refine_with_ik(pose, request)
            if refined is None:
                return KeyframeBuildResult(status="needs_reference", warnings=["IK refinement failed"], routing_hints=["RETRIEVE_REFERENCE"])
            pose, ik_residual = refined
            ik_used = True

        keyframe_id = request.keyframe_id or new_id("keyframe")
        pose_handle = self.keyframe_store.save_pose(pose, source=source.source_type)
        encoded_pose = self.encoder.encode_pose(pose)
        values = torch.zeros(state.task.total_frames, 151, dtype=torch.float32)
        values[target_frame] = encoded_pose
        mask = FeatureMaskBuilder(state.task.total_frames, self.encoder.mapper).for_whole_body_keyframe(
            target_frame,
            control_root_orientation=request.control_root_orientation,
            control_root_translation=request.control_root_translation,
            control_betas=False,
        )
        condition_handle = self.condition_store.save_hard_condition(
            HardMotionCondition(
                values=values,
                mask=mask,
                source_constraint_id=keyframe_id,
                metadata={"condition_type": "keyframe", "source_type": source.source_type},
            )
        )
        verification = KeyframeVerificationSpec(
            keyframe_id=keyframe_id,
            target_frame=target_frame,
            temporal_tolerance_frames=request.temporal_tolerance_frames if request.temporal_tolerance_frames is not None else 3,
            pose_handle=pose_handle,
            metrics=self._metrics(request),
            control_root_orientation=request.control_root_orientation,
            control_root_translation=request.control_root_translation,
        )
        spec = KeyframeSpec(
            keyframe_id=keyframe_id,
            target_segment=request.target_segment,
            target_time_s=request.time,
            target_frame=target_frame,
            temporal_tolerance_frames=verification.temporal_tolerance_frames,
            description=request.description,
            source_type="ik_refined" if ik_used else source.source_type,
            source_reference_ids=source.source_reference_ids,
            source_candidate_id=source.source_candidate_id,
            pose_handle=pose_handle,
            control_root_orientation=request.control_root_orientation,
            control_root_translation=request.control_root_translation,
            hard_condition_handle=condition_handle,
            verification_spec=verification,
            metadata={
                "source_frame": source.source_frame,
                "ik_used": ik_used,
                "ik_residual_m": ik_residual,
                "encoder_version": self.encoder.version,
                "config_version": "phase6_keyframe_v1",
            },
        )
        return KeyframeBuildResult(status="compiled", keyframe=spec, warnings=source.warnings)

    def _refine_with_ik(self, pose: dict, request: KeyframeRequest) -> tuple[dict, float] | None:
        body_pose = pose["body_pose"].clone().reshape(21, 6)
        max_residual = 0.0
        for target in request.pose_targets:
            if target.target_type != "position":
                continue
            xyz = target.values.get("xyz", target.values.get("position"))
            if xyz is None:
                return None
            result = self.kinematics.solve_ik(
                base_pose=body_pose,
                target_joint=target.body_part,
                target_position=torch.as_tensor(xyz, dtype=torch.float32),
                tolerance_m=float(target.values.get("tolerance_m", 0.05)),
            )
            if not result.success:
                return None
            body_pose = result.body_pose
            max_residual = max(max_residual, result.residual_m)
        refined = dict(pose)
        refined["body_pose"] = body_pose
        return refined, max_residual

    def _metrics(self, request: KeyframeRequest) -> list[str]:
        metrics = ["joint_rotation_error", "joint_position_error"]
        if request.control_root_orientation:
            metrics.append("root_orientation_error")
        if request.control_root_translation:
            metrics.append("root_position_error")
        return metrics


def build_keyframe(
    state: MotionAgentState,
    request: KeyframeRequest,
    *,
    condition_store: ConditionStore | None = None,
    keyframe_store: KeyframeStore | None = None,
    candidate_store_root: str | Path | None = None,
) -> KeyframeBuildResult:
    candidate_store = CandidateStore(candidate_store_root) if candidate_store_root is not None else None
    return KeyframeTool(
        condition_store=condition_store,
        keyframe_store=keyframe_store,
        candidate_store=candidate_store,
    ).build(state, request)
