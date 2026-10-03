"""Small deterministic FK/IK adapter used by Phase 6 contract tests."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from motion_agent.constraints.joints import SMPLJointRegistry


CANONICAL_OFFSETS = {
    "pelvis": (0.0, 0.0, 0.0),
    "left_hip": (-0.10, -0.10, 0.0),
    "right_hip": (0.10, -0.10, 0.0),
    "spine1": (0.0, 0.15, 0.0),
    "left_knee": (0.0, -0.35, 0.0),
    "right_knee": (0.0, -0.35, 0.0),
    "spine2": (0.0, 0.15, 0.0),
    "left_ankle": (0.0, -0.35, 0.0),
    "right_ankle": (0.0, -0.35, 0.0),
    "spine3": (0.0, 0.15, 0.0),
    "left_foot": (0.0, -0.05, 0.12),
    "right_foot": (0.0, -0.05, 0.12),
    "neck": (0.0, 0.12, 0.0),
    "left_collar": (-0.08, 0.04, 0.0),
    "right_collar": (0.08, 0.04, 0.0),
    "head": (0.0, 0.16, 0.0),
    "left_shoulder": (-0.18, 0.0, 0.0),
    "right_shoulder": (0.18, 0.0, 0.0),
    "left_elbow": (-0.26, 0.0, 0.0),
    "right_elbow": (0.26, 0.0, 0.0),
    "left_wrist": (-0.24, 0.0, 0.0),
    "right_wrist": (0.24, 0.0, 0.0),
}


@dataclass(frozen=True)
class IKResult:
    success: bool
    body_pose: torch.Tensor
    residual_m: float
    affected_joint_names: list[str]
    message: str = ""


class SimpleKinematics:
    def __init__(self, joints: SMPLJointRegistry | None = None) -> None:
        self.joints = joints or SMPLJointRegistry()

    def canonical_body_pose(self) -> torch.Tensor:
        return torch.zeros(21, 6, dtype=torch.float32)

    def fk(self, body_pose: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        pose = body_pose if body_pose is not None else self.canonical_body_pose()
        if pose.ndim == 1:
            pose = pose.reshape(21, 6)
        positions: dict[str, torch.Tensor] = {}
        for name, info in sorted(self.joints._joints.items(), key=lambda item: item[1].smpl_id):
            offset = torch.tensor(CANONICAL_OFFSETS[name], dtype=torch.float32)
            if info.smpl_id > 0:
                feature = pose[info.smpl_id - 1, :3]
                offset = offset + feature
            parent_pos = torch.zeros(3) if info.parent is None else positions[info.parent]
            positions[name] = parent_pos + offset
        return positions

    def solve_ik(
        self,
        *,
        base_pose: torch.Tensor,
        target_joint: str,
        target_position: torch.Tensor,
        tolerance_m: float,
        max_reach_delta_m: float = 1.5,
    ) -> IKResult:
        joint = self.joints.normalize(target_joint)
        pose = base_pose.clone().reshape(21, 6)
        current = self.fk(pose)[joint]
        delta = target_position.to(dtype=torch.float32) - current
        if float(torch.linalg.vector_norm(delta)) > max_reach_delta_m:
            residual = float(torch.linalg.vector_norm(delta))
            return IKResult(False, pose, residual, self.joints.chain_to(joint), "target outside configured IK reach")
        joint_id = self.joints.joint_id(joint)
        if joint_id == 0:
            residual = float(torch.linalg.vector_norm(delta))
            return IKResult(False, pose, residual, [joint], "pelvis root IK is handled by root constraints")
        pose[joint_id - 1, :3] += delta
        solved_position = self.fk(pose)[joint]
        residual = float(torch.linalg.vector_norm(solved_position - target_position))
        return IKResult(
            success=residual <= tolerance_m,
            body_pose=pose,
            residual_m=residual,
            affected_joint_names=self.joints.chain_to(joint),
            message="" if residual <= tolerance_m else "IK residual above tolerance",
        )
