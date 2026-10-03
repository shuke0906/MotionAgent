"""GEM 151-D feature mapping utilities."""

from __future__ import annotations

from dataclasses import dataclass

from motion_agent.constraints.joints import SMPLJointRegistry


@dataclass(frozen=True)
class FeatureSlice:
    start: int
    end: int


class GEMFeatureMapper:
    FEATURE_DIM = 151
    GROUPS = {
        "body_pose": FeatureSlice(0, 126),
        "betas": FeatureSlice(126, 136),
        "global_orient": FeatureSlice(136, 142),
        "global_orient_gv": FeatureSlice(142, 148),
        "local_transl_vel": FeatureSlice(148, 151),
    }

    def __init__(self, joints: SMPLJointRegistry | None = None) -> None:
        self.joints = joints or SMPLJointRegistry()

    def get_group_slice(self, name: str) -> tuple[int, int]:
        group = self.GROUPS[name]
        return group.start, group.end

    def get_body_joint_slice(self, joint_name_or_id: str | int) -> tuple[int, int] | None:
        joint_id = joint_name_or_id if isinstance(joint_name_or_id, int) else self.joints.joint_id(joint_name_or_id)
        if joint_id == 0:
            return None
        if not 1 <= joint_id <= 21:
            raise ValueError(f"SMPL body joint id out of GEM body_pose range: {joint_id}")
        start = (joint_id - 1) * 6
        return start, start + 6
