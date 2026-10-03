"""Feature mask construction for GEM hard conditions."""

from __future__ import annotations

import torch

from motion_agent.constraints.gem_features import GEMFeatureMapper
from motion_agent.constraints.joints import SMPLJointRegistry


class FeatureMaskBuilder:
    def __init__(self, total_frames: int, mapper: GEMFeatureMapper | None = None) -> None:
        self.total_frames = total_frames
        self.mapper = mapper or GEMFeatureMapper()
        self.joints = self.mapper.joints

    def empty(self) -> torch.Tensor:
        return torch.zeros(self.total_frames, self.mapper.FEATURE_DIM, dtype=torch.float32)

    def for_body_joints(self, joint_names: list[str], frames: list[int]) -> torch.Tensor:
        mask = self.empty()
        for joint in self.joints.expand_body_parts(joint_names):
            feature_slice = self.mapper.get_body_joint_slice(joint)
            if feature_slice is None:
                continue
            start, end = feature_slice
            mask[frames, start:end] = 1.0
        return mask

    def for_kinematic_chain(self, joint_name: str, frames: list[int]) -> torch.Tensor:
        return self.for_body_joints(self.joints.chain_to(joint_name), frames)

    def for_feature_groups(self, group_names: list[str], frames: list[int]) -> torch.Tensor:
        mask = self.empty()
        for name in group_names:
            start, end = self.mapper.get_group_slice(name)
            mask[frames, start:end] = 1.0
        return mask

    def for_whole_body_keyframe(
        self,
        frame: int,
        *,
        control_root_orientation: bool = True,
        control_root_translation: bool = False,
        control_betas: bool = False,
    ) -> torch.Tensor:
        mask = self.empty()
        start, end = self.mapper.get_group_slice("body_pose")
        mask[frame, start:end] = 1.0
        if control_root_orientation:
            for group in ("global_orient", "global_orient_gv"):
                start, end = self.mapper.get_group_slice(group)
                mask[frame, start:end] = 1.0
        if control_root_translation:
            start, end = self.mapper.get_group_slice("local_transl_vel")
            mask[frame, start:end] = 1.0
        if control_betas:
            start, end = self.mapper.get_group_slice("betas")
            mask[frame, start:end] = 1.0
        return mask
