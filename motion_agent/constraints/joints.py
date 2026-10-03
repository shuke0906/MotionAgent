"""Canonical joint names and SMPL/GEM feature ownership."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class JointInfo:
    name: str
    smpl_id: int
    parent: str | None


class SMPLJointRegistry:
    """Single source of truth for joint names used by constraints/keyframes."""

    _joints = {
        "pelvis": JointInfo("pelvis", 0, None),
        "left_hip": JointInfo("left_hip", 1, "pelvis"),
        "right_hip": JointInfo("right_hip", 2, "pelvis"),
        "spine1": JointInfo("spine1", 3, "pelvis"),
        "left_knee": JointInfo("left_knee", 4, "left_hip"),
        "right_knee": JointInfo("right_knee", 5, "right_hip"),
        "spine2": JointInfo("spine2", 6, "spine1"),
        "left_ankle": JointInfo("left_ankle", 7, "left_knee"),
        "right_ankle": JointInfo("right_ankle", 8, "right_knee"),
        "spine3": JointInfo("spine3", 9, "spine2"),
        "left_foot": JointInfo("left_foot", 10, "left_ankle"),
        "right_foot": JointInfo("right_foot", 11, "right_ankle"),
        "neck": JointInfo("neck", 12, "spine3"),
        "left_collar": JointInfo("left_collar", 13, "spine3"),
        "right_collar": JointInfo("right_collar", 14, "spine3"),
        "head": JointInfo("head", 15, "neck"),
        "left_shoulder": JointInfo("left_shoulder", 16, "left_collar"),
        "right_shoulder": JointInfo("right_shoulder", 17, "right_collar"),
        "left_elbow": JointInfo("left_elbow", 18, "left_shoulder"),
        "right_elbow": JointInfo("right_elbow", 19, "right_shoulder"),
        "left_wrist": JointInfo("left_wrist", 20, "left_elbow"),
        "right_wrist": JointInfo("right_wrist", 21, "right_elbow"),
    }

    _aliases = {
        "root": "pelvis",
        "hips": "pelvis",
        "left_hand": "left_wrist",
        "right_hand": "right_wrist",
        "left_foot_end": "left_foot",
        "right_foot_end": "right_foot",
        "fixed_foot": "left_foot",
    }

    _body_parts = {
        "full_body": list(_joints.keys()),
        "body": list(_joints.keys()),
        "right_arm": ["right_shoulder", "right_elbow", "right_wrist"],
        "left_arm": ["left_shoulder", "left_elbow", "left_wrist"],
        "arms": ["left_shoulder", "left_elbow", "left_wrist", "right_shoulder", "right_elbow", "right_wrist"],
        "right_leg": ["right_hip", "right_knee", "right_ankle", "right_foot"],
        "left_leg": ["left_hip", "left_knee", "left_ankle", "left_foot"],
        "legs": ["left_hip", "left_knee", "left_ankle", "left_foot", "right_hip", "right_knee", "right_ankle", "right_foot"],
        "right_foot": ["right_foot"],
        "left_foot": ["left_foot"],
        "right_hand": ["right_wrist"],
        "left_hand": ["left_wrist"],
    }

    def normalize(self, name: str) -> str:
        key = name.lower().strip().replace(" ", "_")
        key = self._aliases.get(key, key)
        if key not in self._joints and key not in self._body_parts:
            raise KeyError(f"unknown joint/body part: {name}")
        return key

    def joint_id(self, name: str) -> int:
        return self._joints[self.normalize(name)].smpl_id

    def joint_name(self, smpl_id: int) -> str:
        for name, info in self._joints.items():
            if info.smpl_id == smpl_id:
                return name
        raise KeyError(f"unknown SMPL joint id: {smpl_id}")

    def expand_body_parts(self, names: list[str]) -> list[str]:
        if not names:
            return []
        expanded: list[str] = []
        for name in names:
            normalized = self.normalize(name)
            candidates = self._body_parts.get(normalized, [normalized])
            for candidate in candidates:
                if candidate not in expanded:
                    expanded.append(candidate)
        return expanded

    def chain_to(self, joint_name: str) -> list[str]:
        current = self.normalize(joint_name)
        chain = [current]
        while self._joints[current].parent is not None:
            current = self._joints[current].parent or "pelvis"
            chain.append(current)
        chain.reverse()
        return chain
