"""Fixture-safe GEM 151-D encoder/decoder adapter."""

from __future__ import annotations

from typing import Any

import torch

from motion_agent.constraints.gem_features import GEMFeatureMapper


class GEMConstraintEncoder:
    """Encodes synthetic or retrieved SMPL-like parameters into GEM's 151-D layout.

    This adapter is intentionally deterministic and normalization-free for Phase 6
    contract tests. The production GEM en/decoder can replace the internals while
    preserving this interface.
    """

    def __init__(self, mapper: GEMFeatureMapper | None = None) -> None:
        self.mapper = mapper or GEMFeatureMapper()
        self.version = "phase6_fixture_gem_encoder_v1"

    def canonical_sequence(self, total_frames: int) -> dict[str, torch.Tensor]:
        return {
            "body_pose": torch.zeros(total_frames, 126, dtype=torch.float32),
            "betas": torch.zeros(total_frames, 10, dtype=torch.float32),
            "global_orient": torch.zeros(total_frames, 6, dtype=torch.float32),
            "global_orient_gv": torch.zeros(total_frames, 6, dtype=torch.float32),
            "local_transl_vel": torch.zeros(total_frames, 3, dtype=torch.float32),
        }

    def encode_sequence(self, smpl: dict[str, Any], total_frames: int | None = None) -> torch.Tensor:
        frames = total_frames or self._infer_frames(smpl)
        encoded = torch.zeros(frames, self.mapper.FEATURE_DIM, dtype=torch.float32)
        body_pose = self._coerce_body_pose(smpl.get("body_pose"), frames)
        betas = self._coerce_feature(smpl.get("betas"), frames, 10)
        global_orient = self._coerce_feature(smpl.get("global_orient"), frames, 6)
        global_orient_gv = self._coerce_feature(smpl.get("global_orient_gv", smpl.get("global_orient")), frames, 6)
        transl_vel = self._coerce_feature(smpl.get("local_transl_vel", smpl.get("transl")), frames, 3)
        for name, value in {
            "body_pose": body_pose,
            "betas": betas,
            "global_orient": global_orient,
            "global_orient_gv": global_orient_gv,
            "local_transl_vel": transl_vel,
        }.items():
            start, end = self.mapper.get_group_slice(name)
            encoded[:, start:end] = value[:, : end - start]
        return encoded

    def encode_pose(self, smpl_pose: dict[str, Any]) -> torch.Tensor:
        return self.encode_sequence({key: self._ensure_frame_dim(value) for key, value in smpl_pose.items()}, total_frames=1)[0]

    def decode_sequence(self, encoded: torch.Tensor) -> dict[str, torch.Tensor]:
        if encoded.ndim != 2 or encoded.shape[1] != self.mapper.FEATURE_DIM:
            raise ValueError("encoded GEM motion must be [frames, 151]")
        decoded: dict[str, torch.Tensor] = {}
        for name in self.mapper.GROUPS:
            start, end = self.mapper.get_group_slice(name)
            decoded[name] = encoded[:, start:end].clone()
        return decoded

    def decode_pose(self, encoded_pose: torch.Tensor) -> dict[str, torch.Tensor]:
        if encoded_pose.ndim != 1:
            raise ValueError("encoded pose must be a single [151] feature row")
        return {key: value[0] for key, value in self.decode_sequence(encoded_pose[None]).items()}

    def _infer_frames(self, smpl: dict[str, Any]) -> int:
        for value in smpl.values():
            tensor = torch.as_tensor(value)
            if tensor.ndim > 1:
                return int(tensor.shape[0])
        return 1

    def _ensure_frame_dim(self, value: Any) -> torch.Tensor:
        tensor = torch.as_tensor(value, dtype=torch.float32)
        return tensor[None] if tensor.ndim == 1 else tensor

    def _coerce_body_pose(self, value: Any, frames: int) -> torch.Tensor:
        if value is None:
            return torch.zeros(frames, 126, dtype=torch.float32)
        tensor = self._ensure_frame_dim(value).to(dtype=torch.float32)
        if tensor.ndim == 2 and tensor.shape == (21, 6):
            tensor = tensor.reshape(1, 126)
        elif tensor.ndim == 2 and tensor.shape == (21, 3):
            padded = torch.zeros(1, 21, 6, dtype=torch.float32)
            padded[0, :, :3] = tensor
            tensor = padded.reshape(1, 126)
        if tensor.ndim == 3 and tensor.shape[1:] == (21, 6):
            tensor = tensor.reshape(tensor.shape[0], 126)
        elif tensor.ndim == 3 and tensor.shape[1:] == (21, 3):
            padded = torch.zeros(tensor.shape[0], 21, 6, dtype=torch.float32)
            padded[:, :, :3] = tensor
            tensor = padded.reshape(tensor.shape[0], 126)
        if tensor.shape[1] == 63:
            padded = torch.zeros(tensor.shape[0], 126, dtype=torch.float32)
            padded[:, :63] = tensor
            tensor = padded
        return self._fit_frames_and_dim(tensor, frames, 126)

    def _coerce_feature(self, value: Any, frames: int, dim: int) -> torch.Tensor:
        if value is None:
            return torch.zeros(frames, dim, dtype=torch.float32)
        tensor = self._ensure_frame_dim(value).to(dtype=torch.float32)
        if tensor.ndim > 2:
            tensor = tensor.reshape(tensor.shape[0], -1)
        return self._fit_frames_and_dim(tensor, frames, dim)

    def _fit_frames_and_dim(self, tensor: torch.Tensor, frames: int, dim: int) -> torch.Tensor:
        if tensor.shape[0] == 1 and frames > 1:
            tensor = tensor.repeat(frames, 1)
        if tensor.shape[0] != frames:
            raise ValueError(f"expected {frames} frames, got {tensor.shape[0]}")
        if tensor.shape[1] == dim:
            return tensor
        fitted = torch.zeros(frames, dim, dtype=torch.float32)
        fitted[:, : min(dim, tensor.shape[1])] = tensor[:, : min(dim, tensor.shape[1])]
        return fitted
