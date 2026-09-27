"""Technical tensor validation for generated candidates."""

from __future__ import annotations

from typing import Any

import torch


REQUIRED_SMPL_KEYS = ("body_pose", "global_orient", "transl", "betas")


def validate_candidate_output(candidate_tensors: dict[str, Any], *, expected_frames: int) -> None:
    motion = candidate_tensors.get("motion_repr")
    if motion is None:
        raise ValueError("candidate missing motion_repr")
    if motion.ndim != 2:
        raise ValueError("motion_repr must have shape [frames, channels]")
    if motion.shape[0] != expected_frames:
        raise ValueError(f"motion_repr frame count mismatch: {motion.shape[0]} != {expected_frames}")
    if motion.shape[-1] != 151:
        raise ValueError(f"motion_repr last dimension must be 151, got {motion.shape[-1]}")
    if not torch.isfinite(motion).all():
        raise ValueError("motion_repr contains NaN or Inf")

    smpl = candidate_tensors.get("body_params_global")
    if not smpl:
        raise ValueError("candidate missing global SMPL")
    for key in REQUIRED_SMPL_KEYS:
        if key not in smpl:
            raise ValueError(f"global SMPL missing {key}")
        value = smpl[key]
        if value.shape[0] != expected_frames:
            raise ValueError(f"global SMPL {key} frame count mismatch")
        if not torch.isfinite(value).all():
            raise ValueError(f"global SMPL {key} contains NaN or Inf")
