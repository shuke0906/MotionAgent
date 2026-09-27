# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-OneWay-Noncommercial
"""MotionAgent-facing adapters for GEM inference.

This module intentionally sits outside the demo script. MotionAgent needs to
build GEM-compatible inference payloads without pretending that a text-only
request came from a video file. The tensors produced here match the public
GEM-SMPL demo's data contract closely enough to be passed directly to
``GEM.predict``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class MotionAgentTextSegment:
    """A normalized text segment for GEM multi-text conditioning."""

    caption: str
    length: int
    name: str | None = None


@dataclass(frozen=True)
class MotionAgentCameraContext:
    """Synthetic camera context for pure-text GEM generation."""

    width: int = 1280
    height: int = 720
    static_camera: bool = True


def estimate_intrinsics(width: int, height: int) -> torch.Tensor:
    """Return a stable pinhole intrinsic matrix matching GEM demo conventions."""

    focal = float(max(width, height))
    return torch.tensor(
        [
            [focal, 0.0, width / 2.0],
            [0.0, focal, height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )


def _identity_camera(length: int, camera: MotionAgentCameraContext) -> dict[str, torch.Tensor]:
    if length <= 0:
        raise ValueError("segment length must be positive")

    r_w2c = torch.eye(3, dtype=torch.float32).unsqueeze(0).expand(length, -1, -1).clone()
    # For a static identity camera the angular and translational velocities are zero.
    cam_angvel = torch.zeros(length, 6, dtype=torch.float32)
    cam_tvel = torch.zeros(length, 3, dtype=torch.float32)
    k_fullimg = estimate_intrinsics(camera.width, camera.height)
    k_fullimg = k_fullimg.unsqueeze(0).expand(length, -1, -1).clone()
    return {
        "R_w2c": r_w2c,
        "cam_angvel": cam_angvel,
        "cam_tvel": cam_tvel,
        "K_fullimg": k_fullimg,
    }


def create_pure_text_segment(
    segment: MotionAgentTextSegment,
    camera: MotionAgentCameraContext | None = None,
) -> dict[str, Any]:
    """Create one zero-visual-feature GEM text segment."""

    camera = camera or MotionAgentCameraContext()
    length = int(segment.length)
    tensors = _identity_camera(length, camera)
    tensors.update(
        {
            "type": "text",
            "caption": segment.caption,
            "length": length,
            "name": segment.name,
            "bbx_xys": torch.zeros(length, 3, dtype=torch.float32),
            "kp2d": torch.zeros(length, 17, 3, dtype=torch.float32),
            "f_imgseq": torch.zeros(length, 1024, dtype=torch.float32),
            "has_img_mask": torch.zeros(length, dtype=torch.bool),
        }
    )
    return tensors


def build_motionagent_text_data(
    segments: list[MotionAgentTextSegment],
    camera: MotionAgentCameraContext | None = None,
    *,
    seed: int | None = None,
    observed_motion_3d: torch.Tensor | None = None,
    motion_mask_3d: torch.Tensor | None = None,
    rm_text_flag: torch.Tensor | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Build a GEM.predict payload for pure-text MotionAgent generation.

    ``observed_motion_3d`` and ``motion_mask_3d`` are optional hard-conditioning
    tensors in GEM's normalized 151-D motion representation. They are not
    interpreted here; this adapter only validates shape and forwards them.
    """

    if not segments:
        raise ValueError("at least one text segment is required")

    camera = camera or MotionAgentCameraContext()
    processed = [create_pure_text_segment(seg, camera) for seg in segments]

    segment_info: list[dict[str, Any]] = []
    current_idx = 0
    for i, seg in enumerate(processed):
        length = int(seg["length"])
        segment_info.append(
            {
                "start": current_idx,
                "end": current_idx + length,
                "type": "text",
                "caption": seg["caption"],
                "name": seg.get("name") or f"text_{i}",
            }
        )
        current_idx += length

    total_length = current_idx
    multi_text_data = {
        "vid": [],
        "caption": [],
        "text_ind": [],
        "window_start": [],
        "window_end": [],
    }
    for i, (seg, info) in enumerate(zip(processed, segment_info)):
        multi_text_data["vid"].append(info["name"])
        multi_text_data["caption"].append(seg["caption"])
        multi_text_data["text_ind"].append(i)
        multi_text_data["window_start"].append(info["start"] / total_length)
        multi_text_data["window_end"].append(info["end"] / total_length)

    multi_text_data["window_start"] = torch.tensor(
        multi_text_data["window_start"], dtype=torch.float32
    )
    multi_text_data["window_end"] = torch.tensor(multi_text_data["window_end"], dtype=torch.float32)

    data: dict[str, Any] = {
        "kp2d": torch.cat([s["kp2d"] for s in processed], dim=0),
        "bbx_xys": torch.cat([s["bbx_xys"] for s in processed], dim=0),
        "K_fullimg": torch.cat([s["K_fullimg"] for s in processed], dim=0),
        "cam_angvel": torch.cat([s["cam_angvel"] for s in processed], dim=0),
        "cam_tvel": torch.cat([s["cam_tvel"] for s in processed], dim=0),
        "R_w2c": torch.cat([s["R_w2c"] for s in processed], dim=0),
        "f_imgseq": torch.cat([s["f_imgseq"] for s in processed], dim=0),
        "has_text": torch.tensor([True], dtype=torch.bool),
        "caption": processed[0]["caption"],
        "mask": {
            "has_img_mask": torch.zeros(total_length, dtype=torch.bool),
            "has_2d_mask": torch.zeros(total_length, dtype=torch.bool),
            "has_cam_mask": torch.zeros(total_length, dtype=torch.bool),
            "has_audio_mask": torch.zeros(total_length, dtype=torch.bool),
            "has_music_mask": torch.zeros(total_length, dtype=torch.bool),
        },
        "length": torch.tensor(total_length, dtype=torch.long),
        "meta": [
            {
                "mode": "default",
                "multi_text_data": multi_text_data,
                "segment_info": segment_info,
                "motionagent_adapter": "pure_text_v1",
            }
        ],
    }

    if seed is not None:
        data["seed"] = int(seed)

    if observed_motion_3d is not None or motion_mask_3d is not None:
        if observed_motion_3d is None or motion_mask_3d is None:
            raise ValueError("observed_motion_3d and motion_mask_3d must be provided together")
        if observed_motion_3d.shape != motion_mask_3d.shape:
            raise ValueError("observed_motion_3d and motion_mask_3d must have identical shapes")
        if observed_motion_3d.ndim != 2 or observed_motion_3d.shape[0] != total_length:
            raise ValueError("hard-condition tensors must have shape [total_frames, motion_dim]")
        data["observed_motion_3d"] = observed_motion_3d.float()
        data["motion_mask_3d"] = motion_mask_3d.float()

    if rm_text_flag is not None:
        data["rm_text_flag"] = rm_text_flag.bool()

    return data, segment_info
