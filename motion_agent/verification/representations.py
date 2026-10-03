"""Documented bridges from GEM's persisted SMPL-X parameters to learned inputs."""

from __future__ import annotations

import importlib
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from motion_agent.verification.backends import BackendUnavailable


@contextmanager
def official_repository(path: str, namespace: str):
    """Scope upstream's generic src/lib imports to avoid cross-project collisions."""
    root = str(Path(path).resolve())
    if not Path(root).is_dir():
        raise BackendUnavailable(f"official runtime directory missing: {root}")
    names = [name for name in sys.modules if name == namespace or name.startswith(namespace + ".")]
    previous = {name: sys.modules.pop(name) for name in names}
    sys.path.insert(0, root)
    try:
        yield
    finally:
        sys.path.remove(root)
        for name in list(sys.modules):
            if name == namespace or name.startswith(namespace + "."):
                sys.modules.pop(name)
        sys.modules.update(previous)


def smpl_to_motioncritic(params: dict, hand_policy: str = "require_full_smpl") -> tuple[torch.Tensor, dict]:
    pose = params["body_pose"].detach().cpu().float().reshape(len(params["body_pose"]), -1, 3)
    frames = len(pose)
    approximation = None
    if pose.shape[1] == 21 and hand_policy == "neutral_terminal_hands":
        pose = torch.cat((pose, torch.zeros(frames, 2, 3)), dim=1)
        approximation = "SMPL-X has no SMPL terminal hand joints; explicit neutral rotations for joints 22/23"
    if pose.shape[1] != 23:
        raise BackendUnavailable("MotionCritic needs 23 body rotations plus root; GEM SMPL-X has 21. "
                                 "Supply a full SMPL retargeting or explicitly select neutral_terminal_hands.")
    root = params["global_orient"].detach().cpu().float().reshape(frames, 1, 3)
    transl = params["transl"].detach().cpu().float().reshape(frames, 1, 3)
    motion = torch.cat((root, pose, transl), dim=1).unsqueeze(0)
    if not torch.isfinite(motion).all():
        raise ValueError("non-finite SMPL motion")
    return motion, {"shape": list(motion.shape), "units": "radians/meters", "joint_order": "SMPL",
                    "coordinate_system": "GEM SMPL world, Y-up", "hand_policy": hand_policy,
                    "approximation": approximation, "representation": "axis_angle_24_plus_xyz"}


def smplx_joints(params: dict, model_path: str | None) -> torch.Tensor:
    if not model_path or not Path(model_path).exists():
        raise BackendUnavailable("licensed SMPL-X model is missing; set SMPLX_MODEL_PATH for FK to 22 joints")
    import smplx
    frames = len(params["body_pose"])
    batch_size = min(32, frames)
    model = (smplx.SMPLX(model_path, gender="neutral", use_pca=False, batch_size=batch_size)
             if Path(model_path).is_file() else
             smplx.create(model_path, model_type="smplx", gender="neutral", use_pca=False, batch_size=batch_size))
    inputs = {key: params[key].detach().cpu().float().reshape(frames, -1)
              for key in ("body_pose", "global_orient", "transl", "betas")}
    if not all(torch.isfinite(value).all() for value in inputs.values()):
        raise ValueError("non-finite SMPL-X FK parameters")
    joints = []
    with torch.inference_mode():
        for start in range(0, frames, batch_size):
            chunk = {key: value[start:start + batch_size] for key, value in inputs.items()}
            length = len(chunk["body_pose"])
            # Explicit neutral face/hands avoid model defaults using the initial batch size.
            for key, width in (("left_hand_pose", 45), ("right_hand_pose", 45),
                               ("expression", 10), ("jaw_pose", 3), ("leye_pose", 3), ("reye_pose", 3)):
                chunk[key] = torch.zeros(length, width)
            if length < batch_size:
                chunk = {key: torch.cat((value, value[-1:].expand(batch_size - length, -1)))
                         for key, value in chunk.items()}
            joints.append(model(**chunk, return_verts=False).joints[:length, :22].detach().cpu())
    return torch.cat(joints)


def joints_to_tmr(joints: torch.Tensor, fps: int, repository: str) -> torch.Tensor:
    if joints.ndim != 3 or joints.shape[1:] != (22, 3) or len(joints) < 3:
        raise ValueError("TMR conversion requires [L,22,3] SMPL-order Y-up joints")
    if not torch.isfinite(joints).all() or fps <= 0:
        raise ValueError("invalid joint data or FPS")
    # HumanML3D features use 20 Hz. Preserve elapsed time with linear resampling.
    source_times = np.arange(len(joints)) / fps
    target_times = np.arange(0, source_times[-1] + 1e-8, 1 / 20)
    source = joints.detach().cpu().numpy()
    resampled = np.stack([np.interp(target_times, source_times, source[:, j, c])
                          for j in range(22) for c in range(3)], axis=-1).reshape(-1, 22, 3)
    with official_repository(repository, "src"):
        conversion = importlib.import_module("src.guofeats").joints_to_guofeats
        features = torch.as_tensor(conversion(resampled), dtype=torch.float32)
    if features.ndim != 2 or features.shape[1] != 263 or not torch.isfinite(features).all():
        raise ValueError("official Guo feature conversion did not produce finite [L-1,263]")
    return features


def resample_critic(motion: torch.Tensor, fps: int) -> torch.Tensor:
    """Match HumanAct12/MDM's 20 Hz without interpolating axis angles across wraps."""
    if fps == 20:
        return motion
    from scipy.spatial.transform import Rotation, Slerp
    frames = motion.shape[1]
    source_times = np.arange(frames) / fps
    times = np.arange(0, source_times[-1] + 1e-8, 1 / 20)
    values = motion[0].cpu().numpy()
    result = np.zeros((len(times), 25, 3), dtype=np.float32)
    for joint in range(24):
        result[:, joint] = Slerp(source_times, Rotation.from_rotvec(values[:, joint]))(times).as_rotvec()
    for component in range(3):
        result[:, 24, component] = np.interp(times, source_times, values[:, 24, component])
    return torch.from_numpy(result).unsqueeze(0)
