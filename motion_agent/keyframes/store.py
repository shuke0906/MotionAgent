"""Filesystem-backed keyframe pose store."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from motion_agent.common.fingerprints import condition_fingerprint
from motion_agent.common.ids import new_artifact_id


class KeyframeStore:
    def __init__(self, root: str | Path = "artifacts/phase6_keyframes") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def save_pose(self, pose: dict[str, Any], *, source: str) -> str:
        fingerprint = condition_fingerprint({"pose": pose, "source": source})
        handle = new_artifact_id("pose", fingerprint)
        path = self.root / f"{handle}.pt"
        if not path.exists():
            torch.save(pose, path)
        return handle

    def load_pose(self, handle: str) -> dict[str, Any]:
        return torch.load(self.root / f"{handle}.pt", map_location="cpu", weights_only=False)
