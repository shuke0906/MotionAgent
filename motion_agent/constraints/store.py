"""Filesystem-backed condition payload store."""

from __future__ import annotations

from pathlib import Path

import torch

from motion_agent.common.fingerprints import condition_fingerprint
from motion_agent.common.ids import new_artifact_id
from motion_agent.constraints.schemas import HardMotionCondition


class ConditionStore:
    def __init__(self, root: str | Path = "artifacts/phase6_conditions") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def save_hard_condition(self, condition: HardMotionCondition, *, handle: str | None = None) -> str:
        fingerprint = condition_fingerprint(
            {
                "source": condition.source_constraint_id,
                "values": condition.values,
                "mask": condition.mask,
                "metadata": condition.metadata,
            }
        )
        handle = handle or new_artifact_id("condition", fingerprint)
        path = self.root / f"{handle}.pt"
        if not path.exists():
            torch.save(
                {
                    "values": condition.values.detach().cpu(),
                    "mask": condition.mask.detach().cpu(),
                    "source_constraint_id": condition.source_constraint_id,
                    "metadata": condition.metadata,
                },
                path,
            )
        return handle

    def load_hard_condition(self, handle: str) -> HardMotionCondition:
        payload = torch.load(self.root / f"{handle}.pt", map_location="cpu", weights_only=False)
        return HardMotionCondition(**payload)
