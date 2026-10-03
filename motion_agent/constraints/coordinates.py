"""Coordinate and target resolution for deterministic constraints."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from motion_agent.constraints.schemas import ConstraintTarget


@dataclass(frozen=True)
class TargetResolution:
    status: str
    target: ConstraintTarget | None = None
    point: torch.Tensor | None = None
    warnings: list[str] = field(default_factory=list)


class CoordinateResolver:
    def resolve_target(self, target: ConstraintTarget | None) -> TargetResolution:
        if target is None:
            return TargetResolution(status="needs_reference", warnings=["target is missing"])
        if target.target_type == "scene_anchor" and not target.geometry_handle:
            return TargetResolution(status="needs_geometry", target=target, warnings=["scene anchor has no geometry handle"])
        if target.target_type == "point":
            return TargetResolution(status="resolved", target=target, point=self.resolve_point(target))
        return TargetResolution(status="resolved", target=target)

    def resolve_point(self, target: ConstraintTarget) -> torch.Tensor:
        values = target.values
        raw = values.get("xyz", values.get("point", values.get("position")))
        if raw is None:
            raise ValueError("point target requires values.xyz, values.point, or values.position")
        point = torch.as_tensor(raw, dtype=torch.float32)
        if tuple(point.shape) != (3,):
            raise ValueError("point target must be a 3-vector")
        return point

    def transform_world_to_root(self, point: torch.Tensor, root_translation: torch.Tensor | None = None) -> torch.Tensor:
        return point - root_translation if root_translation is not None else point

    def transform_root_to_world(self, point: torch.Tensor, root_translation: torch.Tensor | None = None) -> torch.Tensor:
        return point + root_translation if root_translation is not None else point
