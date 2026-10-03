"""Merge hard constraints/keyframes and detect hard-condition conflicts."""

from __future__ import annotations

import torch

from motion_agent.constraints.schemas import (
    CompiledConstraint,
    ConstraintBundle,
    ConstraintConflict,
    HardMotionCondition,
    RewardSpec,
    VerificationSpec,
)
from motion_agent.constraints.store import ConditionStore


def detect_overlap_conflict(
    left: HardMotionCondition,
    right: HardMotionCondition,
    *,
    value_tolerance: float = 1e-5,
) -> list[ConstraintConflict]:
    overlap = (left.mask > 0) & (right.mask > 0)
    if not bool(overlap.any()):
        return []
    diff = torch.abs(left.values - right.values)
    conflicting = overlap & (diff > value_tolerance)
    if not bool(conflicting.any()):
        return []
    frames = torch.where(conflicting.any(dim=1))[0].tolist()
    features = torch.where(conflicting.any(dim=0))[0].tolist()
    return [
        ConstraintConflict(
            constraint_ids=[left.source_constraint_id, right.source_constraint_id],
            frame_indices=[int(x) for x in frames],
            feature_indices=[int(x) for x in features],
            max_value_difference=float(diff[conflicting].max().item()),
            conflict_type="hard_condition_overlap",
        )
    ]


def compose_conditions(
    conditions: list[HardMotionCondition],
    *,
    total_frames: int,
    store: ConditionStore | None = None,
    active_constraint_ids: list[str] | None = None,
    active_keyframe_ids: list[str] | None = None,
    reward_specs: list[RewardSpec] | None = None,
    verification_specs: list[VerificationSpec] | None = None,
) -> ConstraintBundle:
    store = store or ConditionStore()
    observed = torch.zeros(total_frames, 151, dtype=torch.float32)
    mask = torch.zeros(total_frames, 151, dtype=torch.float32)
    conflicts: list[ConstraintConflict] = []
    composed = HardMotionCondition(values=observed, mask=mask, source_constraint_id="composed")
    for condition in conditions:
        new_conflicts = detect_overlap_conflict(composed, condition)
        conflicts.extend(new_conflicts)
        if new_conflicts:
            continue
        active = condition.mask > 0
        observed[active] = condition.values[active]
        mask = torch.maximum(mask, condition.mask)
        composed = HardMotionCondition(values=observed, mask=mask, source_constraint_id="composed")
    handle = None
    if bool(mask.any()):
        handle = store.save_hard_condition(
            HardMotionCondition(values=observed, mask=mask, source_constraint_id="composed_conditions")
        )
    return ConstraintBundle(
        hard_condition_handle=handle,
        reward_specs=reward_specs or [],
        verification_specs=verification_specs or [],
        active_constraint_ids=active_constraint_ids or [],
        active_keyframe_ids=active_keyframe_ids or [],
        mask_density=float(mask.mean().item()),
        conflicts=conflicts,
        generation_blocked=bool(conflicts),
    )


class ConstraintComposer:
    def __init__(self, store: ConditionStore | None = None) -> None:
        self.store = store or ConditionStore()

    def compose_constraints(self, constraints: list[CompiledConstraint], total_frames: int) -> ConstraintBundle:
        return compose_constraints(constraints, total_frames, store=self.store)


def compose_constraints(
    constraints: list[CompiledConstraint],
    total_frames: int,
    *,
    store: ConditionStore | None = None,
) -> ConstraintBundle:
    store = store or ConditionStore()
    conditions = [
        store.load_hard_condition(constraint.hard_condition_handle)
        for constraint in constraints
        if constraint.hard_condition_handle
    ]
    return compose_conditions(
        conditions,
        total_frames=total_frames,
        store=store,
        active_constraint_ids=[constraint.constraint_id for constraint in constraints],
        reward_specs=[constraint.reward_spec for constraint in constraints if constraint.reward_spec is not None],
        verification_specs=[constraint.verification_spec for constraint in constraints],
    )
