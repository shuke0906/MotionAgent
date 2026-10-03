"""Assemble generator-facing conditions from compiled constraints/keyframes."""

from __future__ import annotations

from dataclasses import dataclass

from motion_agent.common.fingerprints import condition_fingerprint
from motion_agent.constraints.composer import compose_conditions
from motion_agent.constraints.schemas import ConstraintBundle, HardMotionCondition, RewardSpec, VerificationSpec
from motion_agent.keyframes.schemas import KeyframeSpec
from motion_agent.constraints.store import ConditionStore
from motion_agent.generation.schemas import GenerationConditionBundle


@dataclass(frozen=True)
class AssembledGenerationConditions:
    bundle: GenerationConditionBundle
    blocked: bool
    conflicts: list[dict]
    mask_density: float


def assemble_generation_conditions(
    *,
    text_condition_id: str,
    text_condition_payload: dict,
    total_frames: int,
    hard_condition_handles: list[str] | None = None,
    reward_specs: list[RewardSpec | dict] | None = None,
    verification_specs: list[VerificationSpec] | None = None,
    keyframe_specs: list[KeyframeSpec] | None = None,
    active_constraint_ids: list[str] | None = None,
    active_keyframe_ids: list[str] | None = None,
    active_reference_ids: list[str] | None = None,
    store: ConditionStore | None = None,
) -> AssembledGenerationConditions:
    """Construct generator inputs and retain future verification/guidance targets.

    Forwarding a hard mask does not guarantee that Frozen GEM satisfies it.
    """

    store = store or ConditionStore()
    handles = hard_condition_handles or []
    conditions: list[HardMotionCondition] = [store.load_hard_condition(handle) for handle in handles]
    composed = compose_conditions(
        conditions,
        total_frames=total_frames,
        store=store,
        active_constraint_ids=active_constraint_ids or [],
        active_keyframe_ids=active_keyframe_ids or [],
        reward_specs=[
            spec if isinstance(spec, RewardSpec) else RewardSpec.model_validate(spec)
            for spec in (reward_specs or [])
        ],
    )
    payload = {
        "text_condition_id": text_condition_id,
        "text_condition": text_condition_payload,
        "hard_condition_handles": handles,
        "composed_hard_condition_handle": composed.hard_condition_handle,
        "reward_specs": [
            spec.model_dump(mode="json") if hasattr(spec, "model_dump") else spec
            for spec in (reward_specs or [])
        ],
        "active_constraint_ids": active_constraint_ids or [],
        "active_keyframe_ids": active_keyframe_ids or [],
        "active_reference_ids": active_reference_ids or [],
        "verification_specs": [spec.model_dump(mode="json") for spec in (verification_specs or [])],
        "keyframe_specs": [spec.model_dump(mode="json") for spec in (keyframe_specs or [])],
    }
    bundle = GenerationConditionBundle(
        text_condition_id=text_condition_id,
        hard_motion_condition_handle=composed.hard_condition_handle,
        reward_specs=[
            spec.model_dump(mode="json") if hasattr(spec, "model_dump") else spec
            for spec in (reward_specs or [])
        ],
        active_constraint_ids=active_constraint_ids or [],
        active_keyframe_ids=active_keyframe_ids or [],
        active_reference_ids=active_reference_ids or [],
        verification_specs=verification_specs or [],
        keyframe_specs=keyframe_specs or [],
        condition_fingerprint=condition_fingerprint(payload),
    )
    return AssembledGenerationConditions(
        bundle=bundle,
        blocked=composed.generation_blocked,
        conflicts=[conflict.model_dump(mode="json") for conflict in composed.conflicts],
        mask_density=composed.mask_density,
    )


def bundle_from_constraint_bundle(
    *,
    text_condition_id: str,
    text_condition_payload: dict,
    constraint_bundle: ConstraintBundle,
    active_reference_ids: list[str] | None = None,
) -> GenerationConditionBundle:
    payload = {
        "text_condition_id": text_condition_id,
        "text_condition": text_condition_payload,
        "hard_condition_handle": constraint_bundle.hard_condition_handle,
        "reward_specs": [spec.model_dump(mode="json") for spec in constraint_bundle.reward_specs],
        "active_constraint_ids": constraint_bundle.active_constraint_ids,
        "active_keyframe_ids": constraint_bundle.active_keyframe_ids,
        "active_reference_ids": active_reference_ids or [],
        "mask_density": constraint_bundle.mask_density,
        "verification_specs": [spec.model_dump(mode="json") for spec in constraint_bundle.verification_specs],
    }
    return GenerationConditionBundle(
        text_condition_id=text_condition_id,
        hard_motion_condition_handle=constraint_bundle.hard_condition_handle,
        reward_specs=[spec.model_dump(mode="json") for spec in constraint_bundle.reward_specs],
        active_constraint_ids=constraint_bundle.active_constraint_ids,
        active_keyframe_ids=constraint_bundle.active_keyframe_ids,
        active_reference_ids=active_reference_ids or [],
        verification_specs=constraint_bundle.verification_specs,
        condition_fingerprint=condition_fingerprint(payload),
    )
