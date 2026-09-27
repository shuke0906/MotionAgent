"""Build Phase 4 generation requests from compiler/state artifacts."""

from __future__ import annotations

from motion_agent.common.fingerprints import condition_fingerprint
from motion_agent.common.ids import new_id
from motion_agent.compiler.schemas import CompilerResult, GEMTextCondition
from motion_agent.generation.schemas import GenerationConditionBundle, GenerationRequest
from motion_agent.generation.seed_manager import derive_seeds
from motion_agent.state.schemas import MotionAgentState


def bundle_for_text_condition(
    condition: GEMTextCondition,
    *,
    text_condition_id: str = "textcond_inline",
    active_constraint_ids: list[str] | None = None,
    active_keyframe_ids: list[str] | None = None,
    active_reference_ids: list[str] | None = None,
) -> GenerationConditionBundle:
    payload = {
        "text_condition": condition.model_dump(mode="json"),
        "active_constraint_ids": active_constraint_ids or [],
        "active_keyframe_ids": active_keyframe_ids or [],
        "active_reference_ids": active_reference_ids or [],
    }
    return GenerationConditionBundle(
        text_condition_id=text_condition_id,
        active_constraint_ids=active_constraint_ids or [],
        active_keyframe_ids=active_keyframe_ids or [],
        active_reference_ids=active_reference_ids or [],
        condition_fingerprint=condition_fingerprint(payload),
    )


def build_generation_request(
    *,
    condition: GEMTextCondition,
    fps: int,
    text_condition_id: str = "textcond_inline",
    generation_id: str | None = None,
    num_candidates: int = 1,
    seed: int | None = None,
) -> GenerationRequest:
    return GenerationRequest(
        generation_id=generation_id or new_id("generation"),
        strategy="normal",
        scope="full",
        fps=fps,
        total_frames=condition.total_frames,
        text_condition=condition,
        condition_bundle=bundle_for_text_condition(condition, text_condition_id=text_condition_id),
        num_candidates=num_candidates,
        seeds=derive_seeds(seed, num_candidates),
    )


def build_generation_request_from_compiler_result(
    result: CompilerResult,
    *,
    generation_id: str | None = None,
    num_candidates: int = 1,
    seed: int | None = None,
) -> GenerationRequest:
    return build_generation_request(
        condition=result.gem_text_condition,
        fps=result.motion_spec.fps,
        generation_id=generation_id,
        num_candidates=num_candidates,
        seed=seed,
    )


def build_generation_request_from_state(
    state: MotionAgentState,
    condition: GEMTextCondition,
    *,
    num_candidates: int = 1,
    seed: int | None = None,
) -> GenerationRequest:
    bundle = GenerationConditionBundle(
        text_condition_id=state.plan.gem_text_condition_id or "textcond_missing",
        active_constraint_ids=[c.constraint_id for c in state.conditions.constraints if c.status == "active"],
        active_keyframe_ids=[k.keyframe_id for k in state.conditions.keyframes if k.status == "active"],
        active_reference_ids=[r.ref_id for r in state.conditions.retrievals if r.status == "active"],
        condition_fingerprint=state.conditions.condition_fingerprint
        or condition_fingerprint(condition.model_dump(mode="json")),
    )
    return GenerationRequest(
        generation_id=new_id("generation"),
        strategy="normal",
        scope="full",
        fps=state.task.fps,
        total_frames=state.task.total_frames,
        text_condition=condition,
        condition_bundle=bundle,
        num_candidates=num_candidates,
        seeds=derive_seeds(seed, num_candidates),
    )
