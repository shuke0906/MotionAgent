"""Deterministic state invariant checks."""

from __future__ import annotations

from typing import Any

from motion_agent.common.errors import MissingArtifactError, StateInvariantError
from motion_agent.state.schemas import MotionAgentState


def _large_container_count(value: Any) -> int:
    if hasattr(value, "shape"):
        shape = getattr(value, "shape", ())
        total = 1
        for dim in shape:
            total *= int(dim)
        return total
    if isinstance(value, dict):
        return sum(_large_container_count(v) for v in value.values())
    if isinstance(value, list):
        return len(value) + sum(_large_container_count(v) for v in value)
    return 0


def referenced_artifact_ids(state: MotionAgentState) -> set[str]:
    ids: set[str] = set()
    if state.plan.motion_spec_id:
        ids.add(state.plan.motion_spec_id)
    if state.plan.gem_text_condition_id:
        ids.add(state.plan.gem_text_condition_id)
    if state.conditions.text.gem_text_condition_id:
        ids.add(state.conditions.text.gem_text_condition_id)
    ids.update(r.ref_id for r in state.conditions.retrievals if r.status == "active")
    for constraint in state.conditions.constraints:
        if constraint.status == "active":
            ids.add(constraint.constraint_id)
            if constraint.verification_spec_id:
                ids.add(constraint.verification_spec_id)
    for keyframe in state.conditions.keyframes:
        if keyframe.status == "active":
            ids.add(keyframe.keyframe_id)
            if keyframe.verification_spec_id:
                ids.add(keyframe.verification_spec_id)
    if state.generation.latest_generation_id:
        ids.add(state.generation.latest_generation_id)
    ids.update(state.generation.candidate_ids)
    if state.generation.latest_tournament_id:
        ids.add(state.generation.latest_tournament_id)
    if state.evaluation.latest_verification_id:
        ids.add(state.evaluation.latest_verification_id)
    if state.evaluation.latest_diagnosis_id:
        ids.add(state.evaluation.latest_diagnosis_id)
    return ids


def validate_state_invariants(
    state: MotionAgentState,
    *,
    artifact_store: Any | None = None,
    previous_version: int | None = None,
) -> None:
    budgets = state.control.budgets
    if min(
        budgets.iterations_left,
        budgets.generations_left,
        budgets.retrieval_calls_left,
        budgets.guided_generations_left,
    ) < 0:
        raise StateInvariantError("generation_budget cannot become negative")

    if previous_version is not None and state.state_version <= previous_version:
        raise StateInvariantError("state_version must increase monotonically")

    if state.control.accepted:
        verification = state.evaluation.verification_summary
        if verification is None or verification.status != "complete" or not verification.overall_pass:
            raise StateInvariantError("accepted=true requires a successful complete VerificationReport")
        if verification.critical_failures:
            raise StateInvariantError("accepted=true requires no critical failures")

    champion = state.generation.champion_candidate_id
    if champion is not None and champion not in state.generation.candidate_ids:
        raise StateInvariantError("champion must reference an existing candidate")

    active_constraints = [c.constraint_id for c in state.conditions.constraints if c.status == "active"]
    if len(active_constraints) != len(set(active_constraints)):
        raise StateInvariantError("active constraint IDs must be unique")

    active_keyframes = [k.keyframe_id for k in state.conditions.keyframes if k.status == "active"]
    if len(active_keyframes) != len(set(active_keyframes)):
        raise StateInvariantError("active keyframe IDs must be unique")

    if state.evaluation.latest_verification_id and state.generation.champion_candidate_id is None:
        raise StateInvariantError("verification requires a champion candidate")

    raw_state = state.model_dump(mode="json")
    if _large_container_count(raw_state) > 5000:
        raise StateInvariantError("large tensors or arrays must not enter canonical Planner state")

    if artifact_store is not None:
        for artifact_id in referenced_artifact_ids(state):
            if not artifact_store.exists(artifact_id):
                raise MissingArtifactError(f"artifact handle does not exist: {artifact_id}")

