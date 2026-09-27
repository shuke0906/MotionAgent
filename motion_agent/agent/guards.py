"""Code-level hard guards for Phase 2 Planner decisions."""

from __future__ import annotations

from motion_agent.agent.actions import GeneratePayload, PlannerAction, PlannerDecision
from motion_agent.agent.budgets import stop_failed_allowed
from motion_agent.common.errors import MotionAgentError
from motion_agent.state.schemas import MotionAgentState


class PlannerGuardError(MotionAgentError):
    """Raised when a Planner decision violates a state guard."""


def validate_planner_decision(state: MotionAgentState, decision: PlannerDecision) -> None:
    action = decision.action

    if action == PlannerAction.GENERATE:
        _validate_generate(state, decision)
    elif action == PlannerAction.ACCEPT:
        _validate_accept(state)
    elif action == PlannerAction.RETRIEVE_REFERENCE:
        _validate_retrieval(state, decision)
    elif action == PlannerAction.STOP_FAILED:
        _validate_stop_failed(state)


def _validate_generate(state: MotionAgentState, decision: PlannerDecision) -> None:
    payload = GeneratePayload.model_validate(decision.payload)
    if state.plan.status != "ready" or state.plan.motion_spec_id is None:
        raise PlannerGuardError("GENERATE requires a ready Motion Plan")
    if not state.conditions.text.ready or state.plan.gem_text_condition_id is None:
        raise PlannerGuardError("GENERATE requires a ready GEM text condition")
    if state.control.budgets.generations_left <= 0:
        raise PlannerGuardError("GENERATE requires generation_budget > 0")
    if payload.strategy == "guided":
        if not state.control.capabilities.guided_generation_available:
            raise PlannerGuardError("guided GENERATE is unavailable")
        if state.control.budgets.guided_generations_left <= 0:
            raise PlannerGuardError("guided GENERATE requires guided generation budget")
    if payload.scope == "segment" and not state.generation.champion_candidate_id:
        raise PlannerGuardError("segment GENERATE requires a previous champion candidate")


def _validate_accept(state: MotionAgentState) -> None:
    verification = state.evaluation.verification_summary
    if verification is None:
        raise PlannerGuardError("ACCEPT requires a fresh VerificationReport")
    if verification.status != "complete":
        raise PlannerGuardError("ACCEPT requires complete verification")
    if not verification.overall_pass:
        raise PlannerGuardError("ACCEPT requires overall_pass=True")
    if verification.critical_failures:
        raise PlannerGuardError("ACCEPT requires no critical failures")
    if state.generation.generation_round > 0 and state.generation.champion_candidate_id is None:
        raise PlannerGuardError("ACCEPT requires a verified champion candidate")


def _validate_retrieval(state: MotionAgentState, decision: PlannerDecision) -> None:
    if state.control.budgets.retrieval_calls_left <= 0:
        raise PlannerGuardError("RETRIEVE_REFERENCE requires retrieval budget")
    payload = decision.typed_payload
    target_segments = decision.target_segments or [0]
    for segment_id in target_segments:
        for retrieval in state.conditions.retrievals:
            if (
                retrieval.status == "active"
                and retrieval.segment_id == segment_id
                and retrieval.retrieval_type == payload.retrieval_type
                and retrieval.purpose == payload.purpose
            ):
                raise PlannerGuardError("duplicate no-progress retrieval is not allowed")


def _validate_stop_failed(state: MotionAgentState) -> None:
    if not stop_failed_allowed(state):
        raise PlannerGuardError("STOP_FAILED requires exhausted budget or unrecoverable diagnosis")

