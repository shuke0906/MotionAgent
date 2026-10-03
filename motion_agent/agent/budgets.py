"""Budget and legal-action helpers for Planner visibility."""

from __future__ import annotations

from motion_agent.agent.actions import PlannerAction
from motion_agent.state.schemas import MotionAgentState


def budget_exhausted(state: MotionAgentState) -> bool:
    budgets = state.control.budgets
    return (
        budgets.iterations_left <= 0
        or budgets.generations_left <= 0
        or budgets.retrieval_calls_left <= 0
    )


def stop_failed_allowed(state: MotionAgentState) -> bool:
    diagnosis = state.evaluation.diagnosis_summary
    return budget_exhausted(state) or (diagnosis is not None and diagnosis.terminal_hint is not None)


def get_legal_actions(state: MotionAgentState) -> list[PlannerAction]:
    if state.evaluation.verification_summary is not None:
        verification = state.evaluation.verification_summary
        if verification.status == "complete" and verification.overall_pass and not verification.critical_failures:
            return [PlannerAction.ACCEPT]

    if state.control.budgets.iterations_left <= 0 or state.control.budgets.generations_left <= 0:
        return [PlannerAction.STOP_FAILED]
    if state.evaluation.diagnosis_summary and state.evaluation.diagnosis_summary.terminal_hint:
        return [PlannerAction.STOP_FAILED]

    actions: list[PlannerAction] = []
    if state.plan.status == "missing":
        actions.append(PlannerAction.COMPILE_MOTION)
    else:
        actions.extend(
            [
                PlannerAction.COMPILE_MOTION,
                PlannerAction.RETRIEVE_REFERENCE,
                PlannerAction.BUILD_CONSTRAINT,
                PlannerAction.BUILD_KEYFRAME,
            ]
        )
        if state.control.budgets.generations_left > 0:
            actions.append(PlannerAction.GENERATE)

    if stop_failed_allowed(state):
        actions.append(PlannerAction.STOP_FAILED)
    return actions
