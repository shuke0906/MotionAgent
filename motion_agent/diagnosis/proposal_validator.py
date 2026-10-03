"""Validation for planner-compatible repair proposals."""

from __future__ import annotations

from motion_agent.agent.actions import PlannerAction, PlannerDecision, validate_action_payload
from motion_agent.diagnosis.schemas import ProposalValidationResult, RepairProposal
from motion_agent.state.schemas import MotionAgentState


def validate_repair_proposal(
    proposal: RepairProposal,
    *,
    state: MotionAgentState | None = None,
    blocked_families: set[tuple[str, str]] | None = None,
    seen_fingerprints: set[str] | None = None,
) -> ProposalValidationResult:
    reasons: list[str] = []
    blocked_families = blocked_families or set()
    seen_fingerprints = seen_fingerprints or set()

    try:
        PlannerAction(proposal.planner_action)
        validate_action_payload(proposal.planner_action, proposal.parameters)
        PlannerDecision(
            action=proposal.planner_action,
            reason_code=proposal.reason_code,
            reason_summary=proposal.expected_effect,
            target_segments=proposal.target_segments,
            payload=proposal.parameters,
        )
    except Exception as exc:
        reasons.append(f"PLANNER_CONTRACT_INVALID:{type(exc).__name__}")

    if (proposal.failure_signature, proposal.repair_family) in blocked_families:
        reasons.append("REPAIR_FAMILY_BLOCKED")
    if proposal.proposal_fingerprint in seen_fingerprints:
        reasons.append("DUPLICATE_PROPOSAL")
    if state is not None:
        reasons.extend(_validate_state_compatibility(proposal, state))

    return ProposalValidationResult(
        proposal_id=proposal.proposal_id,
        status="rejected" if reasons else "accepted",
        reason_codes=reasons,
    )


def _validate_state_compatibility(proposal: RepairProposal, state: MotionAgentState) -> list[str]:
    reasons: list[str] = []
    valid_segments = {segment.segment_id for segment in state.plan.segment_summaries}
    if proposal.target_segments:
        missing = [segment for segment in proposal.target_segments if segment not in valid_segments]
        if missing and valid_segments:
            reasons.append("TARGET_SEGMENT_NOT_FOUND")
    if proposal.planner_action == PlannerAction.GENERATE and state.control.budgets.generations_left <= 0:
        reasons.append("GENERATION_BUDGET_EXHAUSTED")
    if proposal.planner_action == PlannerAction.RETRIEVE_REFERENCE and state.control.budgets.retrieval_calls_left <= 0:
        reasons.append("RETRIEVAL_BUDGET_EXHAUSTED")
    if proposal.planner_action == PlannerAction.BUILD_KEYFRAME and not state.control.capabilities.keyframe_available:
        reasons.append("KEYFRAME_CAPABILITY_UNAVAILABLE")
    if proposal.planner_action == PlannerAction.RETRIEVE_REFERENCE and not state.control.capabilities.retrieval_available:
        reasons.append("RETRIEVAL_CAPABILITY_UNAVAILABLE")
    if proposal.planner_action == PlannerAction.GENERATE and proposal.parameters.get("strategy") == "guided":
        if not state.control.capabilities.guided_generation_available:
            reasons.append("GUIDED_GENERATION_UNAVAILABLE")
    return reasons
