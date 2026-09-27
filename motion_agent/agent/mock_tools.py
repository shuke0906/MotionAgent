"""Deterministic Phase 2 request builders and mock executors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from motion_agent.agent.actions import (
    BuildConstraintPayload,
    BuildKeyframePayload,
    CompileMotionPayload,
    GeneratePayload,
    PlannerAction,
    PlannerDecision,
    RetrieveReferencePayload,
)
from motion_agent.compiler import CompilerRequest, MotionSpecification, compile_motion
from motion_agent.state.artifacts import ArtifactStore
from motion_agent.state.reducer import StateReducer
from motion_agent.state.schemas import (
    ConstraintSummary,
    DiagnosisSummary,
    HeadingContinuitySummary,
    KeyframeSummary,
    MotionAgentState,
    MotionSegmentSummary,
    RepairProposalSummary,
    RetrievalSummary,
    RoutingHintSummary,
    VerificationSummary,
)
from motion_agent.state.versioning import StateStore


@dataclass(frozen=True)
class ToolRuntime:
    state_store: StateStore
    artifact_store: ArtifactStore
    reducer: StateReducer


def _target_segment(decision: PlannerDecision) -> int:
    return (decision.target_segments or [0])[0]


def build_compile_motion_request(decision: PlannerDecision, state: MotionAgentState) -> dict[str, Any]:
    payload = CompileMotionPayload.model_validate(decision.payload)
    return {
        "mode": payload.mode,
        "focus": payload.focus,
        "original_request": state.task.original_request,
        "target_segments": decision.target_segments,
        "duration_s": state.task.target_duration_s,
        "fps": state.task.fps,
        "total_frames": state.task.total_frames,
        "existing_motion_spec_id": state.plan.motion_spec_id,
    }


def execute_compile_motion(runtime: ToolRuntime, state: MotionAgentState, decision: PlannerDecision) -> MotionAgentState:
    request = build_compile_motion_request(decision, state)
    existing_motion_spec = None
    if request["mode"] == "revise" and request["existing_motion_spec_id"]:
        payload = runtime.artifact_store.get(request["existing_motion_spec_id"])
        spec_payload = payload.get("motion_spec", payload)
        existing_motion_spec = MotionSpecification.model_validate(spec_payload)

    result = compile_motion(
        CompilerRequest(
            original_request=request["original_request"],
            mode=request["mode"],
            target_segments=request["target_segments"],
            focus=request["focus"],
            duration_s=request["duration_s"],
            fps=request["fps"],
            total_frames=request["total_frames"],
            existing_motion_spec=existing_motion_spec,
        )
    )
    spec = runtime.artifact_store.put(
        "motion_spec",
        {
            "schema": "MotionSpecification",
            "request": request,
            "motion_spec": result.motion_spec.model_dump(mode="json"),
        },
    )
    text = runtime.artifact_store.put(
        "gem_text_condition",
        {
            "schema": "GEMTextCondition",
            "gem_text_condition": result.gem_text_condition.model_dump(mode="json"),
        },
    )
    segments = [
        MotionSegmentSummary(
            segment_id=segment.segment_id,
            start_s=segment.start_s,
            end_s=segment.end_s,
            action=segment.action,
            body_parts=segment.body_parts,
            style=segment.style,
            repetition=segment.repetition,
            heading_continuity=HeadingContinuitySummary.model_validate(
                segment.heading_continuity.model_dump(mode="python")
            ),
        )
        for segment in result.motion_spec.segments
    ]
    hints = [
        RoutingHintSummary.model_validate(hint.model_dump(mode="python"))
        for hint in result.routing_hints
    ]
    new_state = runtime.reducer.commit_motion_plan(
        state,
        motion_spec_id=spec.artifact_id,
        gem_text_condition_id=text.artifact_id,
        segments=segments,
        routing_hints=hints,
    )
    return runtime.state_store.commit(
        new_state,
        expected_version=state.state_version,
        event_type="compile_motion_committed",
        output_artifact_ids=[spec.artifact_id, text.artifact_id],
    )


def _compile_routing_hints(request: str) -> list[RoutingHintSummary]:
    lowered = request.lower()
    hints: list[RoutingHintSummary] = []
    if any(word in lowered for word in ["limp", "rare", "unusual"]):
        hints.append(
            RoutingHintSummary(
                intent_type="retrieval_candidate",
                segment_id=0,
                reason_code="RARE_STYLE",
                details={"query_hint": request, "preferred_type": "motion"},
            )
        )
    if any(word in lowered for word in ["touch", "contact", "trajectory", "table"]):
        hints.append(
            RoutingHintSummary(
                intent_type="constraint_candidate",
                segment_id=0,
                reason_code="EXPLICIT_CONTACT",
                details={"constraint_type": "contact"},
            )
        )
    if any(word in lowered for word in ["final", "stable", "seated", "pose"]):
        hints.append(
            RoutingHintSummary(
                intent_type="keyframe_candidate",
                segment_id=0,
                reason_code="WHOLE_BODY_END_STATE",
                details={"description": "stable final pose"},
            )
        )
    return hints


def build_retrieval_request(decision: PlannerDecision, state: MotionAgentState) -> dict[str, Any]:
    payload = RetrieveReferencePayload.model_validate(decision.payload)
    return {"target_segment": _target_segment(decision), **payload.model_dump(mode="json")}


def execute_retrieval(runtime: ToolRuntime, state: MotionAgentState, decision: PlannerDecision) -> MotionAgentState:
    request = build_retrieval_request(decision, state)
    reference = runtime.artifact_store.put(
        "reference",
        {
            "schema": "MockRetrievalResult",
            "status": "success",
            "request": request,
            "references": [
                {
                    "caption": request["query"],
                    "motion_handle": f"mock_motion:{request['query']}",
                    "score": 0.91,
                }
            ],
        },
    )
    summary = RetrievalSummary(
        ref_id=reference.artifact_id,
        segment_id=request["target_segment"],
        retrieval_type=request["retrieval_type"],
        purpose=request["purpose"],
        score=0.91,
        status="active",
    )
    new_state = runtime.reducer.add_retrieval(state, summary)
    return runtime.state_store.commit(
        new_state,
        expected_version=state.state_version,
        event_type="retrieval_committed",
        output_artifact_ids=[reference.artifact_id],
    )


def build_constraint_request(decision: PlannerDecision, state: MotionAgentState) -> dict[str, Any]:
    payload = BuildConstraintPayload.model_validate(decision.payload)
    return {"target_segment": _target_segment(decision), **payload.model_dump(mode="json")}


def execute_constraint(runtime: ToolRuntime, state: MotionAgentState, decision: PlannerDecision) -> MotionAgentState:
    request = build_constraint_request(decision, state)
    constraint = runtime.artifact_store.put(
        "constraint",
        {"schema": "MockCompiledConstraint", "status": "compiled_soft", "request": request},
    )
    verification_spec = runtime.artifact_store.put(
        "verification_spec",
        {"schema": "MockVerificationSpec", "constraint_id": constraint.artifact_id, "threshold": 0.05},
    )
    summary = ConstraintSummary(
        constraint_id=constraint.artifact_id,
        segment_id=request["target_segment"],
        constraint_type=request["constraint_type"],
        requested_strength=request["strength"],
        effective_mode="selection_reward",
        verification_spec_id=verification_spec.artifact_id,
    )
    new_state = runtime.reducer.add_or_replace_constraint(state, summary)
    return runtime.state_store.commit(
        new_state,
        expected_version=state.state_version,
        event_type="constraint_committed",
        output_artifact_ids=[constraint.artifact_id, verification_spec.artifact_id],
    )


def build_keyframe_request(decision: PlannerDecision, state: MotionAgentState) -> dict[str, Any]:
    payload = BuildKeyframePayload.model_validate(decision.payload)
    return {"target_segment": _target_segment(decision), **payload.model_dump(mode="json")}


def execute_keyframe(runtime: ToolRuntime, state: MotionAgentState, decision: PlannerDecision) -> MotionAgentState:
    request = build_keyframe_request(decision, state)
    keyframe = runtime.artifact_store.put(
        "keyframe",
        {"schema": "MockKeyframeSpec", "status": "compiled", "request": request},
    )
    verification_spec = runtime.artifact_store.put(
        "keyframe_verification_spec",
        {"schema": "MockKeyframeVerificationSpec", "keyframe_id": keyframe.artifact_id},
    )
    summary = KeyframeSummary(
        keyframe_id=keyframe.artifact_id,
        segment_id=request["target_segment"],
        target_time_s=request["time"],
        source_type=request["source_preference"],
        control_root_translation=False,
        verification_spec_id=verification_spec.artifact_id,
    )
    new_state = runtime.reducer.add_or_replace_keyframe(state, summary)
    return runtime.state_store.commit(
        new_state,
        expected_version=state.state_version,
        event_type="keyframe_committed",
        output_artifact_ids=[keyframe.artifact_id, verification_spec.artifact_id],
    )


def build_generation_request(decision: PlannerDecision, state: MotionAgentState) -> dict[str, Any]:
    payload = GeneratePayload.model_validate(decision.payload)
    return {
        **payload.model_dump(mode="json"),
        "motion_spec_id": state.plan.motion_spec_id,
        "gem_text_condition_id": state.plan.gem_text_condition_id,
        "condition_fingerprint": state.conditions.condition_fingerprint,
        "target_segments": decision.target_segments,
    }


def execute_generation(runtime: ToolRuntime, state: MotionAgentState, decision: PlannerDecision) -> MotionAgentState:
    request = build_generation_request(decision, state)
    mock_outcome = "fail" if "mock_fail" in request["reward_targets"] else "pass"
    generation = runtime.artifact_store.put(
        "generation",
        {"schema": "MockGenerationResult", "status": "success", "request": request, "mock_outcome": mock_outcome},
    )
    candidate_ids = []
    for index in range(request["num_candidates"]):
        candidate = runtime.artifact_store.put(
            "candidate",
            {
                "schema": "MockMotionCandidate",
                "generation_id": generation.artifact_id,
                "seed": index + 1,
                "technical_valid": True,
            },
        )
        candidate_ids.append(candidate.artifact_id)
    new_state = runtime.reducer.commit_generation_result(
        state,
        generation_id=generation.artifact_id,
        candidate_ids=candidate_ids,
        strategy=request["strategy"],
        scope=request["scope"],
        target_segments=request["target_segments"],
    )
    return runtime.state_store.commit(
        new_state,
        expected_version=state.state_version,
        event_type="generation_committed",
        output_artifact_ids=[generation.artifact_id, *candidate_ids],
    )


def execute_tournament(runtime: ToolRuntime, state: MotionAgentState) -> MotionAgentState:
    champion_id = state.generation.candidate_ids[0]
    tournament = runtime.artifact_store.put(
        "tournament",
        {
            "schema": "MockTournamentResult",
            "status": "success",
            "generation_id": state.generation.latest_generation_id,
            "champion_candidate_id": champion_id,
        },
    )
    new_state = runtime.reducer.set_champion(state, champion_id, tournament_id=tournament.artifact_id)
    return runtime.state_store.commit(
        new_state,
        expected_version=state.state_version,
        event_type="tournament_committed",
        output_artifact_ids=[tournament.artifact_id],
    )


def execute_verifier(runtime: ToolRuntime, state: MotionAgentState) -> MotionAgentState:
    generation_payload = runtime.artifact_store.get(state.generation.latest_generation_id)
    passed = generation_payload.get("mock_outcome") != "fail"
    diagnostic_codes = [] if passed else ["PERSISTENT_FAILURE"]
    verification = runtime.artifact_store.put(
        "verification",
        {"schema": "MockVerificationReport", "overall_pass": passed, "diagnostic_codes": diagnostic_codes},
    )
    summary = VerificationSummary(
        status="complete",
        overall_pass=passed,
        passed_check_ids=["technical", "semantic", "naturalness"] if passed else ["technical"],
        failed_check_ids=[] if passed else ["semantic"],
        critical_failures=[] if passed else ["PERSISTENT_FAILURE"],
        affected_segments=[] if passed else [0],
        diagnostic_codes=diagnostic_codes,
    )
    new_state = runtime.reducer.commit_verification_report(
        state,
        verification_id=verification.artifact_id,
        summary=summary,
    )
    return runtime.state_store.commit(
        new_state,
        expected_version=state.state_version,
        event_type="verification_committed",
        output_artifact_ids=[verification.artifact_id],
    )


def execute_diagnosis(runtime: ToolRuntime, state: MotionAgentState) -> MotionAgentState:
    verification = state.evaluation.verification_summary
    if verification is None:
        raise RuntimeError("diagnosis requires verification")
    if verification.overall_pass:
        payload = {"schema": "MockDiagnosisResult", "status": "no_failure", "repair_proposals": []}
        summary = DiagnosisSummary(status="no_failure", terminal_hint=None)
    else:
        terminal_hint = "BUDGET_EXHAUSTED" if state.control.budgets.generations_left <= 0 else None
        proposal = RepairProposalSummary(
            proposal_id="repair_mock_regenerate",
            action="GENERATE",
            reason_code="PERSISTENT_FAILURE",
            target_segments=None,
            confidence=0.8,
        )
        payload = {
            "schema": "MockDiagnosisResult",
            "status": "diagnosed" if terminal_hint is None else "no_valid_repair",
            "root_causes": ["SAMPLING_VARIANCE"],
            "repair_proposals": [proposal.model_dump(mode="json")] if terminal_hint is None else [],
            "terminal_hint": terminal_hint,
        }
        summary = DiagnosisSummary(
            status=payload["status"],
            primary_failures=verification.critical_failures,
            root_causes=["SAMPLING_VARIANCE"],
            target_segments=verification.affected_segments,
            proposal_summaries=[] if terminal_hint else [proposal],
            terminal_hint=terminal_hint,
        )
    diagnosis = runtime.artifact_store.put("diagnosis", payload)
    new_state = runtime.reducer.commit_diagnosis_result(
        state,
        diagnosis_id=diagnosis.artifact_id,
        summary=summary,
    )
    return runtime.state_store.commit(
        new_state,
        expected_version=state.state_version,
        event_type="diagnosis_committed",
        output_artifact_ids=[diagnosis.artifact_id],
    )


RequestBuilder = Callable[[PlannerDecision, MotionAgentState], dict[str, Any]]
ActionExecutor = Callable[[ToolRuntime, MotionAgentState, PlannerDecision], MotionAgentState]

REQUEST_BUILDERS: dict[PlannerAction, RequestBuilder] = {
    PlannerAction.COMPILE_MOTION: build_compile_motion_request,
    PlannerAction.RETRIEVE_REFERENCE: build_retrieval_request,
    PlannerAction.BUILD_CONSTRAINT: build_constraint_request,
    PlannerAction.BUILD_KEYFRAME: build_keyframe_request,
    PlannerAction.GENERATE: build_generation_request,
}

ACTION_EXECUTORS: dict[PlannerAction, ActionExecutor] = {
    PlannerAction.COMPILE_MOTION: execute_compile_motion,
    PlannerAction.RETRIEVE_REFERENCE: execute_retrieval,
    PlannerAction.BUILD_CONSTRAINT: execute_constraint,
    PlannerAction.BUILD_KEYFRAME: execute_keyframe,
    PlannerAction.GENERATE: execute_generation,
}
