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
from motion_agent.common.ids import new_id
from motion_agent.constraints import ConditionStore, ConstraintRequest, ConstraintTarget, TimeSpec, compile_constraint
from motion_agent.diagnosis import DiagnosisRequest, DiagnosisService
from motion_agent.keyframes import KeyframeRequest, build_keyframe
from motion_agent.keyframes.store import KeyframeStore
from motion_agent.retrieval import RetrievalRequest, build_default_retrieval_service
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
from motion_agent.verification.schemas import VerificationReport


@dataclass(frozen=True)
class ToolRuntime:
    state_store: StateStore
    artifact_store: ArtifactStore
    reducer: StateReducer
    compiler: Any = None
    action_executors: Any = None
    selection_executor: Any = None
    verifier_executor: Any = None
    diagnosis_executor: Any = None
    phase11: bool = False
    planner_commit: Any = None


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


def _retrieved_captions_by_segment(runtime: ToolRuntime, state: MotionAgentState) -> dict[int, list[str]]:
    captions_by_segment: dict[int, list[str]] = {}
    for summary in state.conditions.retrievals:
        if summary.status not in {"active", "low_confidence"}:
            continue
        if summary.purpose != "prompt_grounding":
            continue
        try:
            payload = runtime.artifact_store.get(summary.ref_id)
        except FileNotFoundError:
            continue
        references = payload.get("references", [])
        captions = [
            reference.get("caption")
            for reference in references
            if isinstance(reference, dict) and reference.get("caption")
        ]
        if captions:
            captions_by_segment.setdefault(summary.segment_id, []).extend(captions)
    return captions_by_segment


def execute_compile_motion(runtime: ToolRuntime, state: MotionAgentState, decision: PlannerDecision) -> MotionAgentState:
    request = build_compile_motion_request(decision, state)
    existing_motion_spec = None
    if request["mode"] == "revise" and request["existing_motion_spec_id"]:
        payload = runtime.artifact_store.get(request["existing_motion_spec_id"])
        spec_payload = payload.get("motion_spec", payload)
        existing_motion_spec = MotionSpecification.model_validate(spec_payload)

    compiler_call = runtime.compiler.compile if runtime.compiler is not None else compile_motion
    result = compiler_call(
        CompilerRequest(
            original_request=request["original_request"],
            mode=request["mode"],
            target_segments=request["target_segments"],
            focus=request["focus"],
            duration_s=request["duration_s"],
            fps=request["fps"],
            total_frames=request["total_frames"],
            existing_motion_spec=existing_motion_spec,
            retrieved_captions_by_segment=_retrieved_captions_by_segment(runtime, state),
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
            temporal_constraint=segment.temporal_constraint.model_dump(mode="json")
            if segment.temporal_constraint
            else None,
            temporal_mode=segment.temporal_mode,
            temporal_relation=segment.temporal_relation.model_dump(mode="json")
            if segment.temporal_relation
            else None,
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
    request_payload = build_retrieval_request(decision, state)
    result = build_default_retrieval_service().retrieve_reference(
        state,
        RetrievalRequest.model_validate(request_payload),
    )
    reference = runtime.artifact_store.put(
        "reference",
        {
            "schema": "RetrievalResult",
            "request": request_payload,
            **result.model_dump(mode="json"),
        },
    )
    best_score = result.references[0].retrieval_score if result.references else None
    summary_status = "active" if result.status in {"success", "duplicate"} else "low_confidence" if result.status == "low_confidence" else "invalid"
    summary = RetrievalSummary(
        ref_id=reference.artifact_id,
        segment_id=request_payload["target_segment"],
        retrieval_type=request_payload["retrieval_type"],
        purpose=request_payload["purpose"],
        score=best_score,
        query=request_payload["query"],
        retrieval_signature=result.retrieval_meta.get("retrieval_signature"),
        corpus_version=result.retrieval_meta.get("corpus_version"),
        status=summary_status,
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
    time_spec = _time_spec_from_payload(payload.time_range)
    target = _constraint_target_from_payload(payload.target)
    return {
        "mode": payload.mode,
        "target_segment": _target_segment(decision),
        "constraint_type": payload.constraint_type,
        "requested_strength": payload.strength,
        "time_spec": time_spec.model_dump(mode="json"),
        "body_parts": [payload.body_part] if payload.body_part else [],
        "target": target.model_dump(mode="json") if target else None,
        "metadata": {"planner_payload": payload.model_dump(mode="json")},
    }


def _time_spec_from_payload(time_range: list[float] | None) -> TimeSpec:
    if not time_range:
        return TimeSpec(type="segment")
    if len(time_range) == 1:
        return TimeSpec(type="point", time_s=time_range[0])
    return TimeSpec(type="window", start_s=time_range[0], end_s=time_range[1])


def _constraint_target_from_payload(target: str | dict[str, Any] | None) -> ConstraintTarget | None:
    if target is None:
        return None
    if isinstance(target, str):
        lowered = target.lower()
        if any(word in lowered for word in ["table", "chair", "wall", "scene"]):
            return ConstraintTarget(target_type="scene_anchor", coordinate_frame="scene_local", values={"label": target})
        return ConstraintTarget(target_type="self_anchor", coordinate_frame="world", values={"label": target})
    if "target_type" in target:
        return ConstraintTarget.model_validate(target)
    if any(key in target for key in ["xyz", "point", "position"]):
        return ConstraintTarget(target_type="point", coordinate_frame=target.get("coordinate_frame", "world"), values=target)
    if any(key in target for key in ["points", "waypoints", "trajectory"]):
        return ConstraintTarget(target_type="trajectory", coordinate_frame=target.get("coordinate_frame", "world"), values=target)
    return ConstraintTarget(target_type="reference_pose", coordinate_frame="world", values=target)


def execute_constraint(runtime: ToolRuntime, state: MotionAgentState, decision: PlannerDecision) -> MotionAgentState:
    request = build_constraint_request(decision, state)
    if request["mode"] == "remove":
        constraint_id = BuildConstraintPayload.model_validate(decision.payload).target
        if not isinstance(constraint_id, str):
            raise ValueError("remove constraint requires target to carry the constraint id")
        new_state = runtime.reducer.remove_constraint(state, constraint_id)
        return runtime.state_store.commit(new_state, expected_version=state.state_version, event_type="constraint_removed")
    condition_store = ConditionStore()
    result = compile_constraint(state, ConstraintRequest.model_validate(request), store=condition_store)
    constraint = runtime.artifact_store.put(
        "constraint",
        {
            "schema": "CompiledConstraint",
            "status": result.status,
            "request": request,
            "constraint": result.constraint.model_dump(mode="json") if result.constraint else None,
            "warnings": result.warnings,
            "routing_hints": result.routing_hints,
        },
    )
    if result.constraint is None:
        summary = ConstraintSummary(
            constraint_id=constraint.artifact_id,
            segment_id=request["target_segment"],
            constraint_type=request["constraint_type"],
            requested_strength=request["requested_strength"],
            effective_mode="selection_reward",
            status="conflict" if result.status == "conflict" else "removed",
        )
        new_state = runtime.reducer.add_or_replace_constraint(state, summary)
        return runtime.state_store.commit(
            new_state,
            expected_version=state.state_version,
            event_type="constraint_committed",
            output_artifact_ids=[constraint.artifact_id],
        )
    verification_spec = runtime.artifact_store.put(
        "verification_spec",
        {
            "schema": "VerificationSpec",
            **result.constraint.verification_spec.model_dump(mode="json"),
        },
    )
    summary = ConstraintSummary(
        constraint_id=constraint.artifact_id,
        segment_id=request["target_segment"],
        constraint_type=request["constraint_type"],
        requested_strength=request["requested_strength"],
        effective_mode=result.constraint.effective_mode,
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


def _active_retrieved_references(runtime: ToolRuntime, state: MotionAgentState, segment_id: int) -> list[dict[str, Any]]:
    references: list[dict[str, Any]] = []
    for summary in state.conditions.retrievals:
        if summary.status != "active" or summary.segment_id != segment_id or summary.purpose != "keyframe_source":
            continue
        try:
            payload = runtime.artifact_store.get(summary.ref_id)
        except FileNotFoundError:
            continue
        references.extend(payload.get("references", []))
    return references


def execute_keyframe(runtime: ToolRuntime, state: MotionAgentState, decision: PlannerDecision) -> MotionAgentState:
    request = build_keyframe_request(decision, state)
    if request["mode"] == "remove":
        keyframe_id = BuildKeyframePayload.model_validate(decision.payload).description
        new_state = runtime.reducer.remove_keyframe(state, keyframe_id)
        return runtime.state_store.commit(new_state, expected_version=state.state_version, event_type="keyframe_removed")
    request["retrieved_references"] = _active_retrieved_references(runtime, state, request["target_segment"])
    condition_store = ConditionStore()
    result = build_keyframe(
        state,
        KeyframeRequest.model_validate(request),
        condition_store=condition_store,
        keyframe_store=KeyframeStore(),
    )
    keyframe = runtime.artifact_store.put(
        "keyframe",
        {
            "schema": "KeyframeSpec",
            "status": result.status,
            "request": request,
            "keyframe": result.keyframe.model_dump(mode="json") if result.keyframe else None,
            "warnings": result.warnings,
            "routing_hints": result.routing_hints,
        },
    )
    if result.keyframe is None:
        summary = KeyframeSummary(
            keyframe_id=keyframe.artifact_id,
            segment_id=request["target_segment"],
            target_time_s=request["time"],
            source_type=request["source_preference"],
            control_root_translation=False,
            status="removed",
        )
        new_state = runtime.reducer.add_or_replace_keyframe(state, summary)
        return runtime.state_store.commit(
            new_state,
            expected_version=state.state_version,
            event_type="keyframe_committed",
            output_artifact_ids=[keyframe.artifact_id],
        )
    verification_spec = runtime.artifact_store.put(
        "keyframe_verification_spec",
        {
            "schema": "KeyframeVerificationSpec",
            **result.keyframe.verification_spec.model_dump(mode="json"),
        },
    )
    summary = KeyframeSummary(
        keyframe_id=keyframe.artifact_id,
        segment_id=request["target_segment"],
        target_time_s=request["time"],
        source_type=result.keyframe.source_type,
        control_root_translation=result.keyframe.control_root_translation,
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
    real_report = _load_phase10_verification_report(runtime, state)
    if real_report is not None:
        motion_spec = _load_current_motion_spec(runtime, state)
        generation_payload = (
            runtime.artifact_store.get(state.generation.latest_generation_id)
            if state.generation.latest_generation_id
            else {}
        )
        request = DiagnosisRequest(
            diagnosis_id=new_id("diag"),
            verification_report=real_report,
            motion_spec=motion_spec,
            generation_request=generation_payload.get("request") if isinstance(generation_payload, dict) else None,
            repair_history=[],
            budget=state.control.budgets,
            capability_summary=state.control.capabilities,
        )
        result = DiagnosisService().diagnose(request, state=state)
        diagnosis = runtime.artifact_store.put(
            "diagnosis",
            {
                "schema": "DiagnosisResult",
                "diagnosis_result": result.model_dump(mode="json"),
            },
        )
        summary_dict = result.summary()
        summary = DiagnosisSummary(
            status=summary_dict["status"],
            primary_failures=summary_dict["primary_failures"],
            root_causes=summary_dict["root_causes"],
            target_segments=summary_dict["target_segments"],
            proposal_summaries=[
                RepairProposalSummary(
                    proposal_id=item["proposal_id"],
                    action=item["action"],
                    reason_code=item["reason_code"],
                    target_segments=item["target_segments"],
                    confidence=item["confidence"],
                )
                for item in summary_dict["proposal_summaries"]
            ],
            terminal_hint=summary_dict["terminal_hint"],
        )
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


def _load_phase10_verification_report(runtime: ToolRuntime, state: MotionAgentState) -> VerificationReport | None:
    if not state.evaluation.latest_verification_id:
        return None
    try:
        payload = runtime.artifact_store.get(state.evaluation.latest_verification_id)
    except FileNotFoundError:
        return None
    if isinstance(payload, VerificationReport):
        return payload
    if not isinstance(payload, dict):
        return None
    if payload.get("schema") == "VerificationReport":
        data = payload.get("verification_report", payload)
        return VerificationReport.model_validate(data)
    required = {"verification_id", "candidate_id", "plan", "findings", "status", "overall_pass"}
    if required.issubset(payload):
        return VerificationReport.model_validate(payload)
    return None


def _load_current_motion_spec(runtime: ToolRuntime, state: MotionAgentState):
    if not state.plan.motion_spec_id:
        return None
    try:
        payload = runtime.artifact_store.get(state.plan.motion_spec_id)
    except FileNotFoundError:
        return None
    if isinstance(payload, MotionSpecification):
        return payload
    if isinstance(payload, dict):
        spec_payload = payload.get("motion_spec", payload)
        try:
            return MotionSpecification.model_validate(spec_payload)
        except Exception:
            return spec_payload
    return None


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
