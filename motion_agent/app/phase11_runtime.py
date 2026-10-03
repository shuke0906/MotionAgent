"""Canonical local K=1 runtime with explicit injectable component backends."""

from dataclasses import replace
from pathlib import Path

from motion_agent.agent.actions import PlannerAction, PlannerDecision
from motion_agent.agent.guards import PlannerGuardError
from motion_agent.common.fingerprints import stable_fingerprint
from motion_agent.common.ids import new_id
from motion_agent.compiler.schemas import GEMTextCondition, MotionSpecification
from motion_agent.diagnosis import DiagnosisRequest, DiagnosisService
from motion_agent.diagnosis.history import RepairHistory
from motion_agent.diagnosis.normalizer import normalize_failures
from motion_agent.diagnosis.schemas import RepairHistoryEntry
from motion_agent.generation.request_builder import build_generation_request_from_state
from motion_agent.generation.schemas import GenerationResult, MotionCandidate
from motion_agent.state.schemas import (
    BlockedRepairFamily, DiagnosisSummary, RepairHistorySummary, RepairProposalSummary, VerificationSummary,
)
from motion_agent.tournament.k1_selector import SingleCandidateSelector
from motion_agent.verification.artifacts import file_digest
from motion_agent.verification.plan_builder import VerificationPlanBuilder
from motion_agent.verification.schemas import VerificationRequest, VerificationReport, VerifierPolicy


class LocalVerificationPlanBuilder(VerificationPlanBuilder):
    """Local profile uses real visual checks and deterministic tensor sanity checks."""

    def __init__(self, *, visual_frequency=True):
        super().__init__()
        self.visual_frequency = visual_frequency

    def build(self, request):
        plan = super().build(request)
        for check in plan.checks:
            if check.direction in {"semantic", "event_integrity", "event_temporal"} or (
                self.visual_frequency and check.direction == "event_frequency"
            ):
                check.evaluator = "mllm_backend"
                check.render_needed = True
        plan.render_requirements = [{"check_id": c.check_id} for c in plan.checks if c.render_needed]
        plan.evidence_fingerprint = stable_fingerprint([c.model_dump(mode="json") for c in plan.checks])
        return plan


class Phase11LocalRuntime:
    def __init__(self, root, *, compiler, generator, verifier, diagnosis=None):
        self.root = Path(root)
        self.compiler = compiler
        self.generator = generator
        self.verifier = verifier
        self.diagnosis = diagnosis or DiagnosisService()

    def bind(self, runtime):
        return replace(runtime, compiler=self.compiler, phase11=True,
                       action_executors={PlannerAction.GENERATE: self.generate},
                       selection_executor=self.select, verifier_executor=self.verify,
                       diagnosis_executor=self.diagnose, planner_commit=self.commit_decision)

    @staticmethod
    def history(runtime, state):
        if not state.repair.history_artifact_id:
            return []
        payload = runtime.artifact_store.get(state.repair.history_artifact_id)
        return [RepairHistoryEntry.model_validate(entry) for entry in payload["entries"]]

    @staticmethod
    def attach_history(runtime, state, entries):
        handle = runtime.artifact_store.put("repair_history", {"entries": [e.model_dump(mode="json") for e in entries]})
        repair = state.repair.model_copy(deep=True)
        repair.history_artifact_id = handle.artifact_id
        repair.recent_repairs = [RepairHistorySummary(
            failure_signature=e.failure_signature, repair_family=e.repair_family,
            planner_action=e.planner_action.value, target_segments=e.target_segments, outcome=e.outcome,
        ) for e in entries[-6:]]
        repair.blocked_repair_families = [BlockedRepairFamily(
            failure_signature=signature, repair_family=family, reason="NO_IMPROVEMENT",
        ) for signature, family in sorted(RepairHistory(entries).blocked_families())]
        return state.model_copy(update={"repair": repair})

    def commit_decision(self, runtime, state, decision):
        budgets = state.control.budgets
        if budgets.planner_steps_left <= 1 or budgets.iterations_left <= 6:
            if decision.action not in {PlannerAction.ACCEPT, PlannerAction.STOP_FAILED}:
                raise PlannerGuardError("PHASE11_PLANNER_BUDGET_EXHAUSTED")
        if decision.action == PlannerAction.GENERATE and decision.payload.get("num_candidates", 1) != 1:
            raise PlannerGuardError("PHASE11_K1_REQUIRED")
        proposal = None
        diagnosis = state.evaluation.diagnosis_summary
        if diagnosis and decision.action not in {PlannerAction.ACCEPT, PlannerAction.STOP_FAILED}:
            proposal = next((p for p in diagnosis.proposal_summaries
                             if p.action == decision.action.value and p.parameters == decision.payload
                             and p.target_segments == decision.target_segments), None)
            if proposal is None:
                raise PlannerGuardError("PHASE11_REPAIR_NOT_VALIDATED")
            if budgets.repairs_left <= 0:
                raise PlannerGuardError("PHASE11_REPAIR_BUDGET_EXHAUSTED")
            if any(p.failure_signature == proposal.failure_signature and p.repair_family == proposal.repair_family
                   for p in state.repair.blocked_repair_families):
                raise PlannerGuardError("PHASE11_REPAIR_FAMILY_BLOCKED")
        handle = runtime.artifact_store.put("planner_decision", {"schema": "PlannerDecision", **decision.model_dump(mode="json")})
        control = state.control.model_copy(deep=True)
        control.budgets.planner_steps_left -= 1
        updated = state.model_copy(update={"control": control})
        if proposal:
            updated.control.budgets.repairs_left -= 1
            entries = self.history(runtime, state)
            entries.append(RepairHistoryEntry(
                diagnosis_round=state.generation.generation_round,
                failure_signature=proposal.failure_signature, repair_family=proposal.repair_family,
                planner_action=decision.action, target_segments=decision.target_segments or [],
                proposal_fingerprint=proposal.proposal_fingerprint,
            ))
            updated = self.attach_history(runtime, updated, entries)
            updated.repair.active_proposal_id = proposal.proposal_id
        advanced = runtime.reducer._advance(updated, action="PLANNER", result_status="validated",
                                            artifact_ids=[handle.artifact_id], updates={})
        return runtime.state_store.commit(advanced, expected_version=state.state_version,
                                          event_type="planner_decision_committed", output_artifact_ids=[handle.artifact_id])

    def generate(self, runtime, state, decision):
        spec = MotionSpecification.model_validate(runtime.artifact_store.get(state.plan.motion_spec_id)["motion_spec"])
        condition = GEMTextCondition.model_validate(runtime.artifact_store.get(state.plan.gem_text_condition_id)["gem_text_condition"])
        payload = decision.typed_payload
        request = build_generation_request_from_state(
            state, condition, num_candidates=1, seed=state.generation.generation_round + 1,
            strategy=payload.strategy, scope=payload.scope, target_segments=decision.target_segments,
            previous_candidate_id=state.generation.champion_candidate_id,
        ).model_copy(update={"motion_spec": spec})
        if request.condition_bundle.active_constraint_ids or request.condition_bundle.active_keyframe_ids:
            raise ValueError("CPU fixture does not simulate constraint/keyframe satisfaction")
        result, evidence = self.generator.generate(request, round_index=state.generation.generation_round)
        generation = runtime.artifact_store.put("generation", {
            "schema": "GenerationResult", "generation_result": result.model_dump(mode="json"),
            "request": request.model_dump(mode="json"),
            "motion_spec_id": state.plan.motion_spec_id, "text_condition_id": state.plan.gem_text_condition_id,
            "backend": "MOCK_GEM_CPU_FIXTURE", "evidence": evidence,
        }, artifact_id=result.generation_id)
        ids = []
        for candidate in result.candidates:
            runtime.artifact_store.put("candidate", {"schema": "MotionCandidate", "candidate": candidate.model_dump(mode="json"),
                                                       "evidence": evidence}, artifact_id=candidate.candidate_id)
            ids.append(candidate.candidate_id)
        updated = runtime.reducer.commit_generation_result(state, generation_id=generation.artifact_id,
                    candidate_ids=ids, strategy=request.strategy, scope=request.scope, target_segments=request.target_segments)
        return runtime.state_store.commit(updated, expected_version=state.state_version, event_type="generation_committed",
                                          output_artifact_ids=[generation.artifact_id, *ids])

    def select(self, runtime, state):
        payload = runtime.artifact_store.get(state.generation.latest_generation_id)
        result = GenerationResult.model_validate(payload["generation_result"])
        selection = SingleCandidateSelector().select(result)
        if selection.status != "selected" or selection.pairwise_evidence:
            raise ValueError("K1 selection failed")
        handle = runtime.artifact_store.put("selection", {"schema": "SingleCandidateSelectionResult", **selection.model_dump(mode="json")})
        updated = runtime.reducer.set_champion(state, selection.champion_candidate_id, tournament_id=handle.artifact_id)
        return runtime.state_store.commit(updated, expected_version=state.state_version, event_type="k1_selection_committed",
                                          output_artifact_ids=[handle.artifact_id])

    def verify(self, runtime, state):
        spec = MotionSpecification.model_validate(runtime.artifact_store.get(state.plan.motion_spec_id)["motion_spec"])
        payload = runtime.artifact_store.get(state.generation.champion_candidate_id)
        candidate = MotionCandidate.model_validate(payload["candidate"])
        evidence = payload["evidence"]
        request = VerificationRequest(
            verification_id=new_id("verify"), candidate_id=candidate.candidate_id,
            original_request=state.task.original_request, motion_spec=spec, candidate=candidate,
            generation_id=state.generation.latest_generation_id,
            visual_evidence_uri=evidence["path"], visual_evidence_candidate_id=candidate.candidate_id,
            visual_evidence_fingerprint=file_digest(evidence["path"]),
            observed_events=evidence["observed_events"],
            verifier_policy=VerifierPolicy(enable_tmr=False, enable_motioncritic=False, enable_physical_metrics=False),
            threshold_profile="phase11_cpu_control_plane_v1",
        )
        report = self.verifier.verify(request)
        handle = runtime.artifact_store.put("verification", {"schema": "VerificationReport", "verification_report": report.model_dump(mode="json")},
                                            artifact_id=report.verification_id)
        updated = runtime.reducer.commit_verification_report(state, verification_id=handle.artifact_id,
                                                              summary=VerificationSummary.model_validate(report.summary()))
        entries = self.history(runtime, state)
        current = {failure.failure_signature for failure in normalize_failures(report)}
        for entry in entries:
            if entry.outcome == "pending":
                entry.outcome = "resolved" if report.overall_pass else "unchanged" if entry.failure_signature in current else "worse"
                entry.resolved = report.overall_pass
                entry.improved = report.overall_pass
                entry.regressed = entry.outcome == "worse"
        if entries:
            updated = self.attach_history(runtime, updated, entries)
            updated.repair.active_proposal_id = None
        return runtime.state_store.commit(updated, expected_version=state.state_version,
                                          event_type="verification_committed", output_artifact_ids=[handle.artifact_id])

    def diagnose(self, runtime, state):
        report = VerificationReport.model_validate(runtime.artifact_store.get(state.evaluation.latest_verification_id)["verification_report"])
        generation = runtime.artifact_store.get(state.generation.latest_generation_id)
        spec = MotionSpecification.model_validate(runtime.artifact_store.get(state.plan.motion_spec_id)["motion_spec"])
        entries = self.history(runtime, state)
        result = self.diagnosis.diagnose(DiagnosisRequest(
            diagnosis_id=new_id("diag"), verification_report=report, motion_spec=spec,
            generation_request=generation["request"], repair_history=entries,
            budget=state.control.budgets, capability_summary=state.control.capabilities,
        ), state=state)
        # Bind a repair fingerprint to its source evidence so a new candidate is a distinct attempt.
        for proposal in result.repair_proposals:
            proposal.proposal_fingerprint = stable_fingerprint({"proposal": proposal.proposal_fingerprint,
                                                               "verification": report.verification_id})
        if state.control.budgets.repairs_left <= 0 or state.control.budgets.generations_left <= 0:
            result.repair_proposals = []
            result.status = "no_valid_repair"
            result.terminal_hint = "BUDGET_EXHAUSTED"
        elif not result.repair_proposals or all(p.planner_action == PlannerAction.STOP_FAILED for p in result.repair_proposals):
            result.terminal_hint = "UNRECOVERABLE_TOOL_FAILURE"
        handle = runtime.artifact_store.put("diagnosis", {"schema": "DiagnosisResult", "diagnosis_result": result.model_dump(mode="json")},
                                            artifact_id=result.diagnosis_id)
        summary_data = result.summary()
        summary_data["proposal_summaries"] = [RepairProposalSummary(
            proposal_id=p.proposal_id, action=p.planner_action.value, reason_code=p.reason_code,
            target_segments=p.target_segments, confidence=p.confidence,
            repair_family=p.repair_family, failure_signature=p.failure_signature,
            proposal_fingerprint=p.proposal_fingerprint, parameters=p.parameters,
        ) for p in result.repair_proposals]
        updated = runtime.reducer.commit_diagnosis_result(state, diagnosis_id=handle.artifact_id,
                                                          summary=DiagnosisSummary.model_validate(summary_data))
        return runtime.state_store.commit(updated, expected_version=state.state_version, event_type="diagnosis_committed",
                                          output_artifact_ids=[handle.artifact_id])
