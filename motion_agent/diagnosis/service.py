"""Phase 10 Diagnosis and targeted repair planning service."""

from __future__ import annotations

from motion_agent.common.fingerprints import evidence_fingerprint
from motion_agent.diagnosis.cache import DiagnosisCache, diagnosis_cache_key
from motion_agent.diagnosis.clustering import cluster_failures
from motion_agent.diagnosis.history import RepairHistory
from motion_agent.diagnosis.llm import DiagnosisLLMBackend, UnavailableDiagnosisLLMBackend
from motion_agent.diagnosis.normalizer import normalize_failures
from motion_agent.diagnosis.proposal_validator import validate_repair_proposal
from motion_agent.diagnosis.proposals import build_repair_proposals, deduplicate_proposals
from motion_agent.diagnosis.root_cause import diagnose_root_causes
from motion_agent.diagnosis.routing import route_repair_families
from motion_agent.diagnosis.schemas import (
    DiagnosisMetrics,
    DiagnosisRequest,
    DiagnosisResult,
    ProposalValidationResult,
    RepairProposal,
)
from motion_agent.diagnosis.store import DiagnosisStore
from motion_agent.state.schemas import MotionAgentState


class DiagnosisService:
    def __init__(
        self,
        *,
        cache: DiagnosisCache | None = None,
        store: DiagnosisStore | None = None,
        llm_backend: DiagnosisLLMBackend | None = None,
    ) -> None:
        self.cache = cache or DiagnosisCache()
        self.store = store or DiagnosisStore()
        self.llm_backend = llm_backend or UnavailableDiagnosisLLMBackend()

    def diagnose(self, request: DiagnosisRequest, *, state: MotionAgentState | None = None) -> DiagnosisResult:
        key = diagnosis_cache_key(request)
        cached = self.cache.get(key)
        if cached is not None:
            return cached

        report = request.verification_report
        evidence_fp = evidence_fingerprint(report.model_dump(mode="json"))
        if report.overall_pass:
            result = DiagnosisResult(
                diagnosis_id=request.diagnosis_id,
                verification_id=report.verification_id,
                source_verification_report_id=report.verification_id,
                status="no_failure",
                evidence_fingerprint=evidence_fp,
                cache_key=key,
                llm_status=self.llm_backend.status,
                diagnosis_summary="No repair required.",
            )
            return self._save(key, result)

        if report.status == "failed_service":
            result = DiagnosisResult(
                diagnosis_id=request.diagnosis_id,
                verification_id=report.verification_id,
                source_verification_report_id=report.verification_id,
                status="no_valid_repair",
                terminal_hint="UNRECOVERABLE_TOOL_FAILURE",
                evidence_fingerprint=evidence_fp,
                cache_key=key,
                llm_status=self.llm_backend.status,
                diagnosis_summary="Verification infrastructure failed; no motion repair is supported.",
            )
            return self._save(key, result)

        failures = normalize_failures(report)
        clusters = cluster_failures(failures)
        repair_history = RepairHistory(request.repair_history)
        root_causes = diagnose_root_causes(
            clusters=clusters,
            failures=failures,
            motion_spec=request.motion_spec or request.source_trace.get("motion_spec"),
            generation_request=request.generation_request or request.source_trace.get("generation_request"),
            history=request.repair_history,
        )

        preserve = _preserve_requirements(report)
        proposals: list[RepairProposal] = []
        for cluster in clusters:
            cause = next(cause for cause in root_causes if cause.cluster_id == cluster.cluster_id)
            families = route_repair_families(cluster=cluster, cause=cause, history=request.repair_history)
            proposals.extend(
                build_repair_proposals(
                    cluster=cluster,
                    cause=cause,
                    repair_families=families,
                    preserve=preserve,
                )
            )
        proposals = deduplicate_proposals(proposals)

        valid: list[RepairProposal] = []
        rejected: list[ProposalValidationResult] = []
        seen = repair_history.duplicate_fingerprints()
        blocked = repair_history.blocked_families()
        accepted_fingerprints: set[str] = set()
        for proposal in proposals:
            validation = validate_repair_proposal(
                proposal,
                state=state,
                blocked_families=blocked,
                seen_fingerprints=seen | accepted_fingerprints,
            )
            if validation.status == "accepted":
                valid.append(proposal)
                accepted_fingerprints.add(proposal.proposal_fingerprint)
            else:
                rejected.append(validation)

        status = "diagnosed"
        terminal_hint = None
        if not valid:
            status = "no_valid_repair"
            terminal_hint = "BUDGET_EXHAUSTED" if request.budget.generations_left <= 0 else None
        elif any(cause.ambiguous or cause.cause_code == "UNKNOWN" for cause in root_causes):
            status = "ambiguous"

        result = DiagnosisResult(
            diagnosis_id=request.diagnosis_id,
            verification_id=report.verification_id,
            source_verification_report_id=report.verification_id,
            status=status,  # type: ignore[arg-type]
            failure_cases=failures,
            failure_clusters=clusters,
            root_causes=root_causes,
            repair_proposals=valid[:3],
            rejected_proposals=rejected,
            unresolved_failures=[failure.failure_signature for failure in failures],
            terminal_hint=terminal_hint,
            no_improvement_state=repair_history.no_improvement_states(),
            preserved_evidence_refs=preserve,
            repair_history_summary=request.repair_history,
            metrics=DiagnosisMetrics(),
            evidence_fingerprint=evidence_fp,
            cache_key=key,
            llm_status=self.llm_backend.status,
            diagnosis_summary=_summary(failures, root_causes, valid),
        )
        return self._save(key, result)

    def _save(self, key: str, result: DiagnosisResult) -> DiagnosisResult:
        self.cache.put(key, result)
        return self.store.put(result)


def _preserve_requirements(report) -> list[str]:
    preserved = [f"check:{finding.check_id}" for finding in report.findings if finding.status == "pass"]
    for ref in report.evidence_refs:
        preserved.append(f"evidence:{ref}")
    return preserved


def _summary(failures, root_causes, proposals) -> str:
    if not failures:
        return "No verification failures."
    causes = ", ".join(cause.cause_code for cause in root_causes)
    actions = ", ".join(proposal.planner_action.value for proposal in proposals) or "none"
    return f"Diagnosed {len(failures)} failure(s); root causes: {causes}; proposed planner actions: {actions}."
