"""Phase 10 Diagnosis and targeted repair planning schemas."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, model_validator

from motion_agent.agent.actions import PlannerAction
from motion_agent.state.schemas import BudgetState, CapabilitySummary, StrictModel
from motion_agent.verification.schemas import VerificationReport


FailureFamily = Literal[
    "SEMANTIC",
    "EVENT",
    "CONSTRAINT",
    "KEYFRAME",
    "NATURALNESS",
    "PHYSICAL",
    "PRESERVATION",
    "TECHNICAL",
    "VERIFICATION_INFRA",
    "UNKNOWN",
]

RepairFamily = Literal[
    "SEMANTIC_RECOMPILE",
    "REFERENCE_GROUNDING",
    "CONSTRAINT_REBUILD",
    "KEYFRAME_REBUILD",
    "REGENERATE",
    "PHYSICAL_REGENERATE",
    "PRESERVATION_REPAIR",
    "STOP_UNRESOLVABLE",
]

RootCauseCode = Literal[
    "SPECIFICATION_INCOMPLETE",
    "SEMANTIC_COMPILATION_LOSS",
    "CAPTION_MISMATCH",
    "GENERATION_SEMANTIC_EXECUTION_FAILURE",
    "TEMPORAL_RELATION_EXECUTION_FAILURE",
    "REPETITION_EXECUTION_INSUFFICIENT",
    "MISSING_MOTION_PRIOR",
    "SAMPLING_VARIANCE",
    "CONSTRAINT_TOO_WEAK",
    "CONSTRAINT_TOO_STRONG",
    "CONSTRAINT_TARGET_INADEQUATE",
    "KEYFRAME_INADEQUATE",
    "GUIDANCE_NEEDED",
    "SEGMENT_BOUNDARY_ARTIFACT",
    "PHYSICAL_SAMPLING_ARTIFACT",
    "CONDITION_APPLICATION_FAILURE",
    "TECHNICAL_TOOL_FAILURE",
    "VERIFIER_INFRASTRUCTURE_FAILURE",
    "UNKNOWN",
]

DiagnosisStatus = Literal["no_failure", "diagnosed", "ambiguous", "no_valid_repair"]
RepairOutcome = Literal["pending", "resolved", "improved", "unchanged", "worse", "failed_execution"]
ValidationStatus = Literal["accepted", "rejected"]
ProgressStatus = Literal["failed", "improved", "resolved", "unchanged", "worse"]


REPAIR_FAMILY_TO_PLANNER_ACTION: dict[str, PlannerAction] = {
    "SEMANTIC_RECOMPILE": PlannerAction.COMPILE_MOTION,
    "REFERENCE_GROUNDING": PlannerAction.RETRIEVE_REFERENCE,
    "CONSTRAINT_REBUILD": PlannerAction.BUILD_CONSTRAINT,
    "KEYFRAME_REBUILD": PlannerAction.BUILD_KEYFRAME,
    "REGENERATE": PlannerAction.GENERATE,
    "PHYSICAL_REGENERATE": PlannerAction.GENERATE,
    "PRESERVATION_REPAIR": PlannerAction.GENERATE,
    "STOP_UNRESOLVABLE": PlannerAction.STOP_FAILED,
}


class FailureCase(StrictModel):
    failure_id: str
    source_check_id: str
    verifier_type: str
    diagnostic_code: str
    failure_family: FailureFamily
    severity: Literal["info", "minor", "major", "critical"] = "major"
    required: bool
    candidate_id: str
    target_segments: list[int] = Field(default_factory=list)
    target_body_parts: list[str] = Field(default_factory=list)
    target_event: str | None = None
    expected: dict[str, Any] = Field(default_factory=dict)
    observed: dict[str, Any] = Field(default_factory=dict)
    threshold: float | None = None
    measured_value: float | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    evidence_refs: list[str] = Field(default_factory=list)
    status: Literal["fail", "uncertain", "error"]
    source_evaluator_version: str = "unknown"
    source_spec_ids: list[str] = Field(default_factory=list)
    failure_signature: str


class FailureCluster(StrictModel):
    cluster_id: str
    failure_family: FailureFamily
    target_segments: list[int] = Field(default_factory=list)
    target_body_parts: list[str] = Field(default_factory=list)
    target_event: str | None = None
    primary_failure_id: str
    supporting_failure_ids: list[str] = Field(default_factory=list)
    dependent_failure_ids: list[str] = Field(default_factory=list)
    failure_signature: str


class RootCauseHypothesis(StrictModel):
    hypothesis_id: str
    cluster_id: str
    cause_code: RootCauseCode
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_failure_ids: list[str]
    supporting_facts: list[str] = Field(default_factory=list)
    contradicting_facts: list[str] = Field(default_factory=list)
    affected_segments: list[int] = Field(default_factory=list)
    ambiguous: bool = False


class RepairTarget(StrictModel):
    target_segments: list[int] = Field(default_factory=list)
    target_body_parts: list[str] = Field(default_factory=list)
    target_event: str | None = None
    scope: Literal["full", "segment"] = "full"


class RepairProposal(StrictModel):
    proposal_id: str
    repair_family: RepairFamily
    planner_action: PlannerAction
    reason_code: str
    target_segments: list[int] | None = None
    target_body_parts: list[str] = Field(default_factory=list)
    focus: list[str] = Field(default_factory=list)
    parameters: dict[str, Any] = Field(default_factory=dict)
    evidence_refs: list[str] = Field(default_factory=list)
    expected_effect: str
    preserve_requirements: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    blocked: bool = False
    rejected_reason: str | None = None
    failure_signature: str
    proposal_fingerprint: str

    @model_validator(mode="after")
    def validate_family_action_alignment(self) -> "RepairProposal":
        expected = REPAIR_FAMILY_TO_PLANNER_ACTION[self.repair_family]
        if self.planner_action != expected:
            raise ValueError(f"{self.repair_family} must map to {expected.value}")
        return self


class ProposalValidationResult(StrictModel):
    proposal_id: str
    status: ValidationStatus
    reason_codes: list[str] = Field(default_factory=list)


class RepairHistoryEntry(StrictModel):
    diagnosis_round: int
    failure_signature: str
    repair_family: RepairFamily
    planner_action: PlannerAction
    target_segments: list[int] = Field(default_factory=list)
    target_body_parts: list[str] = Field(default_factory=list)
    pre_repair_metric: float | None = None
    post_repair_metric: float | None = None
    threshold: float | None = None
    outcome: RepairOutcome = "pending"
    improved: bool = False
    resolved: bool = False
    regressed: bool = False
    new_failures: list[str] = Field(default_factory=list)
    proposal_fingerprint: str


class NoImprovementState(StrictModel):
    failure_signature: str
    repair_family: RepairFamily
    unchanged_or_worse_attempts: int = 0
    blocked: bool = False


class DiagnosisRequest(StrictModel):
    diagnosis_id: str
    verification_report: VerificationReport
    motion_spec: Any | None = None
    generation_request: Any | None = None
    source_trace: dict[str, Any] = Field(default_factory=dict)
    repair_history: list[RepairHistoryEntry] = Field(default_factory=list)
    budget: BudgetState = Field(default_factory=BudgetState)
    capability_summary: CapabilitySummary = Field(default_factory=CapabilitySummary)
    diagnosis_policy_version: str = "phase10_rules_v1"
    rule_version: str = "phase10_rules_v1"
    llm_model_version: str | None = None
    prompt_version: str | None = None


class DiagnosisMetrics(StrictModel):
    repair_success_rate: float | None = None
    resolved_failure_rate: float | None = None
    improvement_rate: float | None = None
    no_improvement_rate: float | None = None
    regression_rate: float | None = None
    average_repair_rounds: float | None = None
    repeated_repair_family_rate: float | None = None
    budget_exhaustion_rate: float | None = None
    targeted_repair_success: bool | None = None


class DiagnosisResult(StrictModel):
    diagnosis_id: str
    verification_id: str
    source_verification_report_id: str
    status: DiagnosisStatus
    failure_cases: list[FailureCase] = Field(default_factory=list)
    failure_clusters: list[FailureCluster] = Field(default_factory=list)
    root_causes: list[RootCauseHypothesis] = Field(default_factory=list)
    repair_proposals: list[RepairProposal] = Field(default_factory=list)
    rejected_proposals: list[ProposalValidationResult] = Field(default_factory=list)
    unresolved_failures: list[str] = Field(default_factory=list)
    terminal_hint: Literal["UNRECOVERABLE_TOOL_FAILURE", "BUDGET_EXHAUSTED"] | None = None
    no_improvement_state: list[NoImprovementState] = Field(default_factory=list)
    preserved_evidence_refs: list[str] = Field(default_factory=list)
    repair_history_summary: list[RepairHistoryEntry] = Field(default_factory=list)
    metrics: DiagnosisMetrics = Field(default_factory=DiagnosisMetrics)
    evidence_fingerprint: str
    cache_key: str
    llm_status: str = "IMPLEMENTED_NOT_EXECUTED_NO_CREDENTIALS"
    diagnosis_summary: str

    def summary(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "primary_failures": [f.diagnostic_code for f in self.failure_cases],
            "root_causes": [h.cause_code for h in self.root_causes],
            "target_segments": sorted({s for f in self.failure_cases for s in f.target_segments}),
            "proposal_summaries": [
                {
                    "proposal_id": p.proposal_id,
                    "action": p.planner_action.value,
                    "reason_code": p.reason_code,
                    "target_segments": p.target_segments,
                    "confidence": p.confidence,
                }
                for p in self.repair_proposals
            ],
            "terminal_hint": self.terminal_hint,
        }
