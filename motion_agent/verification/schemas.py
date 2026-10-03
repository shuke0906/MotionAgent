"""Phase 9 Multi-Verifier schemas."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from motion_agent.compiler.schemas import MotionSpecification
from motion_agent.constraints.schemas import VerificationSpec
from motion_agent.generation.schemas import MotionCandidate
from motion_agent.keyframes.schemas import KeyframeVerificationSpec
from motion_agent.state.schemas import StrictModel


VerifierDirection = Literal[
    "technical",
    "semantic",
    "event_integrity",
    "event_temporal",
    "event_frequency",
    "naturalness",
    "constraint",
    "keyframe",
    "physical",
    "preservation",
    "heading_continuity",
]

FindingStatus = Literal["pass", "fail", "uncertain", "error", "not_applicable"]
ReportStatus = Literal["complete", "incomplete", "invalid", "failed_service"]


class VerifierPolicy(StrictModel):
    enable_mllm_semantic: bool = True
    enable_atom_event_verifier: bool = True
    enable_motioncritic: bool = True
    enable_tmr: bool = True
    enable_physical_metrics: bool = True
    enable_optional_checks: bool = True
    collect_all_failures: bool = True


class VerificationRequest(StrictModel):
    verification_id: str
    candidate_id: str
    original_request: str
    motion_spec: MotionSpecification
    candidate: MotionCandidate | dict[str, Any] | None = None
    constraint_verification_specs: list[VerificationSpec] = Field(default_factory=list)
    keyframe_verification_specs: list[KeyframeVerificationSpec] = Field(default_factory=list)
    generation_id: str | None = None
    generation_scope: Literal["full", "segment"] = "full"
    target_segments: list[int] | None = None
    previous_candidate_id: str | None = None
    previous_candidate: dict[str, Any] | None = None
    verifier_policy: VerifierPolicy = Field(default_factory=VerifierPolicy)
    threshold_profile: str = "phase9_default_v1"
    thresholds: dict[str, float] = Field(default_factory=dict)
    evaluator_versions: dict[str, str] = Field(default_factory=dict)
    render_profile: str = "none"
    prompt_versions: dict[str, str] = Field(default_factory=dict)
    observed_events: dict[str, Any] = Field(default_factory=dict)
    observed_measurements: dict[str, float] = Field(default_factory=dict)
    physical_evidence: dict[str, float] = Field(default_factory=dict)
    semantic_fixture_status: FindingStatus | None = None
    force_checks: list[VerifierDirection] = Field(default_factory=list)
    candidate_fingerprint: str | None = None
    spec_version: str = "motion_spec_v1"
    visual_evidence_uri: str | None = None
    visual_evidence_candidate_id: str | None = None
    visual_evidence_fingerprint: str | None = None
    storyboard_profile: str = "scenario_v1"


class VerifierCheck(StrictModel):
    check_id: str
    direction: VerifierDirection
    evaluator: str
    required: bool
    critical: bool = False
    target_segments: list[int] = Field(default_factory=list)
    body_parts: list[str] = Field(default_factory=list)
    source_spec_ids: list[str] = Field(default_factory=list)
    threshold_config: dict[str, Any] = Field(default_factory=dict)
    render_needed: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class VerificationPlan(StrictModel):
    verification_id: str
    candidate_id: str
    checks: list[VerifierCheck]
    required_check_ids: list[str]
    optional_check_ids: list[str]
    render_requirements: list[dict[str, Any]] = Field(default_factory=list)
    evidence_fingerprint: str


class VerifierFinding(StrictModel):
    check_id: str
    direction: VerifierDirection
    status: FindingStatus
    required: bool
    diagnostic_code: str | None = None
    message: str | None = None
    measured_value: float | None = None
    threshold: float | None = None
    expected: dict[str, Any] = Field(default_factory=dict)
    observed: dict[str, Any] = Field(default_factory=dict)
    evidence_refs: list[str] = Field(default_factory=list)
    evaluator_version: str = "unknown"
    artifact_refs: list[str] = Field(default_factory=list)


class VerificationReport(StrictModel):
    verification_id: str
    candidate_id: str
    plan: VerificationPlan
    findings: list[VerifierFinding]
    failed_required_checks: list[str]
    warnings: list[str] = Field(default_factory=list)
    status: ReportStatus
    overall_pass: bool
    threshold_profile: str
    evaluator_versions: dict[str, str]
    artifact_refs: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "overall_pass": self.overall_pass,
            "passed_check_ids": [f.check_id for f in self.findings if f.status == "pass"],
            "failed_check_ids": [f.check_id for f in self.findings if f.status == "fail"],
            "warning_check_ids": [f.check_id for f in self.findings if not f.required and f.status != "pass"],
            "uncertain_check_ids": [f.check_id for f in self.findings if f.status in {"uncertain", "error"}],
            "critical_failures": [
                f.diagnostic_code for f in self.findings if f.required and f.status != "pass" and f.diagnostic_code
            ],
            "affected_segments": sorted(
                {
                    segment
                    for check in self.plan.checks
                    if any(f.check_id == check.check_id and f.status != "pass" for f in self.findings)
                    for segment in check.target_segments
                }
            ),
            "diagnostic_codes": [f.diagnostic_code for f in self.findings if f.diagnostic_code],
        }
