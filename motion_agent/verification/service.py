"""Phase 9 verification service."""

from __future__ import annotations

from motion_agent.verification.aggregator import aggregate_verification
from motion_agent.verification.cache import VerificationCache, verifier_cache_key
from motion_agent.verification.constraint import verify_constraint
from motion_agent.verification.event import verify_event_frequency, verify_event_integrity, verify_event_temporal
from motion_agent.verification.keyframe import verify_keyframe
from motion_agent.verification.naturalness import verify_kinematic_naturalness
from motion_agent.verification.physical import verify_foot_skating, verify_ground_penetration, verify_smoothness
from motion_agent.verification.plan_builder import VerificationPlanBuilder
from motion_agent.verification.preservation import verify_preservation
from motion_agent.verification.schemas import VerificationRequest, VerificationReport, VerifierCheck, VerifierFinding
from motion_agent.verification.semantic import SemanticVerifierBackend, UnavailableSemanticBackend
from motion_agent.verification.store import VerificationStore
from motion_agent.verification.backends import BackendUnavailable, UnavailableNaturalnessBackend, backend_fingerprint
from motion_agent.verification.artifacts import load_candidate
from motion_agent.verification.technical import run_technical_verifier


class VerificationService:
    def __init__(
        self,
        *,
        plan_builder: VerificationPlanBuilder | None = None,
        semantic_backend: SemanticVerifierBackend | None = None,
        naturalness_backend=None,
        mllm_backend=None,
        evidence_preparer=None,
        cache: VerificationCache | None = None,
        store: VerificationStore | None = None,
    ) -> None:
        self.plan_builder = plan_builder or VerificationPlanBuilder()
        self.semantic_backend = semantic_backend or UnavailableSemanticBackend()
        self.naturalness_backend = naturalness_backend or UnavailableNaturalnessBackend()
        if mllm_backend is None:
            from motion_agent.verification.mllm.backend import UnavailableMLLMBackend
            mllm_backend = UnavailableMLLMBackend()
        self.mllm_backend = mllm_backend
        self.evidence_preparer = evidence_preparer
        self.cache = cache or VerificationCache()
        self.store = store or VerificationStore()

    def verify(self, request: VerificationRequest) -> VerificationReport:
        if request.candidate is not None and not isinstance(request.candidate, dict):
            try:
                request = request.model_copy(update={"candidate": load_candidate(request)})
            except (FileNotFoundError, ValueError, OSError):
                pass
        if self.evidence_preparer is not None:
            request = self.evidence_preparer(request)
        plan = self.plan_builder.build(request)
        findings: list[VerifierFinding] = []
        technical_failed = False
        for check in plan.checks:
            if technical_failed and check.direction != "technical":
                continue
            if check.direction == "physical":
                physical_findings = self._run_physical_group(request, check)
                findings.extend(physical_findings)
                if any(f.status != "pass" for f in physical_findings):
                    group_status = next((status for status in ("error", "uncertain", "fail")
                                         if any(f.status == status for f in physical_findings)), "uncertain")
                    findings.append(
                        VerifierFinding(
                            check_id=check.check_id,
                            direction="physical",
                            status=group_status,
                            required=check.required,
                            diagnostic_code="PHYSICAL_METRIC_FAILURE" if group_status == "fail" else "PHYSICAL_EVIDENCE_INCOMPLETE",
                            evaluator_version=request.evaluator_versions.get(check.evaluator, "physical_metrics_v1"),
                        )
                    )
                else:
                    findings.append(
                        VerifierFinding(
                            check_id=check.check_id,
                            direction="physical",
                            status="pass",
                            required=check.required,
                            evaluator_version=request.evaluator_versions.get(check.evaluator, "physical_metrics_v1"),
                        )
                    )
                continue
            finding = self._run_cached(request, check)
            findings.append(finding)
            if check.direction == "technical" and finding.status != "pass":
                technical_failed = True
        report = aggregate_verification(request, plan, findings)
        return self.store.put(report)

    def _run_cached(self, request: VerificationRequest, check: VerifierCheck) -> VerifierFinding:
        backend = self._backend_for(check)
        try:
            key = verifier_cache_key(request, check, backend_identity=backend_fingerprint(backend) if backend else "deterministic_v1")
        except (FileNotFoundError, OSError, ValueError, BackendUnavailable):
            return self._run_check(request, check)
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        try:
            finding = self._run_check(request, check)
        except Exception as exc:
            return VerifierFinding(check_id=check.check_id, direction=check.direction, status="error", required=check.required,
                                   diagnostic_code="VERIFIER_EXECUTION_ERROR", message=type(exc).__name__,
                                   evaluator_version=request.evaluator_versions.get(check.evaluator, "unknown"))
        if finding.status not in {"error", "uncertain"}:
            self.cache.put(key, finding)
        return finding

    def _backend_for(self, check):
        if check.evaluator == "mllm_backend":
            return self.mllm_backend
        if check.evaluator == "motioncritic_adapter":
            return self.naturalness_backend
        if check.direction == "semantic" and check.evaluator != "heading_geometry_v1":
            return self.semantic_backend
        return None

    def _run_check(self, request: VerificationRequest, check: VerifierCheck) -> VerifierFinding:
        if check.evaluator == "heading_geometry_v1":
            import math
            value = request.observed_measurements[check.metadata["metric"]]
            threshold = check.threshold_config["minimum_turn_deg"]
            if not math.isfinite(value):
                return VerifierFinding(check_id=check.check_id, direction="semantic", status="error", required=check.required,
                                       diagnostic_code="INVALID_HEADING_EVIDENCE", evaluator_version="heading_geometry_v1")
            oriented = value if check.metadata["direction"] == "left" else -value
            return VerifierFinding(check_id=check.check_id, direction="semantic", required=check.required,
                                   status="pass" if oriented > threshold else "fail", measured_value=value,
                                   threshold=threshold, expected={"direction": check.metadata["direction"]},
                                   diagnostic_code=None if oriented > threshold else "DIRECTION_MISMATCH",
                                   observed={"source": "deterministic_geometry", "units": "degrees", "left_positive": True},
                                   evaluator_version="heading_geometry_v1")
        backend = self._backend_for(check)
        if backend is not None:
            return backend.verify(request, check)
        if check.direction == "technical":
            return run_technical_verifier(request, check)
        if check.direction == "event_integrity":
            return verify_event_integrity(request, check)
        if check.direction == "event_temporal":
            return verify_event_temporal(request, check)
        if check.direction == "event_frequency":
            return verify_event_frequency(request, check)
        if check.direction == "naturalness" and check.evaluator == "kinematic_v1":
            return verify_kinematic_naturalness(request, check)
        if check.direction == "constraint":
            return verify_constraint(request, check)
        if check.direction == "keyframe":
            return verify_keyframe(request, check)
        if check.direction == "preservation":
            return verify_preservation(request, check)
        return VerifierFinding(
            check_id=check.check_id,
            direction=check.direction,
            status="error",
            required=check.required,
            diagnostic_code="VERIFIER_NOT_IMPLEMENTED",
            evaluator_version=request.evaluator_versions.get(check.evaluator, "unknown"),
        )

    def _run_physical_group(self, request: VerificationRequest, check: VerifierCheck) -> list[VerifierFinding]:
        try:
            if isinstance(request.candidate, dict) and request.candidate.get("fk_error"):
                raise ValueError("FK evidence preparation failed")
            return [verify_foot_skating(request, check), verify_ground_penetration(request, check), verify_smoothness(request, check)]
        except (ValueError, RuntimeError, IndexError) as exc:
            return [VerifierFinding(check_id=f"{check.check_id}_evidence", direction="physical", status="error",
                                    required=check.required, diagnostic_code="PHYSICAL_EVIDENCE_INVALID", message=type(exc).__name__,
                                    evaluator_version="physical_metrics_v1")]
