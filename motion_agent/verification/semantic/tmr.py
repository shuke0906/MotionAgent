"""Semantic verifier backend boundary."""

from __future__ import annotations

from motion_agent.verification.schemas import FindingStatus, VerificationRequest, VerifierCheck, VerifierFinding
from motion_agent.verification.backends import FindingBackend


class SemanticVerifierBackend(FindingBackend):
    evaluator_version = "semantic_backend_unknown"

    def verify(self, request: VerificationRequest, check: VerifierCheck) -> VerifierFinding:
        raise NotImplementedError


class UnavailableSemanticBackend(SemanticVerifierBackend):
    evaluator_version = "tmr_unavailable"

    def verify(self, request: VerificationRequest, check: VerifierCheck) -> VerifierFinding:
        return VerifierFinding(
            check_id=check.check_id,
            direction="semantic",
            status="uncertain",
            required=check.required,
            diagnostic_code="SEMANTIC_BACKEND_UNAVAILABLE",
            message="Production TMR/MLLM semantic verifier is not connected.",
            evaluator_version=self.evaluator_version,
        )


class FixtureSemanticBackend(SemanticVerifierBackend):
    def __init__(self, status: FindingStatus = "pass", *, evaluator_version: str = "fixture_semantic_v1") -> None:
        self.status = status
        self.evaluator_version = evaluator_version
        self.calls = 0

    def cache_identity(self):
        return {"backend": "fixture_semantic", "version": self.evaluator_version, "status": self.status}

    def verify(self, request: VerificationRequest, check: VerifierCheck) -> VerifierFinding:
        self.calls += 1
        status = request.semantic_fixture_status or self.status
        code = None
        if status == "fail":
            code = "SEMANTIC_MISMATCH"
        elif status == "uncertain":
            code = "SEMANTIC_UNCERTAIN"
        elif status == "error":
            code = "SEMANTIC_BACKEND_ERROR"
        return VerifierFinding(
            check_id=check.check_id,
            direction="semantic",
            status=status,
            required=check.required,
            diagnostic_code=code,
            observed={"fixture": True, "original_request": request.original_request},
            evaluator_version=self.evaluator_version,
        )
