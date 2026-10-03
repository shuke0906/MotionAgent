"""Provider-independent finding contract and safe backend failure handling."""

from __future__ import annotations

from motion_agent.common.fingerprints import stable_fingerprint
from motion_agent.verification.schemas import VerificationRequest, VerifierCheck, VerifierFinding


class BackendUnavailable(RuntimeError):
    pass


class FindingBackend:
    evaluator_version = "backend_v1"

    def cache_identity(self) -> dict:
        return {"backend": type(self).__name__, "code_version": self.evaluator_version}

    def verify(self, request: VerificationRequest, check: VerifierCheck) -> VerifierFinding:
        raise NotImplementedError


class NaturalnessVerifierBackend(FindingBackend):
    pass


class MLLMVerifierBackend(FindingBackend):
    pass


def finding(check: VerifierCheck, version: str, status: str, **kwargs) -> VerifierFinding:
    return VerifierFinding(check_id=check.check_id, direction=check.direction, required=check.required,
                           status=status, evaluator_version=version, **kwargs)


class FixtureNaturalnessBackend(NaturalnessVerifierBackend):
    def __init__(self, status: str = "pass"):
        self.status = status

    def verify(self, request, check):
        return finding(check, "fixture_naturalness_v1", self.status, observed={"fixture": True})

    def cache_identity(self):
        return {"backend": "fixture_naturalness_v1", "status": self.status}


class UnavailableNaturalnessBackend(NaturalnessVerifierBackend):
    def verify(self, request, check):
        return finding(check, "motioncritic_unavailable", "uncertain", diagnostic_code="MOTIONCRITIC_UNAVAILABLE")


def backend_fingerprint(backend) -> str:
    identity = backend.cache_identity() if hasattr(backend, "cache_identity") else {
        "backend": type(backend).__name__, "version": backend.evaluator_version,
        "fixture_status": getattr(backend, "status", None),
    }
    return stable_fingerprint(identity)
