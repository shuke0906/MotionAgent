"""Small version-aware verifier cache."""

from __future__ import annotations

from motion_agent.common.fingerprints import stable_fingerprint
from motion_agent.verification.schemas import VerificationRequest, VerifierCheck, VerifierFinding
from motion_agent.verification.artifacts import candidate_identity, file_digest, evidence_identity


def verifier_cache_key(request: VerificationRequest, check: VerifierCheck, *, backend_identity: str = "none") -> str:
    evaluator_version = request.evaluator_versions.get(check.evaluator, "unknown")
    prompt_version = request.prompt_versions.get(check.direction, "none")
    payload = {
        "candidate_id": request.candidate_id,
        "candidate_fingerprint": candidate_identity(request),
        "motion_spec": request.motion_spec.model_dump(mode="json"),
        "spec_version": request.spec_version,
        "check": check.model_dump(mode="json"),
        "backend_identity": backend_identity,
        "visual_evidence": file_digest(request.visual_evidence_uri) if request.visual_evidence_uri else None,
        "visual_binding": request.visual_evidence_candidate_id,
        "storyboard_profile": request.storyboard_profile,
        "thresholds": request.thresholds,
        "previous_candidate": evidence_identity(request.previous_candidate),
        "target_segments": request.target_segments,
        "semantic_fixture_status": request.semantic_fixture_status,
        "check_type": check.direction,
        "check_id": check.check_id,
        "source_spec_ids": check.source_spec_ids,
        "threshold_profile": request.threshold_profile,
        "threshold_config": check.threshold_config,
        "evaluator": check.evaluator,
        "evaluator_version": evaluator_version,
        "render_profile": request.render_profile,
        "prompt_version": prompt_version,
        "observed_events": request.observed_events,
        "observed_measurements": request.observed_measurements,
        "physical_evidence": request.physical_evidence,
    }
    return stable_fingerprint(payload)


class VerificationCache:
    def __init__(self) -> None:
        self._findings: dict[str, VerifierFinding] = {}
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> VerifierFinding | None:
        finding = self._findings.get(key)
        if finding is None:
            self.misses += 1
        else:
            self.hits += 1
        return finding.model_copy(deep=True) if finding else None

    def put(self, key: str, finding: VerifierFinding) -> None:
        self._findings[key] = finding.model_copy(deep=True)
