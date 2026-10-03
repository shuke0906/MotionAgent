"""AToM-style event integrity fixture verifier."""

from __future__ import annotations

from motion_agent.verification.schemas import VerificationRequest, VerifierCheck, VerifierFinding


def verify_event_integrity(request: VerificationRequest, check: VerifierCheck) -> VerifierFinding:
    required = [segment.action for segment in request.motion_spec.segments]
    observed = request.observed_events.get("present", required)
    missing = [event for event in required if event not in observed]
    return VerifierFinding(
        check_id=check.check_id,
        direction="event_integrity",
        status="fail" if missing else "pass",
        required=check.required,
        diagnostic_code="EVENT_MISSING" if missing else None,
        expected={"events": required},
        observed={"events": observed, "missing": missing},
        evaluator_version=request.evaluator_versions.get(check.evaluator, "event_fixture_v1"),
    )
