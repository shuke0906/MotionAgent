"""AToM-style event frequency verifier."""

from __future__ import annotations

from motion_agent.verification.schemas import VerificationRequest, VerifierCheck, VerifierFinding


def verify_event_frequency(request: VerificationRequest, check: VerifierCheck) -> VerifierFinding:
    expected = int(check.metadata["expected_count"])
    action = check.metadata["action"]
    counts = request.observed_events.get("counts", {})
    observed = int(counts.get(str(check.target_segments[0]), counts.get(action, expected)))
    status = "pass" if observed == expected else "fail"
    return VerifierFinding(
        check_id=check.check_id,
        direction="event_frequency",
        status=status,
        required=check.required,
        diagnostic_code=None if status == "pass" else "EVENT_FREQUENCY_MISMATCH",
        measured_value=float(observed),
        threshold=float(expected),
        expected={"count": expected, "action": action},
        observed={"count": observed},
        evaluator_version=request.evaluator_versions.get(check.evaluator, "event_fixture_v1"),
    )
