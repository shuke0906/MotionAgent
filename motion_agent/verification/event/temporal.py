"""AToM-style event temporal verifier."""

from __future__ import annotations

from motion_agent.verification.schemas import VerificationRequest, VerifierCheck, VerifierFinding


def verify_event_temporal(request: VerificationRequest, check: VerifierCheck) -> VerifierFinding:
    expected_order = [segment.action for segment in sorted(request.motion_spec.segments, key=lambda s: s.segment_id)]
    observed_order = request.observed_events.get("order", expected_order)
    status = "pass" if observed_order == expected_order else "fail"
    return VerifierFinding(
        check_id=check.check_id,
        direction="event_temporal",
        status=status,
        required=check.required,
        diagnostic_code=None if status == "pass" else "EVENT_ORDER_MISMATCH",
        expected={"order": expected_order},
        observed={"order": observed_order},
        evaluator_version=request.evaluator_versions.get(check.evaluator, "event_fixture_v1"),
    )
