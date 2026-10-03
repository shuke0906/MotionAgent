"""Foot-skating metric check."""

from __future__ import annotations

from motion_agent.verification.schemas import VerificationRequest, VerifierCheck, VerifierFinding
from motion_agent.verification.physical.evidence import metric_value, missing_metric


def verify_foot_skating(request: VerificationRequest, check: VerifierCheck) -> VerifierFinding:
    value = metric_value(request, check, "foot_skating")
    if value is None:
        return missing_metric(check, "foot_skating")
    threshold = float(check.threshold_config.get("foot_skating", 0.20))
    return VerifierFinding(
        check_id=f"{check.check_id}_foot_skating",
        direction="physical",
        status="pass" if value <= threshold else "fail",
        required=check.required,
        diagnostic_code=None if value <= threshold else "FOOT_SKATING",
        measured_value=value,
        threshold=threshold,
        evaluator_version=request.evaluator_versions.get(check.evaluator, "physical_metrics_v1"),
    )
