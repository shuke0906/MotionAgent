"""Smoothness metric check."""

from __future__ import annotations

from motion_agent.verification.schemas import VerificationRequest, VerifierCheck, VerifierFinding
from motion_agent.verification.physical.evidence import metric_value, missing_metric


def verify_smoothness(request: VerificationRequest, check: VerifierCheck) -> VerifierFinding:
    value = metric_value(request, check, "smoothness")
    if value is None:
        return missing_metric(check, "smoothness")
    threshold = float(check.threshold_config.get("smoothness", 10.0))
    return VerifierFinding(
        check_id=f"{check.check_id}_smoothness",
        direction="physical",
        status="pass" if value <= threshold else "fail",
        required=check.required,
        diagnostic_code=None if value <= threshold else "MOTION_JITTER",
        measured_value=value,
        threshold=threshold,
        evaluator_version=request.evaluator_versions.get(check.evaluator, "physical_metrics_v1"),
    )
