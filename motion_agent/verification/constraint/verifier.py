"""Constraint VerificationSpec executor."""

from __future__ import annotations

from motion_agent.constraints.schemas import VerificationSpec
from motion_agent.verification.schemas import VerificationRequest, VerifierCheck, VerifierFinding


def _spec_from_check(check: VerifierCheck) -> VerificationSpec:
    return VerificationSpec.model_validate(check.metadata["spec"])


def verify_constraint(request: VerificationRequest, check: VerifierCheck) -> VerifierFinding:
    spec = _spec_from_check(check)
    key_candidates = [spec.verification_spec_id, spec.constraint_id, check.check_id]
    value = None
    for key in key_candidates:
        if key in request.observed_measurements:
            value = request.observed_measurements[key]
            break
    if value is None:
        value = float(spec.target.get("observed_value", 0.0))
    passed = float(value) <= spec.pass_threshold
    return VerifierFinding(
        check_id=check.check_id,
        direction="constraint",
        status="pass" if passed else "fail",
        required=check.required,
        diagnostic_code=None if passed else f"CONSTRAINT_{spec.metric_type.upper()}",
        measured_value=float(value),
        threshold=spec.pass_threshold,
        expected={"metric": spec.metric_type, "threshold": spec.pass_threshold, "units": spec.units},
        observed={"value": float(value), "constraint_id": spec.constraint_id},
        evaluator_version=request.evaluator_versions.get(check.evaluator, "constraint_metrics_v1"),
    )
