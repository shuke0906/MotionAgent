"""KeyframeVerificationSpec executor."""

from __future__ import annotations

from motion_agent.keyframes.schemas import KeyframeVerificationSpec
from motion_agent.verification.schemas import VerificationRequest, VerifierCheck, VerifierFinding


def verify_keyframe(request: VerificationRequest, check: VerifierCheck) -> VerifierFinding:
    spec = KeyframeVerificationSpec.model_validate(check.metadata["spec"])
    rotation_key = f"{spec.keyframe_id}:rotation"
    position_key = f"{spec.keyframe_id}:position"
    rotation = request.observed_measurements.get(rotation_key, 0.0)
    position = request.observed_measurements.get(position_key, 0.0)
    failures: list[str] = []
    if spec.rotation_threshold_deg is not None and rotation > spec.rotation_threshold_deg:
        failures.append("rotation")
    if spec.position_threshold_m is not None and position > spec.position_threshold_m:
        failures.append("position")
    return VerifierFinding(
        check_id=check.check_id,
        direction="keyframe",
        status="fail" if failures else "pass",
        required=check.required,
        diagnostic_code="KEYFRAME_POSE_MISMATCH" if failures else None,
        measured_value=max(float(rotation), float(position)),
        threshold=max(float(spec.rotation_threshold_deg or 0.0), float(spec.position_threshold_m or 0.0)),
        expected={
            "rotation_threshold_deg": spec.rotation_threshold_deg,
            "position_threshold_m": spec.position_threshold_m,
            "target_frame": spec.target_frame,
        },
        observed={"rotation_error_deg": rotation, "position_error_m": position, "failed_metrics": failures},
        evaluator_version=request.evaluator_versions.get(check.evaluator, "keyframe_metrics_v1"),
    )
