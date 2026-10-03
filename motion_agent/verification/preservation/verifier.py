"""Segment preservation verifier."""

from __future__ import annotations

import torch

from motion_agent.verification.schemas import VerificationRequest, VerifierCheck, VerifierFinding


def verify_preservation(request: VerificationRequest, check: VerifierCheck) -> VerifierFinding:
    if not isinstance(request.candidate, dict) or request.previous_candidate is None:
        return VerifierFinding(
            check_id=check.check_id,
            direction="preservation",
            status="error",
            required=check.required,
            diagnostic_code="PRESERVATION_REFERENCE_MISSING",
            evaluator_version=request.evaluator_versions.get(check.evaluator, "preservation_metrics_v1"),
        )
    current = request.candidate.get("motion_repr")
    previous = request.previous_candidate.get("motion_repr")
    if not isinstance(current, torch.Tensor) or not isinstance(previous, torch.Tensor) or current.shape != previous.shape:
        return VerifierFinding(
            check_id=check.check_id,
            direction="preservation",
            status="error",
            required=check.required,
            diagnostic_code="PRESERVATION_TENSOR_INVALID",
            evaluator_version=request.evaluator_versions.get(check.evaluator, "preservation_metrics_v1"),
        )
    preserve = torch.ones(current.shape[0], dtype=torch.bool)
    for segment_id in request.target_segments or []:
        start, end = request.motion_spec.segments[segment_id].start_frame, request.motion_spec.segments[segment_id].end_frame
        if start is not None and end is not None:
            preserve[start:end] = False
    diff = current[preserve] - previous[preserve]
    rmse = float(torch.sqrt(torch.mean(diff * diff)).item()) if diff.numel() else 0.0
    threshold = float(check.threshold_config.get("outside_segment_rmse", 0.05))
    return VerifierFinding(
        check_id=check.check_id,
        direction="preservation",
        status="pass" if rmse <= threshold else "fail",
        required=check.required,
        diagnostic_code=None if rmse <= threshold else "OUTSIDE_SEGMENT_CHANGED",
        measured_value=rmse,
        threshold=threshold,
        evaluator_version=request.evaluator_versions.get(check.evaluator, "preservation_metrics_v1"),
    )
