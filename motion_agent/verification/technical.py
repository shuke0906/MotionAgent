"""Deterministic technical verifier."""

from __future__ import annotations

from typing import Any

import torch

from motion_agent.verification.schemas import VerificationRequest, VerifierCheck, VerifierFinding
from motion_agent.verification.artifacts import load_candidate


def _candidate_payload(request: VerificationRequest) -> dict[str, Any] | None:
    try:
        return load_candidate(request)
    except (FileNotFoundError, ValueError, OSError):
        return None


def run_technical_verifier(request: VerificationRequest, check: VerifierCheck) -> VerifierFinding:
    payload = _candidate_payload(request)
    if payload is None:
        return VerifierFinding(
            check_id=check.check_id,
            direction="technical",
            status="error",
            required=check.required,
            diagnostic_code="CANDIDATE_ARTIFACT_MISSING",
            message="candidate payload or artifact handle is unavailable",
            evaluator_version=request.evaluator_versions.get(check.evaluator, "technical_v1"),
        )
    motion = payload.get("motion_repr")
    if motion is None or not isinstance(motion, torch.Tensor):
        return VerifierFinding(
            check_id=check.check_id,
            direction="technical",
            status="fail",
            required=check.required,
            diagnostic_code="MOTION_TENSOR_MISSING",
            evaluator_version=request.evaluator_versions.get(check.evaluator, "technical_v1"),
        )
    if motion.ndim != 2:
        return VerifierFinding(
            check_id=check.check_id,
            direction="technical",
            status="fail",
            required=check.required,
            diagnostic_code="MOTION_SHAPE_INVALID",
            observed={"shape": list(motion.shape)},
            evaluator_version=request.evaluator_versions.get(check.evaluator, "technical_v1"),
        )
    if motion.shape[0] != request.motion_spec.total_frames or motion.shape[1] != 151:
        return VerifierFinding(
            check_id=check.check_id,
            direction="technical",
            status="fail",
            required=check.required,
            diagnostic_code="MOTION_DIMENSION_MISMATCH",
            expected={"frames": request.motion_spec.total_frames, "dim": 151},
            observed={"shape": list(motion.shape)},
            evaluator_version=request.evaluator_versions.get(check.evaluator, "technical_v1"),
        )
    if torch.isnan(motion).any():
        return VerifierFinding(
            check_id=check.check_id,
            direction="technical",
            status="fail",
            required=check.required,
            diagnostic_code="MOTION_NAN",
            evaluator_version=request.evaluator_versions.get(check.evaluator, "technical_v1"),
        )
    if torch.isinf(motion).any():
        return VerifierFinding(
            check_id=check.check_id,
            direction="technical",
            status="fail",
            required=check.required,
            diagnostic_code="MOTION_INF",
            evaluator_version=request.evaluator_versions.get(check.evaluator, "technical_v1"),
        )
    smpl = payload.get("body_params_global")
    required_keys = {"body_pose", "global_orient", "transl", "betas"}
    if not isinstance(smpl, dict) or not required_keys.issubset(smpl.keys()):
        return VerifierFinding(
            check_id=check.check_id,
            direction="technical",
            status="fail",
            required=check.required,
            diagnostic_code="SMPL_GLOBAL_MISSING",
            observed={"keys": sorted(smpl.keys()) if isinstance(smpl, dict) else []},
            evaluator_version=request.evaluator_versions.get(check.evaluator, "technical_v1"),
        )
    if any(not isinstance(smpl[key], torch.Tensor) or not torch.isfinite(smpl[key]).all() for key in required_keys):
        return VerifierFinding(check_id=check.check_id, direction="technical", status="fail", required=check.required,
                               diagnostic_code="SMPL_NONFINITE_OR_INVALID", evaluator_version="technical_v1")
    return VerifierFinding(
        check_id=check.check_id,
        direction="technical",
        status="pass",
        required=check.required,
        observed={"shape": list(motion.shape), "smpl_keys": sorted(smpl.keys())},
        evaluator_version=request.evaluator_versions.get(check.evaluator, "technical_v1"),
    )
