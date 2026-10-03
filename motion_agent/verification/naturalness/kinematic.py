"""Deterministic naturalness sanity checks."""

from __future__ import annotations

import torch

from motion_agent.verification.schemas import VerificationRequest, VerifierCheck, VerifierFinding


def verify_kinematic_naturalness(request: VerificationRequest, check: VerifierCheck) -> VerifierFinding:
    if not isinstance(request.candidate, dict) or not isinstance(request.candidate.get("motion_repr"), torch.Tensor):
        return VerifierFinding(
            check_id=check.check_id,
            direction="naturalness",
            status="uncertain",
            required=check.required,
            diagnostic_code="KINEMATIC_EVIDENCE_MISSING",
            evaluator_version=request.evaluator_versions.get(check.evaluator, "kinematic_v1"),
        )
    motion = request.candidate["motion_repr"].float()
    if check.metadata.get("use_smpl_parameters"):
        params = request.candidate.get("body_params_global")
        if not params:
            return VerifierFinding(check_id=check.check_id, direction="naturalness", status="uncertain", required=check.required,
                                   diagnostic_code="KINEMATIC_EVIDENCE_MISSING", evaluator_version="kinematic_smpl_v2")
        transl = params["transl"].reshape(-1, 3).float()
        fps = request.motion_spec.fps
        acceleration = torch.diff(transl, n=2, dim=0) * fps ** 2
        jerk = torch.diff(transl, n=3, dim=0) * fps ** 3
        pose = params["body_pose"].reshape(len(transl), -1, 3).float()
        value = float(torch.linalg.vector_norm(jerk, dim=-1).max()) if len(jerk) else 0.0
        threshold = request.thresholds.get("kinematic_jerk", 100.0)
        # Axis-angle step is evidence only; exact SO(3) angular metrics require a separately calibrated rule.
        rotation_spike = float(torch.linalg.vector_norm(torch.diff(pose, dim=0), dim=-1).max()) if len(pose) > 1 else 0.0
        return VerifierFinding(check_id=check.check_id, direction="naturalness", status="pass" if value <= threshold else "fail",
                               required=check.required, measured_value=value, threshold=threshold,
                               diagnostic_code=None if value <= threshold else "KINEMATIC_JITTER",
                               observed={"root_jerk_m_s3": value,
                                         "root_acceleration_m_s2": float(torch.linalg.vector_norm(acceleration, dim=-1).max()) if len(acceleration) else 0.0,
                                         "body_rotation_step_rad": rotation_spike, "source": "actual_smpl_parameters"},
                               evaluator_version="kinematic_smpl_v2")
    jerk_threshold = float(request.thresholds.get("kinematic_jerk", 100.0))
    if motion.shape[0] < 4:
        return VerifierFinding(
            check_id=check.check_id,
            direction="naturalness",
            status="pass",
            required=check.required,
            measured_value=0.0,
            threshold=jerk_threshold,
            evaluator_version=request.evaluator_versions.get(check.evaluator, "kinematic_v1"),
        )
    jerk = torch.diff(motion[:, 148:151], n=3, dim=0)
    value = float(torch.linalg.vector_norm(jerk, dim=1).max().item()) if jerk.numel() else 0.0
    status = "pass" if value <= jerk_threshold else "fail"
    return VerifierFinding(
        check_id=check.check_id,
        direction="naturalness",
        status=status,
        required=check.required,
        diagnostic_code=None if status == "pass" else "KINEMATIC_JITTER",
        measured_value=value,
        threshold=jerk_threshold,
        evaluator_version=request.evaluator_versions.get(check.evaluator, "kinematic_v1"),
    )
