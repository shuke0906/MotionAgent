"""Actual geometric evidence, with missing measurements never treated as zero."""

import torch

from motion_agent.verification.schemas import VerifierFinding


def metric_value(request, check, metric):
    if metric in request.physical_evidence:
        value = float(request.physical_evidence[metric])
        if not torch.isfinite(torch.tensor(value)):
            raise ValueError("physical metric is not finite")
        return value
    if not check.metadata.get("require_actual_evidence"):
        return 0.0
    joints = request.candidate.get("joints_yup") if isinstance(request.candidate, dict) else None
    if joints is None:
        return None
    if not torch.isfinite(joints).all():
        raise ValueError("non-finite physical joints")
    if metric == "ground_penetration":
        return float(torch.clamp(-joints[..., 1].min(), min=0))
    if metric == "smoothness":
        acceleration = torch.diff(joints, n=2, dim=0) * request.motion_spec.fps ** 2
        return float(torch.linalg.vector_norm(acceleration, dim=-1).max()) if len(acceleration) else None
    feet = joints[:, [7, 8, 10, 11]]
    velocities = torch.linalg.vector_norm(torch.diff(feet[..., [0, 2]], dim=0), dim=-1) * request.motion_spec.fps
    contact = (feet[:-1, :, 1] < 0.05) & (feet[1:, :, 1] < 0.05)
    return float(velocities[contact].mean()) if contact.any() else None


def missing_metric(check, metric):
    return VerifierFinding(check_id=f"{check.check_id}_{metric}", direction="physical", status="uncertain",
                           required=check.required, diagnostic_code="PHYSICAL_EVIDENCE_MISSING",
                           message=f"Actual {metric} measurement or FK joints unavailable", evaluator_version="physical_metrics_v1")
