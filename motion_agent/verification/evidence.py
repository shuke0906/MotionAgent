"""Reusable FK evidence for deterministic geometry and learned semantic checks."""

from pathlib import Path

from motion_agent.verification.artifacts import candidate_identity, file_digest, load_candidate
from motion_agent.verification.backends import BackendUnavailable
from motion_agent.verification.representations import smplx_joints


class SMPLXEvidencePreparer:
    def __init__(self, model_path):
        self.model_path = model_path
        self.cache = {}

    def __call__(self, request):
        try:
            payload = load_candidate(request)
            if payload.get("joints_yup") is not None:
                return request.model_copy(update={"candidate": payload})
            if not self.model_path or not Path(self.model_path).is_file():
                return request
            identity = (candidate_identity(request), file_digest(self.model_path))
            joints = self.cache.get(identity)
            if joints is None:
                joints = smplx_joints(payload["body_params_global"], self.model_path)
                self.cache[identity] = joints
            return request.model_copy(update={"candidate": {**payload, "joints_yup": joints,
                "fk_evidence": {"model_sha256": identity[1], "joint_order": "SMPL_first22",
                                "coordinate_system": "Y-up/meters", "conversion": "smplx_fk_v2"}}})
        except (BackendUnavailable, FileNotFoundError):
            return request
        except (ValueError, RuntimeError, KeyError, OSError) as exc:
            if isinstance(request.candidate, dict):
                return request.model_copy(update={"candidate": {**request.candidate, "fk_error": type(exc).__name__}})
            return request
