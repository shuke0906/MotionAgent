"""MotionCritic adapter boundary."""

from __future__ import annotations

import importlib
from pathlib import Path

import torch

from motion_agent.verification.artifacts import file_digest, load_candidate
from motion_agent.verification.backends import BackendUnavailable, NaturalnessVerifierBackend, finding
from motion_agent.verification.representations import official_repository, smpl_to_motioncritic, resample_critic


class MotionCriticBackend(NaturalnessVerifierBackend):
    evaluator_version = "motioncritic_adapter_v1"

    def __init__(self, config):
        self.config = config
        self.model = None
        self.inference_calls = 0

    def cache_identity(self):
        path = Path(self.config.motioncritic_checkpoint)
        return {"code": self.evaluator_version, "checkpoint": str(path),
                "sha256": file_digest(path) if path.is_file() else "missing",
                "runtime_sha256": file_digest(Path(self.config.motioncritic_repository) / "lib/model/critic.py")
                if (Path(self.config.motioncritic_repository) / "lib/model/critic.py").is_file() else "missing",
                "threshold": self.config.motioncritic_threshold,
                "hand_policy": self.config.motioncritic_hand_policy, "windows": "contiguous_60frames_20hz_v1"}

    def load(self):
        if self.model is not None:
            return
        checkpoint = Path(self.config.motioncritic_checkpoint)
        if not checkpoint.is_file():
            raise BackendUnavailable(f"official MotionCritic checkpoint missing: {checkpoint}")
        with official_repository(self.config.motioncritic_repository, "lib"):
            model_class = importlib.import_module("lib.model.critic").MotionCritic
            model = model_class(depth=3, dim_feat=256, dim_rep=512, mlp_ratio=4)
            weights = torch.load(checkpoint, map_location="cpu", weights_only=True)["model_state_dict"]
            model.load_state_dict({key.removeprefix("module."): value for key, value in weights.items()}, strict=True)
            self.model = model.to(self.config.device).eval()

    def score(self, motion: torch.Tensor) -> list[float]:
        self.load()
        scores = []
        # Evaluate contiguous windows; do not squash long clips into a short pseudo-motion.
        with torch.inference_mode():
            for start in range(0, motion.shape[1], 60):
                window = motion[:, start:start + 60].to(self.config.device)
                output = self.model.batch_critic(window)
                if output.numel() != 1 or not torch.isfinite(output).all():
                    raise ValueError("MotionCritic returned a non-finite or nonscalar score")
                scores.append(float(output.item()))
                self.inference_calls += 1
        return scores

    def verify(self, request, check):
        try:
            self.load()
            params = load_candidate(request)["body_params_global"]
            motion, conversion = smpl_to_motioncritic(params, self.config.motioncritic_hand_policy)
            motion = resample_critic(motion, request.motion_spec.fps)
            conversion.update({"model_fps": 20, "source_fps": request.motion_spec.fps,
                               "resampling": "SO3_slerp_rotations_linear_translation", "model_shape": list(motion.shape)})
            scores = self.score(motion)
            value = min(scores)
            threshold = request.thresholds.get("motioncritic", self.config.motioncritic_threshold)
            status = "uncertain" if threshold is None else ("pass" if value >= threshold else "fail")
            return finding(check, self.evaluator_version, status, measured_value=value, threshold=threshold,
                           diagnostic_code="MOTIONCRITIC_THRESHOLD_UNCALIBRATED" if threshold is None else
                           ("NATURALNESS_MISMATCH" if status == "fail" else None),
                           observed={"window_scores": scores, "aggregation": "minimum", "conversion": conversion,
                                     "source": "real_motioncritic", "threshold_status": self.config.threshold_status})
        except (BackendUnavailable, FileNotFoundError, ModuleNotFoundError) as exc:
            return finding(check, self.evaluator_version, "uncertain", diagnostic_code="MOTIONCRITIC_UNAVAILABLE", message=str(exc))
        except Exception as exc:
            return finding(check, self.evaluator_version, "error", diagnostic_code="MOTIONCRITIC_INFERENCE_ERROR", message=type(exc).__name__)

from motion_agent.verification.schemas import VerificationRequest, VerifierCheck, VerifierFinding


def verify_motioncritic_unavailable(request: VerificationRequest, check: VerifierCheck) -> VerifierFinding:
    return VerifierFinding(
        check_id=check.check_id,
        direction="naturalness",
        status="uncertain",
        required=check.required,
        diagnostic_code="MOTIONCRITIC_UNAVAILABLE",
        message="MotionCritic weights/runtime are not connected.",
        evaluator_version=request.evaluator_versions.get(check.evaluator, "motioncritic_unavailable"),
    )
