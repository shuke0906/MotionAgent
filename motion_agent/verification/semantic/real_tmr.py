"""Real official TMR runtime, shared with retrieval through OfficialTMREncoder."""

from __future__ import annotations

from pathlib import Path

from motion_agent.retrieval.tmr_encoder import OfficialTMREncoder
from motion_agent.verification.artifacts import file_digest, load_candidate
from motion_agent.verification.backends import BackendUnavailable, finding
from motion_agent.verification.representations import joints_to_tmr, smplx_joints
from motion_agent.verification.semantic.text_policy import semantic_text
from motion_agent.verification.semantic.tmr import SemanticVerifierBackend


class TMRBackend(SemanticVerifierBackend):
    evaluator_version = "tmr_adapter_v2"

    def __init__(self, config):
        self.config = config
        self.encoder = OfficialTMREncoder(config.tmr_repository, config.tmr_run_dir, config.device)
        self.model_loaded = False
        self.inference_calls = 0

    def cache_identity(self):
        fk_path = Path(self.config.smplx_model_path) if self.config.smplx_model_path else None
        return {"code": self.evaluator_version, "runtime": self.encoder.cache_identity(),
                "fk_model": self.config.smplx_model_path, "threshold": self.config.tmr_threshold,
                "fk_hash": file_digest(fk_path) if fk_path and fk_path.is_file() else None,
                "text_policy": "dsl_coarse_v1", "conversion": "smplx_fk22_yup_guo20hz_v2"}

    def verify(self, request, check):
        try:
            self.encoder.load()
            self.model_loaded = True
            payload = load_candidate(request)
            joints = payload.get("joints_yup")
            if joints is None:
                joints = smplx_joints(payload["body_params_global"], self.config.smplx_model_path)
            features = joints_to_tmr(joints, request.motion_spec.fps, self.config.tmr_repository)
            text = semantic_text(request.motion_spec)
            score = self.encoder.similarity(text, features)
            self.inference_calls += 1
            threshold = request.thresholds.get("tmr", self.config.tmr_threshold)
            status = "uncertain" if threshold is None else ("pass" if score >= threshold else "fail")
            return finding(check, self.evaluator_version, status, measured_value=score, threshold=threshold,
                           diagnostic_code="TMR_THRESHOLD_UNCALIBRATED" if threshold is None else
                           ("SEMANTIC_MISMATCH" if status == "fail" else None),
                           observed={"semantic_text": text, "feature_shape": list(features.shape), "score_scale": "(cosine+1)/2",
                                     "threshold_status": self.config.threshold_status, "source": "real_tmr"})
        except (BackendUnavailable, FileNotFoundError, ModuleNotFoundError) as exc:
            return finding(check, self.evaluator_version, "uncertain", diagnostic_code="TMR_BACKEND_UNAVAILABLE", message=str(exc))
        except Exception as exc:
            return finding(check, self.evaluator_version, "error", diagnostic_code="TMR_INFERENCE_ERROR", message=type(exc).__name__)
