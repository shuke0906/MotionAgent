"""Explicit learned-backend selection; credentials remain in the environment."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field

from motion_agent.llm.config import load_dotenv
from motion_agent.state.schemas import StrictModel


class LearnedVerifierConfig(StrictModel):
    semantic_backend: Literal["tmr", "unavailable", "fixture"] = "tmr"
    naturalness_backend: Literal["motioncritic", "unavailable", "fixture"] = "motioncritic"
    mllm_backend: Literal["real", "unavailable", "fixture"] = "real"
    tmr_repository: str = "vendor/TMR"
    tmr_run_dir: str = "vendor/TMR/models/tmr_humanml3d_guoh3dfeats"
    smplx_model_path: str | None = None
    motioncritic_repository: str = "vendor/MotionCritic/MotionCritic"
    motioncritic_checkpoint: str = "vendor/MotionCritic/MotionCritic/pretrained/motioncritic_pre.pth"
    motioncritic_hand_policy: Literal["require_full_smpl", "neutral_terminal_hands"] = "require_full_smpl"
    motioncritic_required: bool = True
    device: str = "cpu"
    provider: Literal["openai"] = "openai"
    mllm_model: str = "gpt-4o-2024-08-06"
    mllm_api_key_env: str = "MLLM_API_KEY"
    mllm_timeout_s: float = Field(default=60, gt=0)
    mllm_max_calls: int = Field(default=16, ge=0)
    mllm_image_detail: Literal["low", "high"] = "low"
    mllm_contact_sheet: bool = False
    mllm_rate_limit_retry_s: float = Field(default=65, ge=0, le=120)
    output_root: str = "outputs/phase9b_real_verifiers"
    tmr_threshold: float | None = None
    motioncritic_threshold: float | None = None
    threshold_status: str = "PROVISIONAL"
    storyboard_max_frames: int = Field(default=240, ge=2)

    @classmethod
    def load(cls, path: str | Path = "configs/learned_verifiers.yaml"):
        load_dotenv()
        config = cls.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8"))["verification"])
        overrides = {field: os.environ[env] for field, env in {
            "provider": "MLLM_PROVIDER", "mllm_model": "MLLM_MODEL", "tmr_run_dir": "TMR_RUN_DIR",
            "smplx_model_path": "SMPLX_MODEL_PATH", "motioncritic_checkpoint": "MOTIONCRITIC_CHECKPOINT",
        }.items() if os.environ.get(env)}
        if not os.environ.get(config.mllm_api_key_env) and os.environ.get("OPENAI_API_KEY"):
            overrides["mllm_api_key_env"] = "OPENAI_API_KEY"
        return cls.model_validate({**config.model_dump(), **overrides})


def create_verification_service(config: LearnedVerifierConfig):
    from motion_agent.verification.backends import FixtureNaturalnessBackend, UnavailableNaturalnessBackend
    from motion_agent.verification.mllm.backend import MockMLLMBackend, RealMLLMBackend, UnavailableMLLMBackend
    from motion_agent.verification.naturalness.motioncritic import MotionCriticBackend
    from motion_agent.verification.plan_builder import VerificationPlanBuilder
    from motion_agent.verification.semantic.tmr import FixtureSemanticBackend, UnavailableSemanticBackend
    from motion_agent.verification.semantic.real_tmr import TMRBackend
    from motion_agent.verification.service import VerificationService
    from motion_agent.verification.evidence import SMPLXEvidencePreparer

    semantic = {"tmr": lambda: TMRBackend(config), "fixture": FixtureSemanticBackend,
                "unavailable": UnavailableSemanticBackend}[config.semantic_backend]()
    naturalness = {"motioncritic": lambda: MotionCriticBackend(config), "fixture": FixtureNaturalnessBackend,
                   "unavailable": UnavailableNaturalnessBackend}[config.naturalness_backend]()
    mllm = {"real": lambda: RealMLLMBackend(config), "fixture": MockMLLMBackend,
            "unavailable": UnavailableMLLMBackend}[config.mllm_backend]()
    return VerificationService(plan_builder=VerificationPlanBuilder(learned_config=config),
                               semantic_backend=semantic, naturalness_backend=naturalness, mllm_backend=mllm,
                               evidence_preparer=SMPLXEvidencePreparer(config.smplx_model_path))
