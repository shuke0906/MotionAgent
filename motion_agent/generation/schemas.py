"""Canonical Phase 4 generation schemas."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, model_validator

from motion_agent.compiler.schemas import GEMTextCondition
from motion_agent.state.schemas import StrictModel


class GenerationOutputPolicy(StrictModel):
    persist_tensors: bool = True
    persist_raw_prediction: bool = False
    output_root: str = "outputs/phase4_validation"
    artifact_root: str = "artifacts/phase4_validation"


class GenerationConditionBundle(StrictModel):
    text_condition_id: str
    hard_motion_condition_handle: str | None = None
    reward_specs: list[dict[str, Any]] = Field(default_factory=list)
    active_constraint_ids: list[str] = Field(default_factory=list)
    active_keyframe_ids: list[str] = Field(default_factory=list)
    active_reference_ids: list[str] = Field(default_factory=list)
    condition_fingerprint: str


class GenerationRequest(StrictModel):
    generation_id: str
    strategy: Literal["normal", "guided"] = "normal"
    scope: Literal["full", "segment"] = "full"
    target_segments: list[int] | None = None
    fps: int = 30
    total_frames: int
    text_condition: GEMTextCondition
    condition_bundle: GenerationConditionBundle
    previous_candidate_id: str | None = None
    num_candidates: int = 1
    seeds: list[int]
    postprocess_policy: Literal["gem_default", "none", "constraint_safe"] = "gem_default"
    output_policy: GenerationOutputPolicy = Field(default_factory=GenerationOutputPolicy)
    timeout_s: float | None = None

    @model_validator(mode="after")
    def validate_request_shape(self) -> "GenerationRequest":
        if self.total_frames <= 0:
            raise ValueError("total_frames must be positive")
        if self.num_candidates <= 0:
            raise ValueError("num_candidates must be positive")
        if len(self.seeds) != self.num_candidates:
            raise ValueError("len(seeds) must equal num_candidates")
        if self.text_condition.total_frames != self.total_frames:
            raise ValueError("text_condition.total_frames must match request.total_frames")
        return self


class GenerationPreflightReport(StrictModel):
    status: Literal["ready", "blocked"]
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    estimated_frames: int
    hard_mask_density: float = 0.0
    condition_fingerprint: str


class SamplerMetadata(StrictModel):
    sampler: str = "gem_checkpoint_default"
    seed: int
    torch_version: str | None = None
    cuda_version: str | None = None
    checkpoint_version: str
    gem_config_hash: str | None = None


class CandidateMetadata(StrictModel):
    seed: int
    generation_id: str
    condition_fingerprint: str
    candidate_fingerprint: str
    checkpoint_version: str
    gem_config_hash: str | None = None
    sampler: SamplerMetadata
    scope: str
    target_segments: list[int] | None = None
    postprocess_policy: str
    active_constraint_ids: list[str] = Field(default_factory=list)
    active_keyframe_ids: list[str] = Field(default_factory=list)
    active_reference_ids: list[str] = Field(default_factory=list)
    runtime_ms: float
    peak_gpu_memory_mb: float | None = None
    frame_count: int
    motion_dim: int
    technical_valid: bool


class MotionCandidate(StrictModel):
    candidate_id: str
    generation_id: str
    fingerprint: str
    motion_repr_uri: str
    smpl_global_uri: str
    smpl_incam_uri: str | None = None
    metadata_uri: str
    metadata: CandidateMetadata


class CandidateStoreRecord(StrictModel):
    candidate: MotionCandidate
    manifest_uri: str


class CandidateFailure(StrictModel):
    seed: int
    error_type: str
    message: str


class GenerationResult(StrictModel):
    generation_id: str
    status: Literal["success", "partial", "failed", "blocked"]
    candidates: list[MotionCandidate] = Field(default_factory=list)
    failed_candidates: list[CandidateFailure] = Field(default_factory=list)
    condition_fingerprint: str
    runtime_ms: float
    preflight: GenerationPreflightReport
