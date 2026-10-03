"""Typed schemas for Phase 5 retrieval."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, PositiveInt, model_validator

from motion_agent.state.schemas import StrictModel


RetrievalType = Literal["caption", "motion", "pose", "trajectory"]
RetrievalPurpose = Literal["prompt_grounding", "motion_prior", "keyframe_source", "constraint_source"]
RetrievalStatus = Literal["success", "low_confidence", "empty", "error", "duplicate"]
SplitName = Literal["train", "val", "test"]


class MotionIdMap(StrictModel):
    canonical_motion_id: str
    tmr_keyid: str
    gem_mid: str | None = None


class MotionRecord(StrictModel):
    motion_id: str
    split: SplitName
    captions: list[str] = Field(default_factory=list)
    duration_s: float
    fps: int
    num_frames: int
    source: str
    smpl_handle: str | None = None
    tmr_motion_index: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class CaptionRecord(StrictModel):
    caption_id: str
    motion_id: str
    caption: str
    text_embedding_index: int
    source: str
    segment_start: float | None = None
    segment_end: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalRequest(StrictModel):
    target_segment: int
    query: str = Field(min_length=1)
    retrieval_type: RetrievalType
    purpose: RetrievalPurpose
    top_k: PositiveInt = 5
    filters: dict[str, Any] = Field(default_factory=dict)


class RetrievalQuery(StrictModel):
    text: str
    action: str | None = None
    body_parts: list[str] = Field(default_factory=list)
    direction: str | None = None
    speed: str | None = None
    style: list[str] = Field(default_factory=list)
    repetition: int | None = None
    purpose: RetrievalPurpose
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchHit(StrictModel):
    index: int
    score: float


class RetrievalCandidate(StrictModel):
    ref_id: str
    caption: str
    motion_id: str | None = None
    caption_id: str | None = None
    tmr_score: float
    duration_s: float | None = None
    split: SplitName | None = None
    source: str
    motion_handle: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RankedReference(StrictModel):
    candidate: RetrievalCandidate
    final_score: float
    component_scores: dict[str, float] = Field(default_factory=dict)
    rank: int


class RetrievedReference(StrictModel):
    ref_id: str
    source: str
    motion_id: str | None = None
    caption_id: str | None = None
    caption: str | None = None
    retrieval_score: float
    duration_s: float | None = None
    motion_handle: str | None = None
    pose_handle: str | None = None
    trajectory_handle: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalResult(StrictModel):
    request_id: str
    target_segment: int
    retrieval_type: RetrievalType
    purpose: RetrievalPurpose
    status: RetrievalStatus
    references: list[RetrievedReference] = Field(default_factory=list)
    query_used: RetrievalQuery | None = None
    retrieval_meta: dict[str, Any] = Field(default_factory=dict)


class RetrievalConfig(StrictModel):
    allowed_splits: list[SplitName] = Field(default_factory=lambda: ["train"])
    retrieve_top_n: PositiveInt = 50
    default_top_k: PositiveInt = 5
    low_confidence_threshold: float = 0.20
    corpus_version: str = "fixture_humanml3d_tiny_v1"
    tmr_model_version: str = "deterministic_keyword_encoder_v1"

    @model_validator(mode="after")
    def enforce_train_only_default(self) -> "RetrievalConfig":
        if not self.allowed_splits:
            raise ValueError("allowed_splits must not be empty")
        return self


class RetrievalIndex(StrictModel):
    motions: list[MotionRecord]
    captions: list[CaptionRecord]
    motion_embeddings: list[list[float]]
    caption_embeddings: list[list[float]]
    motion_index_to_id: dict[int, str]
    motion_id_to_index: dict[str, int]
    caption_index_to_id: dict[int, str]
    caption_id_to_index: dict[str, int]
    id_map: list[MotionIdMap] = Field(default_factory=list)
    manifest: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_index_shapes(self) -> "RetrievalIndex":
        if len(self.motion_embeddings) != len(self.motions):
            raise ValueError("motion embedding count must match motions")
        if len(self.caption_embeddings) != len(self.captions):
            raise ValueError("caption embedding count must match captions")
        return self
