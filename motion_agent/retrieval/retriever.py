"""Framework-agnostic retrieval service."""

from __future__ import annotations

import time
from typing import Any

from motion_agent.common.fingerprints import stable_fingerprint
from motion_agent.common.ids import id_from_fingerprint
from motion_agent.retrieval.filters import CandidateFilter
from motion_agent.retrieval.query_builder import build_retrieval_query
from motion_agent.retrieval.reranker import StructuredReranker
from motion_agent.retrieval.schemas import (
    CaptionRecord,
    MotionIdMap,
    MotionRecord,
    RankedReference,
    RetrievalCandidate,
    RetrievalConfig,
    RetrievalIndex,
    RetrievalRequest,
    RetrievalResult,
    RetrievedReference,
)
from motion_agent.retrieval.search_backend import MatrixCosineSearchBackend
from motion_agent.retrieval.signatures import make_retrieval_signature, normalize_query_text
from motion_agent.retrieval.tmr_encoder import DeterministicTextEncoder
from motion_agent.retrieval.view_extractor import pose_handle_for, trajectory_handle_for
from motion_agent.state.schemas import MotionAgentState


class RetrievalService:
    def __init__(
        self,
        *,
        index: RetrievalIndex,
        config: RetrievalConfig | None = None,
        encoder: DeterministicTextEncoder | None = None,
    ) -> None:
        self.index = index
        self.config = config or RetrievalConfig()
        self.encoder = encoder or DeterministicTextEncoder()
        self.motion_search = MatrixCosineSearchBackend(index.motion_embeddings)
        self.caption_search = MatrixCosineSearchBackend(index.caption_embeddings)
        self.filter = CandidateFilter(allowed_splits=self.config.allowed_splits)
        self.reranker = StructuredReranker()

    def retrieve_reference(self, state: MotionAgentState, request: RetrievalRequest) -> RetrievalResult:
        started = time.perf_counter()
        try:
            segment = self._get_segment(state, request.target_segment)
            query = build_retrieval_query(segment, request.query, request.purpose)
            signature = make_retrieval_signature(
                target_segment=request.target_segment,
                normalized_query=query.text,
                retrieval_type=request.retrieval_type,
                purpose=request.purpose,
                corpus_version=self.config.corpus_version,
            )
            duplicate = self._find_duplicate(state, signature)
            if duplicate is not None:
                return RetrievalResult(
                    request_id=signature,
                    target_segment=request.target_segment,
                    retrieval_type=request.retrieval_type,
                    purpose=request.purpose,
                    status="duplicate",
                    references=[],
                    query_used=query,
                    retrieval_meta={
                        "duplicate_ref_id": duplicate,
                        "retrieval_signature": signature,
                        "corpus_version": self.config.corpus_version,
                    },
            )

            query_embedding = self.encoder.encode_text(query.text)
            if not any(abs(value) > 1e-12 for value in query_embedding):
                return RetrievalResult(
                    request_id=signature,
                    target_segment=request.target_segment,
                    retrieval_type=request.retrieval_type,
                    purpose=request.purpose,
                    status="low_confidence",
                    references=[],
                    query_used=query,
                    retrieval_meta={
                        "retrieval_signature": signature,
                        "corpus_version": self.config.corpus_version,
                        "tmr_model_version": self.config.tmr_model_version,
                        "allowed_splits": self.config.allowed_splits,
                        "top_n": 0,
                        "top_k": request.top_k,
                        "candidate_count": 0,
                        "filtered_count": 0,
                        "latency_ms": (time.perf_counter() - started) * 1000.0,
                        "index_kind": self.index.manifest.get("index_kind", "fixture"),
                        "low_confidence_reason": "query_embedding_zero",
                    },
                )
            top_n = int(request.filters.get("retrieve_top_n", self.config.retrieve_top_n))
            hits = self._search(request.retrieval_type, query_embedding, top_n=top_n)
            candidates = self._to_candidates(request.retrieval_type, hits)
            filtered = self.filter.apply(candidates, query=query, target_duration_s=self._segment_duration(segment))
            ranked = self.reranker.rerank(
                filtered,
                query=query,
                target_duration_s=self._segment_duration(segment),
                top_k=request.top_k or self.config.default_top_k,
            )
            references = [
                self._to_reference(item, retrieval_type=request.retrieval_type, query_text=query.text)
                for item in ranked
            ]
            status = self._status(references)
            latency_ms = (time.perf_counter() - started) * 1000.0
            return RetrievalResult(
                request_id=signature,
                target_segment=request.target_segment,
                retrieval_type=request.retrieval_type,
                purpose=request.purpose,
                status=status,
                references=references,
                query_used=query,
                retrieval_meta={
                    "retrieval_signature": signature,
                    "corpus_version": self.config.corpus_version,
                    "tmr_model_version": self.config.tmr_model_version,
                    "allowed_splits": self.config.allowed_splits,
                    "top_n": top_n,
                    "top_k": request.top_k,
                    "candidate_count": len(candidates),
                    "filtered_count": len(filtered),
                    "latency_ms": latency_ms,
                    "index_kind": self.index.manifest.get("index_kind", "fixture"),
                },
            )
        except Exception as exc:
            return RetrievalResult(
                request_id="retrieval_error",
                target_segment=request.target_segment,
                retrieval_type=request.retrieval_type,
                purpose=request.purpose,
                status="error",
                references=[],
                query_used=None,
                retrieval_meta={"error": str(exc)},
            )

    def _get_segment(self, state: MotionAgentState, target_segment: int) -> Any:
        for segment in state.plan.segment_summaries:
            if segment.segment_id == target_segment:
                return segment
        raise ValueError(f"target_segment does not exist in motion plan: {target_segment}")

    def _segment_duration(self, segment: Any) -> float | None:
        start = getattr(segment, "start_s", None)
        end = getattr(segment, "end_s", None)
        if start is None or end is None:
            return None
        return float(end - start)

    def _find_duplicate(self, state: MotionAgentState, signature: str) -> str | None:
        for retrieval in state.conditions.retrievals:
            if retrieval.status == "active" and retrieval.retrieval_signature == signature:
                return retrieval.ref_id
        return None

    def _search(self, retrieval_type: str, query_embedding: list[float], *, top_n: int):
        if retrieval_type == "caption":
            return self.caption_search.search(query_embedding, top_n)
        return self.motion_search.search(query_embedding, top_n)

    def _to_candidates(self, retrieval_type: str, hits) -> list[RetrievalCandidate]:
        candidates: list[RetrievalCandidate] = []
        motions_by_id = {motion.motion_id: motion for motion in self.index.motions}
        for hit in hits:
            if retrieval_type == "caption":
                caption = self.index.captions[hit.index]
                motion = motions_by_id.get(caption.motion_id)
                candidates.append(_candidate_from_caption(caption, motion, hit.score))
            else:
                motion = self.index.motions[hit.index]
                candidates.append(_candidate_from_motion(motion, hit.score))
        return candidates

    def _to_reference(self, ranked: RankedReference, *, retrieval_type: str, query_text: str) -> RetrievedReference:
        candidate = ranked.candidate
        motion_handle = candidate.motion_handle if retrieval_type in {"motion", "pose", "trajectory"} else None
        pose_handle = pose_handle_for(ranked, query_text=query_text) if retrieval_type == "pose" else None
        trajectory_handle = (
            trajectory_handle_for(ranked, query_text=query_text) if retrieval_type == "trajectory" else None
        )
        return RetrievedReference(
            ref_id=id_from_fingerprint(
                "reference",
                stable_fingerprint(
                    {
                        "candidate_ref": candidate.ref_id,
                        "retrieval_type": retrieval_type,
                        "rank": ranked.rank,
                        "score": round(ranked.final_score, 6),
                    },
                    namespace="retrieved_reference",
                ),
            ),
            source=candidate.source,
            motion_id=candidate.motion_id,
            caption_id=candidate.caption_id,
            caption=candidate.caption,
            retrieval_score=ranked.final_score,
            duration_s=candidate.duration_s,
            motion_handle=motion_handle,
            pose_handle=pose_handle,
            trajectory_handle=trajectory_handle,
            metadata={
                **candidate.metadata,
                "rank": ranked.rank,
                "tmr_score": candidate.tmr_score,
                "component_scores": ranked.component_scores,
                "retrieval_type": retrieval_type,
            },
        )

    def _status(self, references: list[RetrievedReference]) -> str:
        if not references:
            return "empty"
        best = max(reference.retrieval_score for reference in references)
        if best < self.config.low_confidence_threshold:
            return "low_confidence"
        return "success"


def _candidate_from_caption(caption: CaptionRecord, motion: MotionRecord | None, score: float) -> RetrievalCandidate:
    duration_s = motion.duration_s if motion else None
    split = motion.split if motion else caption.metadata.get("split")
    motion_handle = motion.smpl_handle if motion else None
    metadata: dict[str, Any] = {"caption_record": caption.model_dump(mode="json")}
    if motion:
        metadata.update(motion.metadata)
    return RetrievalCandidate(
        ref_id=caption.caption_id,
        caption=caption.caption,
        motion_id=caption.motion_id,
        caption_id=caption.caption_id,
        tmr_score=score,
        duration_s=duration_s,
        split=split,
        source=caption.source,
        motion_handle=motion_handle,
        metadata=metadata,
    )


def _candidate_from_motion(motion: MotionRecord, score: float) -> RetrievalCandidate:
    caption = motion.captions[0] if motion.captions else motion.motion_id
    return RetrievalCandidate(
        ref_id=motion.motion_id,
        caption=caption,
        motion_id=motion.motion_id,
        caption_id=None,
        tmr_score=score,
        duration_s=motion.duration_s,
        split=motion.split,
        source=motion.source,
        motion_handle=motion.smpl_handle,
        metadata=motion.metadata,
    )


def build_default_retrieval_service() -> RetrievalService:
    return RetrievalService(index=build_fixture_index())


def build_fixture_index() -> RetrievalIndex:
    encoder = DeterministicTextEncoder()
    motions = [
        MotionRecord(
            motion_id="hml_train_walk_forward_limp",
            split="train",
            captions=[
                "a person walks forward with an asymmetric limping gait",
                "a person moves forward with an uneven injured gait",
            ],
            duration_s=4.0,
            fps=20,
            num_frames=80,
            source="HumanML3D-fixture",
            smpl_handle="gem_smpl:hml_train_walk_forward_limp",
            tmr_motion_index=0,
            metadata={"action": "walk", "style": ["limping", "asymmetric"], "direction": "forward"},
        ),
        MotionRecord(
            motion_id="hml_train_sit_down",
            split="train",
            captions=["a person sits down into a stable seated pose"],
            duration_s=3.0,
            fps=20,
            num_frames=60,
            source="HumanML3D-fixture",
            smpl_handle="gem_smpl:hml_train_sit_down",
            tmr_motion_index=1,
            metadata={"action": "sit_down"},
        ),
        MotionRecord(
            motion_id="hml_train_wave_right_hand",
            split="train",
            captions=["a person waves the right hand several times"],
            duration_s=2.0,
            fps=20,
            num_frames=40,
            source="HumanML3D-fixture",
            smpl_handle="gem_smpl:hml_train_wave_right_hand",
            tmr_motion_index=2,
            metadata={"action": "wave", "body_parts": ["right_hand"]},
        ),
        MotionRecord(
            motion_id="hml_train_walk_backward",
            split="train",
            captions=["a person walks backward carefully"],
            duration_s=4.5,
            fps=20,
            num_frames=90,
            source="HumanML3D-fixture",
            smpl_handle="gem_smpl:hml_train_walk_backward",
            tmr_motion_index=3,
            metadata={"action": "walk", "direction": "backward"},
        ),
        MotionRecord(
            motion_id="hml_test_jump",
            split="test",
            captions=["a person jumps upward"],
            duration_s=1.5,
            fps=20,
            num_frames=30,
            source="HumanML3D-fixture",
            smpl_handle="gem_smpl:hml_test_jump",
            tmr_motion_index=4,
            metadata={"action": "jump"},
        ),
    ]
    captions: list[CaptionRecord] = []
    for motion in motions:
        for index, caption in enumerate(motion.captions):
            captions.append(
                CaptionRecord(
                    caption_id=f"{motion.motion_id}:cap{index}",
                    motion_id=motion.motion_id,
                    caption=caption,
                    text_embedding_index=len(captions),
                    source=motion.source,
                    metadata={"split": motion.split},
                )
            )
    return RetrievalIndex(
        motions=motions,
        captions=captions,
        motion_embeddings=[encoder.encode_motion_caption(motion.captions) for motion in motions],
        caption_embeddings=[encoder.encode_text(caption.caption) for caption in captions],
        motion_index_to_id={index: motion.motion_id for index, motion in enumerate(motions)},
        motion_id_to_index={motion.motion_id: index for index, motion in enumerate(motions)},
        caption_index_to_id={index: caption.caption_id for index, caption in enumerate(captions)},
        caption_id_to_index={caption.caption_id: index for index, caption in enumerate(captions)},
        id_map=[
            MotionIdMap(canonical_motion_id=motion.motion_id, tmr_keyid=motion.motion_id, gem_mid=motion.motion_id)
            for motion in motions
        ]
        + [MotionIdMap(canonical_motion_id="hml_unmapped_caption_only", tmr_keyid="hml_unmapped_caption_only")],
        manifest={
            "corpus_version": "fixture_humanml3d_tiny_v1",
            "tmr_model_version": "deterministic_keyword_encoder_v1",
            "split_policy": {"allowed_splits": ["train"]},
            "index_kind": "fixture",
        },
    )
