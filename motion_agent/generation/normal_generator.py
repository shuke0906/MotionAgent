"""Minimal normal/full/text-only Phase 4 generator."""

from __future__ import annotations

import time
from typing import Protocol

import torch

from motion_agent.common.fingerprints import candidate_fingerprint
from motion_agent.generation.candidate_store import CandidateStore
from motion_agent.generation.gem_adapter import extract_candidate_tensors, prepare_pure_text_input
from motion_agent.generation.output_validator import validate_candidate_output
from motion_agent.generation.schemas import (
    CandidateFailure,
    CandidateMetadata,
    GenerationRequest,
    GenerationResult,
    SamplerMetadata,
)
from motion_agent.generation.seed_manager import seeded_torch_rng
from motion_agent.generation.validators import validate_generation_request
from motion_agent.state.schemas import GEMAdapterContext


class PredictModel(Protocol):
    def predict(self, data: dict, static_cam: bool = True, postproc: bool = True, seed: int | None = None) -> dict:
        ...


def _peak_memory_mb() -> float | None:
    if not torch.cuda.is_available():
        return None
    return float(torch.cuda.max_memory_allocated() / (1024 * 1024))


def _fingerprint(request: GenerationRequest, seed: int, checkpoint_version: str) -> str:
    return candidate_fingerprint(
        {
            "generation_id": request.generation_id,
            "condition_fingerprint": request.condition_bundle.condition_fingerprint,
            "text_condition": request.text_condition.model_dump(mode="json"),
            "seed": seed,
            "checkpoint_version": checkpoint_version,
            "scope": request.scope,
            "postprocess_policy": request.postprocess_policy,
        }
    )


def generate_motion(
    request: GenerationRequest,
    *,
    model: PredictModel,
    checkpoint_version: str,
    adapter_context: GEMAdapterContext | None = None,
    candidate_store: CandidateStore | None = None,
) -> GenerationResult:
    start = time.perf_counter()
    preflight = validate_generation_request(request)
    if preflight.status != "ready":
        return GenerationResult(
            generation_id=request.generation_id,
            status="blocked",
            condition_fingerprint=request.condition_bundle.condition_fingerprint,
            runtime_ms=(time.perf_counter() - start) * 1000,
            preflight=preflight,
        )

    store = candidate_store or CandidateStore(request.output_policy.artifact_root)
    candidates = []
    failures: list[CandidateFailure] = []

    for seed in request.seeds:
        fingerprint = _fingerprint(request, seed, checkpoint_version)
        cached = store.get_by_fingerprint(fingerprint)
        if cached is not None:
            candidates.append(cached)
            continue

        one_start = time.perf_counter()
        try:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            data, _ = prepare_pure_text_input(request, adapter_context, seed=seed)
            with seeded_torch_rng(seed, "cuda" if torch.cuda.is_available() else None):
                raw_pred = model.predict(
                    data,
                    static_cam=True,
                    postproc=request.postprocess_policy != "none",
                    seed=seed,
                )
            tensors = extract_candidate_tensors(raw_pred)
            validate_candidate_output(tensors, expected_frames=request.total_frames)
            runtime_ms = (time.perf_counter() - one_start) * 1000
            motion = tensors["motion_repr"]
            metadata = CandidateMetadata(
                seed=seed,
                generation_id=request.generation_id,
                condition_fingerprint=request.condition_bundle.condition_fingerprint,
                candidate_fingerprint=fingerprint,
                checkpoint_version=checkpoint_version,
                sampler=SamplerMetadata(
                    seed=seed,
                    torch_version=torch.__version__,
                    cuda_version=torch.version.cuda,
                    checkpoint_version=checkpoint_version,
                ),
                scope=request.scope,
                target_segments=request.target_segments,
                postprocess_policy=request.postprocess_policy,
                active_constraint_ids=request.condition_bundle.active_constraint_ids,
                active_keyframe_ids=request.condition_bundle.active_keyframe_ids,
                active_reference_ids=request.condition_bundle.active_reference_ids,
                runtime_ms=runtime_ms,
                peak_gpu_memory_mb=_peak_memory_mb(),
                frame_count=int(motion.shape[0]),
                motion_dim=int(motion.shape[-1]),
                technical_valid=True,
            )
            record = store.save(tensors=tensors, metadata=metadata)
            candidates.append(record.candidate)
        except Exception as exc:
            failures.append(
                CandidateFailure(seed=seed, error_type=exc.__class__.__name__, message=str(exc))
            )

    if len(candidates) == request.num_candidates:
        status = "success"
    elif candidates:
        status = "partial"
    else:
        status = "failed"

    return GenerationResult(
        generation_id=request.generation_id,
        status=status,
        candidates=candidates,
        failed_candidates=failures,
        condition_fingerprint=request.condition_bundle.condition_fingerprint,
        runtime_ms=(time.perf_counter() - start) * 1000,
        preflight=preflight,
    )
