"""Minimal normal/full/text-only Phase 4 generator."""

from __future__ import annotations

import time
from typing import Protocol

import torch

from motion_agent.common.fingerprints import candidate_fingerprint
from motion_agent.common.errors import MissingArtifactError, WorkerUnavailableError
from motion_agent.constraints.store import ConditionStore
from motion_agent.generation.candidate_store import CandidateStore
from motion_agent.generation.gem_adapter import extract_candidate_tensors, prepare_pure_text_input
from motion_agent.generation.output_validator import validate_candidate_output
from motion_agent.generation.schemas import (
    CandidateFailure,
    CandidateFailureType,
    CandidateMetadata,
    GenerationConditionBundle,
    GenerationRequest,
    GenerationResult,
    SamplerMetadata,
)
from motion_agent.generation.seed_manager import seeded_torch_rng
from motion_agent.generation.segment_inpaint import build_segment_inpaint_condition
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
            "hard_motion_condition_handle": request.condition_bundle.hard_motion_condition_handle,
            "text_condition": request.text_condition.model_dump(mode="json"),
            "motion_spec": request.motion_spec.model_dump(mode="json") if request.motion_spec else None,
            "verification_specs": [spec.model_dump(mode="json") for spec in request.condition_bundle.verification_specs],
            "keyframe_specs": [spec.model_dump(mode="json") for spec in request.condition_bundle.keyframe_specs],
            "seed": seed,
            "checkpoint_version": checkpoint_version,
            "scope": request.scope,
            "target_segments": request.target_segments,
            "previous_candidate_id": request.previous_candidate_id,
            "postprocess_policy": request.postprocess_policy,
        }
    )


def _classify_failure(exc: Exception) -> tuple[CandidateFailureType, bool, bool]:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return "oom", True, True
    if isinstance(exc, TimeoutError):
        return "timeout", True, False
    if isinstance(exc, (MissingArtifactError, FileNotFoundError)):
        return "missing_artifact", False, False
    if isinstance(exc, WorkerUnavailableError):
        return "worker_unavailable", True, False
    if isinstance(exc, ValueError):
        return "invalid_output", False, False
    return "runtime_error", False, False


def _with_segment_condition(
    request: GenerationRequest,
    *,
    candidate_store: CandidateStore,
    condition_store: ConditionStore,
) -> tuple[GenerationRequest, float]:
    if request.scope != "segment":
        return request, 0.0
    handle, mask_density = build_segment_inpaint_condition(
        request=request,
        candidate_store=candidate_store,
        condition_store=condition_store,
    )
    bundle = request.condition_bundle.model_copy(update={"hard_motion_condition_handle": handle})
    patched = request.model_copy(
        update={
            "condition_bundle": bundle,
            "postprocess_policy": "constraint_safe",
        }
    )
    return patched, mask_density


def generate_motion(
    request: GenerationRequest,
    *,
    model: PredictModel,
    checkpoint_version: str,
    adapter_context: GEMAdapterContext | None = None,
    candidate_store: CandidateStore | None = None,
    condition_store: ConditionStore | None = None,
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
    condition_store = condition_store or ConditionStore()
    segment_mask_density = 0.0
    try:
        request, segment_mask_density = _with_segment_condition(
            request,
            candidate_store=store,
            condition_store=condition_store,
        )
    except Exception as exc:
        failure_type, retryable, cleanup = _classify_failure(exc)
        failures = [
            CandidateFailure(
                seed=seed,
                error_type=failure_type,
                message=str(exc),
                retryable=retryable,
                cleanup_performed=cleanup,
            )
            for seed in request.seeds
        ]
        return GenerationResult(
            generation_id=request.generation_id,
            status="failed",
            failed_candidates=failures,
            condition_fingerprint=request.condition_bundle.condition_fingerprint,
            runtime_ms=(time.perf_counter() - start) * 1000,
            preflight=preflight,
        )
    candidates = []
    failures: list[CandidateFailure] = []
    cache_hits = 0
    inference_calls = 0

    for seed in request.seeds:
        fingerprint = _fingerprint(request, seed, checkpoint_version)
        cached = store.get_by_fingerprint(fingerprint)
        if cached is not None:
            candidates.append(cached)
            cache_hits += 1
            continue

        one_start = time.perf_counter()
        try:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            data, _ = prepare_pure_text_input(
                request,
                adapter_context,
                seed=seed,
                condition_store=condition_store,
            )
            with seeded_torch_rng(seed, "cuda" if torch.cuda.is_available() else None):
                raw_pred = model.predict(
                    data,
                    static_cam=True,
                    postproc=request.postprocess_policy == "gem_default",
                    seed=seed,
                )
            inference_calls += 1
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
                generation_request=request,
            )
            record = store.save(tensors=tensors, metadata=metadata)
            candidates.append(record.candidate)
        except Exception as exc:
            failure_type, retryable, cleanup = _classify_failure(exc)
            failures.append(
                CandidateFailure(
                    seed=seed,
                    error_type=failure_type,
                    message=str(exc),
                    retryable=retryable,
                    cleanup_performed=cleanup,
                )
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
        cache_hits=cache_hits,
        inference_calls=inference_calls,
    )
