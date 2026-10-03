"""Build full-sequence preservation inputs; true inpainting is deferred.

The mask is forwarded through the existing GEM interface. It does not enforce
outside-segment preservation without future generation-time conditioning.
"""

from __future__ import annotations

import torch

from motion_agent.common.fingerprints import condition_fingerprint
from motion_agent.constraints.composer import compose_conditions
from motion_agent.constraints.gem_features import GEMFeatureMapper
from motion_agent.constraints.schemas import HardMotionCondition
from motion_agent.constraints.store import ConditionStore
from motion_agent.generation.candidate_store import CandidateStore
from motion_agent.generation.schemas import GenerationRequest


def segment_frame_ranges(request: GenerationRequest) -> list[tuple[int, int]]:
    if request.target_segments is None:
        raise ValueError("segment regeneration requires target_segments")
    ranges: list[tuple[int, int]] = []
    for segment_id in request.target_segments:
        start, end = request.text_condition.bounds_for_segment(segment_id)
        if start < 0 or end > request.total_frames or start >= end:
            raise ValueError(f"target segment {segment_id} resolves to invalid frame range")
        ranges.append((start, end))
    return ranges


def build_preservation_condition(
    *,
    request: GenerationRequest,
    candidate_store: CandidateStore,
) -> HardMotionCondition:
    if request.previous_candidate_id is None:
        raise ValueError("segment regeneration requires previous_candidate_id")
    loaded = candidate_store.load(request.previous_candidate_id)
    previous_motion = loaded["motion_repr"].detach().cpu().float()
    if tuple(previous_motion.shape) != (request.total_frames, 151):
        raise ValueError("previous candidate motion_repr shape does not match request")

    mask = torch.ones_like(previous_motion)
    for start, end in segment_frame_ranges(request):
        mask[start:end, :] = 0.0

    mapper = GEMFeatureMapper()
    beta_start, beta_end = mapper.get_group_slice("betas")
    mask[:, beta_start:beta_end] = 1.0

    fingerprint = condition_fingerprint(
        {
            "previous_candidate_id": request.previous_candidate_id,
            "target_segments": request.target_segments,
            "total_frames": request.total_frames,
            "motion_shape": list(previous_motion.shape),
        }
    )
    return HardMotionCondition(
        values=previous_motion,
        mask=mask,
        source_constraint_id=f"segment_inpaint_{fingerprint[:12]}",
        metadata={
            "type": "segment_inpaint_preservation",
            "previous_candidate_id": request.previous_candidate_id,
            "target_segments": request.target_segments or [],
            "preserved_frame_ranges": _preserved_ranges(request.total_frames, segment_frame_ranges(request)),
        },
    )


def build_segment_inpaint_condition(
    *,
    request: GenerationRequest,
    candidate_store: CandidateStore,
    condition_store: ConditionStore | None = None,
) -> tuple[str, float]:
    """Create a merged full-sequence inpaint condition and return its handle/density."""

    condition_store = condition_store or ConditionStore()
    preservation = build_preservation_condition(request=request, candidate_store=candidate_store)
    conditions = [preservation]
    if request.condition_bundle.hard_motion_condition_handle:
        conditions.append(condition_store.load_hard_condition(request.condition_bundle.hard_motion_condition_handle))
    bundle = compose_conditions(
        conditions,
        total_frames=request.total_frames,
        store=condition_store,
        active_constraint_ids=request.condition_bundle.active_constraint_ids,
        active_keyframe_ids=request.condition_bundle.active_keyframe_ids,
    )
    if bundle.generation_blocked or bundle.hard_condition_handle is None:
        raise ValueError("segment inpaint hard conditions conflict or produced no condition")
    return bundle.hard_condition_handle, bundle.mask_density


def _preserved_ranges(total_frames: int, target_ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    cursor = 0
    for start, end in sorted(target_ranges):
        if cursor < start:
            ranges.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < total_frames:
        ranges.append((cursor, total_frames))
    return ranges
