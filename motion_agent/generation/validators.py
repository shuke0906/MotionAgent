"""Preflight validation for Phase 4 GenerationRequest."""

from __future__ import annotations

from motion_agent.generation.schemas import GenerationPreflightReport, GenerationRequest


def validate_generation_request(
    request: GenerationRequest,
    *,
    max_request_frames: int = 600,
    guided_generation_available: bool = False,
) -> GenerationPreflightReport:
    errors: list[str] = []
    warnings: list[str] = []
    condition = request.text_condition

    if request.strategy == "guided" and not guided_generation_available:
        errors.append("guided_generation_not_available")
    if request.scope == "segment" and not request.previous_candidate_id:
        errors.append("segment_scope_requires_previous_candidate")
    if request.scope == "segment" and not request.target_segments:
        errors.append("segment_scope_requires_target_segments")
    if request.scope == "full" and request.target_segments:
        warnings.append("target_segments_ignored_for_full_generation")
    if request.total_frames > max_request_frames:
        errors.append("request_exceeds_max_request_frames")
    if len(condition.captions) == 0:
        errors.append("gem_text_condition_missing_captions")
    if not (len(condition.captions) == len(condition.window_start) == len(condition.window_end)):
        errors.append("caption_window_count_mismatch")
    for start, end in zip(condition.window_start, condition.window_end):
        if not (0 <= start < end <= 1):
            errors.append("invalid_temporal_window")
            break
    if request.target_segments:
        for segment_id in request.target_segments:
            try:
                condition.bounds_for_segment(segment_id)
            except ValueError:
                errors.append("invalid_target_segment")
                break
    if (
        request.condition_bundle.hard_motion_condition_handle
        or request.condition_bundle.active_keyframe_ids
        or request.scope == "segment"
    ) and request.postprocess_policy == "gem_default":
        errors.append("constraint_safe_postprocess_required")
    if len(request.seeds) != len(set(request.seeds)):
        warnings.append("duplicate_seeds")

    return GenerationPreflightReport(
        status="blocked" if errors else "ready",
        errors=errors,
        warnings=warnings,
        estimated_frames=request.total_frames,
        condition_fingerprint=request.condition_bundle.condition_fingerprint,
    )
