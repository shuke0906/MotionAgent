"""Adapter between MotionAgent Phase 4 schemas and GEM pure-text inference."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch

from motion_agent.compiler.gem_text_compiler import build_multi_text_data
from motion_agent.constraints.store import ConditionStore
from motion_agent.generation.schemas import GenerationRequest
from motion_agent.state.schemas import GEMAdapterContext


REPO_ROOT = Path(__file__).resolve().parents[2]
GENMO_ROOT = REPO_ROOT / "vendor" / "GENMO"
# The lightweight payload builder is shipped even when external GENMO is absent.
ADAPTER_ROOT = GENMO_ROOT if (GENMO_ROOT / "gem/motionagent.py").is_file() else REPO_ROOT / "integrations/GENMO"
if str(ADAPTER_ROOT) not in sys.path:
    sys.path.insert(0, str(ADAPTER_ROOT))

from gem.motionagent import (  # noqa: E402
    MotionAgentCameraContext,
    MotionAgentTextSegment,
    build_motionagent_text_data,
)


def _segment_bounds(request: GenerationRequest) -> list[tuple[int, int]]:
    starts = [round(value * request.total_frames) for value in request.text_condition.window_start]
    ends = [round(value * request.total_frames) for value in request.text_condition.window_end]
    bounds = list(zip(starts, ends))
    if not bounds or any(not 0 <= start < end <= request.total_frames for start, end in bounds):
        raise ValueError("GEMTextCondition windows do not resolve to valid frame lengths")
    cursor = 0
    for start, end in sorted(bounds):
        if start > cursor:
            raise ValueError("GEMTextCondition windows must cover the timeline")
        cursor = max(cursor, end)
    if cursor != request.total_frames:
        raise ValueError("GEMTextCondition windows must cover the timeline")
    return bounds


def prepare_pure_text_input(
    request: GenerationRequest,
    context: GEMAdapterContext | None = None,
    *,
    seed: int | None = None,
    condition_store: ConditionStore | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    context = context or GEMAdapterContext(
        fps=request.fps,
        total_frames=request.total_frames,
    )
    camera = MotionAgentCameraContext(
        width=context.width,
        height=context.height,
        static_camera=context.static_camera,
    )
    bounds = _segment_bounds(request)
    # Camera/image placeholders span the full sequence; text windows may overlap.
    carrier = MotionAgentTextSegment(caption=request.text_condition.captions[0], length=request.total_frames)
    data, _ = build_motionagent_text_data([carrier], camera, seed=seed)
    segment_info = [
        {"start": start, "end": end, "type": "text", "caption": caption, "name": f"segment_{index}"}
        for index, (caption, (start, end)) in enumerate(zip(request.text_condition.captions, bounds))
    ]
    multi_text = build_multi_text_data(request.text_condition)
    multi_text["window_start"] = torch.tensor(multi_text["window_start"], dtype=torch.float32)
    multi_text["window_end"] = torch.tensor(multi_text["window_end"], dtype=torch.float32)
    data["meta"][0]["multi_text_data"] = multi_text
    data["meta"][0]["segment_info"] = segment_info
    data["meta"][0]["condition_bundle"] = request.condition_bundle.model_dump(mode="json")
    if request.motion_spec is not None:
        data["meta"][0]["motion_specification"] = request.motion_spec.model_dump(mode="json")
    attach_hard_motion_condition(data, request, condition_store=condition_store)
    return data, segment_info


def attach_hard_motion_condition(
    data: dict[str, Any],
    request: GenerationRequest,
    *,
    condition_store: ConditionStore | None = None,
) -> None:
    handle = request.condition_bundle.hard_motion_condition_handle
    if not handle:
        return
    condition = (condition_store or ConditionStore()).load_hard_condition(handle)
    if tuple(condition.values.shape) != (request.total_frames, 151):
        raise ValueError("hard-condition tensors must be [total_frames, 151]")
    if not torch.isfinite(condition.values).all() or not torch.isfinite(condition.mask).all():
        raise ValueError("hard-condition values and mask must be finite")
    if not ((condition.mask == 0) | (condition.mask == 1)).all():
        raise ValueError("hard-condition mask must be binary")
    data["observed_motion_3d"] = condition.values
    data["motion_mask_3d"] = condition.mask
    data["rm_text_flag"] = torch.zeros((), dtype=torch.bool)


def extract_candidate_tensors(raw_pred: dict[str, Any]) -> dict[str, Any]:
    try:
        motion_repr = raw_pred["net_outputs"]["model_output"]["pred_x"]
    except KeyError as exc:
        raise ValueError("GEM output missing net_outputs.model_output.pred_x") from exc
    if motion_repr.ndim == 3 and motion_repr.shape[0] == 1:
        motion_repr = motion_repr[0]
    return {
        "motion_repr": motion_repr.detach().cpu(),
        "body_params_global": {
            key: value.detach().cpu() for key, value in raw_pred["body_params_global"].items()
        },
        "body_params_incam": {
            key: value.detach().cpu() for key, value in raw_pred.get("body_params_incam", {}).items()
        }
        or None,
    }
