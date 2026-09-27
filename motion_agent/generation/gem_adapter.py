"""Adapter between MotionAgent Phase 4 schemas and GEM pure-text inference."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch

from motion_agent.compiler.gem_text_compiler import build_multi_text_data
from motion_agent.generation.schemas import GenerationRequest
from motion_agent.state.schemas import GEMAdapterContext


REPO_ROOT = Path(__file__).resolve().parents[2]
GENMO_ROOT = REPO_ROOT / "vendor" / "GENMO"
if GENMO_ROOT.exists() and str(GENMO_ROOT) not in sys.path:
    sys.path.insert(0, str(GENMO_ROOT))

from gem.motionagent import (  # noqa: E402
    MotionAgentCameraContext,
    MotionAgentTextSegment,
    build_motionagent_text_data,
)


def _segment_lengths(request: GenerationRequest) -> list[int]:
    starts = [round(value * request.total_frames) for value in request.text_condition.window_start]
    ends = [round(value * request.total_frames) for value in request.text_condition.window_end]
    if starts:
        starts[0] = 0
    if ends:
        ends[-1] = request.total_frames
    lengths = [end - start for start, end in zip(starts, ends)]
    if any(length <= 0 for length in lengths) or sum(lengths) != request.total_frames:
        raise ValueError("GEMTextCondition windows do not resolve to valid frame lengths")
    return lengths


def prepare_pure_text_input(
    request: GenerationRequest,
    context: GEMAdapterContext | None = None,
    *,
    seed: int | None = None,
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
    lengths = _segment_lengths(request)
    segments = [
        MotionAgentTextSegment(caption=caption, length=length, name=f"segment_{index}")
        for index, (caption, length) in enumerate(zip(request.text_condition.captions, lengths))
    ]
    data, segment_info = build_motionagent_text_data(segments, camera, seed=seed)
    multi_text = build_multi_text_data(request.text_condition)
    multi_text["window_start"] = torch.tensor(multi_text["window_start"], dtype=torch.float32)
    multi_text["window_end"] = torch.tensor(multi_text["window_end"], dtype=torch.float32)
    data["meta"][0]["multi_text_data"] = multi_text
    return data, segment_info


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
