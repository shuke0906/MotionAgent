"""Compile MotionSpecification captions into GEM text windows."""

from __future__ import annotations

from typing import Any

from motion_agent.compiler.schemas import GEMTextCondition, MotionSegment


def compile_gem_text_condition(segments: list[MotionSegment], *, total_frames: int) -> GEMTextCondition:
    return GEMTextCondition(
        captions=[segment.gem_caption or segment.action for segment in segments],
        window_start=[segment.normalized_start for segment in segments],
        window_end=[segment.normalized_end for segment in segments],
        total_frames=total_frames,
    )


def build_multi_text_data(condition: GEMTextCondition) -> dict[str, Any]:
    return {
        "vid": [f"segment_{index}" for index in range(len(condition.captions))],
        "caption": list(condition.captions),
        "text_ind": list(range(len(condition.captions))),
        "window_start": list(condition.window_start),
        "window_end": list(condition.window_end),
    }
