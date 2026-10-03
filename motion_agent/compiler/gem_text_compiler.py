"""Compile MotionSpecification captions into GEM text windows."""

from __future__ import annotations

from typing import Any

from motion_agent.compiler.caption_optimizer import compose_concurrent_caption
from motion_agent.compiler.schemas import GEMTextCondition, MotionSegment


def compile_gem_text_condition(segments: list[MotionSegment], *, total_frames: int) -> GEMTextCondition:
    boundaries = sorted({value for segment in segments for value in (segment.start_frame, segment.end_frame)})
    captions, starts, ends = [], [], []
    for start, end in zip(boundaries, boundaries[1:]):
        active = [segment for segment in segments if segment.start_frame <= start and end <= segment.end_frame]
        if not active:
            raise ValueError("compiled text windows must cover the timeline")
        captions.append(compose_concurrent_caption(active))
        starts.append(round(start / total_frames, 6))
        ends.append(round(end / total_frames, 6))
    return GEMTextCondition(
        captions=captions,
        window_start=starts,
        window_end=ends,
        total_frames=total_frames,
        segment_bounds={segment.segment_id: (segment.start_frame, segment.end_frame) for segment in segments},
    )


def build_multi_text_data(condition: GEMTextCondition) -> dict[str, Any]:
    return {
        "vid": [f"segment_{index}" for index in range(len(condition.captions))],
        "caption": list(condition.captions),
        "text_ind": list(range(len(condition.captions))),
        "window_start": list(condition.window_start),
        "window_end": list(condition.window_end),
    }
