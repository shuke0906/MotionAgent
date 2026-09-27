"""Deterministic timeline and frame boundary compiler."""

from __future__ import annotations

from motion_agent.compiler.schemas import MotionSegment


def compile_timeline(
    segments: list[MotionSegment],
    *,
    total_duration_s: float,
    fps: int,
    total_frames: int,
) -> list[MotionSegment]:
    if not segments:
        raise ValueError("timeline requires at least one segment")

    if len(segments) == 1:
        return [_with_bounds(segments[0], 0.0, total_duration_s, 0, total_frames, total_frames)]

    explicit_durations = [segment.duration_weight for segment in segments if segment.explicit_start_s is None and segment.explicit_end_s is None]
    weight_sum = sum(explicit_durations) or float(len(segments))
    cursor_s = 0.0
    cursor_frame = 0
    compiled: list[MotionSegment] = []
    for index, segment in enumerate(segments):
        if index == len(segments) - 1:
            end_s = total_duration_s
            end_frame = total_frames
        else:
            duration_s = total_duration_s * (segment.duration_weight / weight_sum)
            end_s = min(total_duration_s, cursor_s + duration_s)
            end_frame = round(end_s * fps)
        compiled.append(_with_bounds(segment, cursor_s, end_s, cursor_frame, end_frame, total_frames))
        cursor_s = end_s
        cursor_frame = end_frame
    return compiled


def _with_bounds(
    segment: MotionSegment,
    start_s: float,
    end_s: float,
    start_frame: int,
    end_frame: int,
    total_frames: int,
) -> MotionSegment:
    return segment.model_copy(
        update={
            "start_s": round(start_s, 4),
            "end_s": round(end_s, 4),
            "start_frame": start_frame,
            "end_frame": end_frame,
            "normalized_start": round(start_frame / total_frames, 6),
            "normalized_end": round(end_frame / total_frames, 6),
        }
    )
