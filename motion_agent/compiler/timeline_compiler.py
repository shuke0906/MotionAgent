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

    if any(segment.simultaneous_with is not None for segment in segments):
        return _compile_overlapping_timeline(segments, total_duration_s, fps, total_frames)

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


def _compile_overlapping_timeline(segments, total_duration_s, fps, total_frames):
    stages = [s for s in segments if s.simultaneous_with is None or s.continuation_of is not None]
    fixed = sum(s.explicit_duration_s or 0 for s in stages)
    free = [s for s in stages if s.explicit_duration_s is None]
    if fixed > total_duration_s or (free and fixed >= total_duration_s):
        raise ValueError("explicit durations exceed available timeline")
    if not free and abs(fixed - total_duration_s) > 1e-6:
        raise ValueError("explicit durations must cover the requested timeline")
    weight = sum(s.duration_weight for s in free)
    compiled = {}
    cursor_s = 0.0
    for index, segment in enumerate(stages):
        duration = segment.explicit_duration_s or (total_duration_s - fixed) * segment.duration_weight / weight
        end_s = total_duration_s if index == len(stages) - 1 else cursor_s + duration
        current = _with_bounds(segment, cursor_s, end_s, round(cursor_s * fps), round(end_s * fps), total_frames)
        compiled[segment.segment_id] = current
        if segment.continuation_of is not None:
            parent = compiled.get(segment.continuation_of)
            if parent is None:
                raise ValueError("continuation parent must be an earlier timeline stage")
            compiled[parent.segment_id] = _with_bounds(parent, parent.start_s, end_s, parent.start_frame, current.end_frame, total_frames)
        cursor_s = end_s
    for segment in segments:
        if segment.segment_id in compiled:
            continue
        parent = compiled.get(segment.simultaneous_with)
        if parent is None:
            raise ValueError("simultaneous parent must exist")
        compiled[segment.segment_id] = _with_bounds(segment, parent.start_s, parent.end_s, parent.start_frame, parent.end_frame, total_frames)
    return [compiled[s.segment_id] for s in segments]


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
            "duration_s": round(end_s - start_s, 4),
            "start_frame": start_frame,
            "end_frame": end_frame,
            "normalized_start": round(start_frame / total_frames, 6),
            "normalized_end": round(end_frame / total_frames, 6),
        }
    )
