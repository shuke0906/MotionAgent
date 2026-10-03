"""Resolve planner time specs into frame indices."""

from __future__ import annotations

from motion_agent.constraints.schemas import ResolvedTimeSpec, TimeSpec
from motion_agent.state.schemas import MotionAgentState, MotionSegmentSummary


class TimeResolver:
    def resolve(self, state: MotionAgentState, target_segment: int, time_spec: TimeSpec) -> ResolvedTimeSpec:
        segment = self._segment_for(state, target_segment)
        fps = state.task.fps
        total_frames = state.task.total_frames
        segment_start_frame = round(segment.start_s * fps)
        segment_end_frame = min(total_frames - 1, round(segment.end_s * fps) - 1)
        if time_spec.type == "segment":
            start_frame, end_frame = segment_start_frame, segment_end_frame
        elif time_spec.type == "point":
            if time_spec.time_s is None:
                raise ValueError("point TimeSpec requires time_s")
            frame = self._time_to_frame(time_spec.time_s, segment.start_s, fps, time_spec.segment_relative)
            start_frame = end_frame = frame
        else:
            if time_spec.start_s is None or time_spec.end_s is None:
                raise ValueError("window TimeSpec requires start_s and end_s")
            start_frame = self._time_to_frame(time_spec.start_s, segment.start_s, fps, time_spec.segment_relative)
            end_frame = self._time_to_frame(time_spec.end_s, segment.start_s, fps, time_spec.segment_relative)
        start_frame = max(segment_start_frame, min(total_frames - 1, start_frame))
        end_frame = max(segment_start_frame, min(total_frames - 1, end_frame))
        if end_frame < start_frame:
            raise ValueError("resolved time window is empty")
        if start_frame < segment_start_frame or end_frame > segment_end_frame:
            raise ValueError("resolved frames must stay inside target segment")
        return ResolvedTimeSpec(
            frame_indices=list(range(start_frame, end_frame + 1)),
            start_frame=start_frame,
            end_frame=end_frame,
        )

    def _time_to_frame(self, seconds: float, segment_start_s: float, fps: int, segment_relative: bool) -> int:
        return round((segment_start_s + seconds if segment_relative else seconds) * fps)

    def _segment_for(self, state: MotionAgentState, target_segment: int) -> MotionSegmentSummary:
        for segment in state.plan.segment_summaries:
            if segment.segment_id == target_segment:
                return segment
        return MotionSegmentSummary(
            segment_id=target_segment,
            start_s=0.0,
            end_s=state.task.total_frames / state.task.fps,
            action="unknown",
        )
