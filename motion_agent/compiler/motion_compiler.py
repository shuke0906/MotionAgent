"""Framework-agnostic Phase 3 Motion Compiler service."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from motion_agent.compiler.caption_optimizer import optimize_caption
from motion_agent.compiler.continuity import apply_heading_continuity
from motion_agent.compiler.control_intent import detect_control_intents
from motion_agent.compiler.gem_text_compiler import compile_gem_text_condition
from motion_agent.compiler.schemas import CompilerResult, MotionSpecification
from motion_agent.compiler.semantic_parser import parse_motion_request
from motion_agent.compiler.timeline_compiler import compile_timeline
from motion_agent.compiler.validators import (
    validate_caption_length,
    validate_gem_text_condition,
    validate_heading_continuity,
    validate_motion_semantics,
    validate_timeline,
)
from motion_agent.state.schemas import StrictModel


class CompilerRequest(StrictModel):
    original_request: str
    mode: Literal["initial", "revise"] = "initial"
    target_segments: list[int] | None = None
    focus: list[str] = Field(default_factory=list)
    duration_s: float = 6.0
    fps: int = 30
    total_frames: int = 180
    existing_motion_spec: MotionSpecification | None = None


class MotionCompiler:
    def compile(self, request: CompilerRequest) -> CompilerResult:
        if request.mode == "revise":
            segments = self._revise_segments(request)
        else:
            segments = parse_motion_request(request.original_request)

        segments = [segment.model_copy(update={"segment_id": index}) for index, segment in enumerate(segments)]
        segments = compile_timeline(
            segments,
            total_duration_s=request.duration_s,
            fps=request.fps,
            total_frames=request.total_frames,
        )
        segments = apply_heading_continuity(segments)
        hints = detect_control_intents(segments)
        captioned = [segment.model_copy(update={"gem_caption": optimize_caption(segment)}) for segment in segments]
        spec = MotionSpecification(
            original_request=request.original_request,
            duration_s=request.duration_s,
            fps=request.fps,
            total_frames=request.total_frames,
            segments=captioned,
            control_intents=hints,
        )
        condition = compile_gem_text_condition(captioned, total_frames=request.total_frames)
        validate_timeline(captioned, total_frames=request.total_frames)
        validate_caption_length(condition)
        validate_gem_text_condition(condition)
        validate_heading_continuity(captioned)
        validate_motion_semantics(spec, condition)
        changed = self._changed_segments(request, spec)
        return CompilerResult(
            motion_spec=spec,
            gem_text_condition=condition,
            routing_hints=hints,
            changed_segments=changed,
            warnings=[],
        )

    def _revise_segments(self, request: CompilerRequest):
        if request.existing_motion_spec is None:
            return parse_motion_request(request.original_request)
        if not request.target_segments:
            return request.existing_motion_spec.segments

        target_ids = set(request.target_segments)
        replacements = parse_motion_request(request.original_request)
        if len(replacements) == 1 and len(target_ids) == 1:
            target_id = next(iter(target_ids))
            replacement = replacements[0].model_copy(update={"segment_id": target_id})
            return [
                replacement if segment.segment_id == target_id else segment
                for segment in request.existing_motion_spec.segments
            ]
        return [
            replacements.pop(0).model_copy(update={"segment_id": segment.segment_id})
            if segment.segment_id in target_ids and replacements
            else segment
            for segment in request.existing_motion_spec.segments
        ]

    def _changed_segments(self, request: CompilerRequest, spec: MotionSpecification) -> list[int]:
        if request.mode == "initial" or request.existing_motion_spec is None:
            return [segment.segment_id for segment in spec.segments]
        target_ids = request.target_segments or []
        return list(target_ids)


def compile_motion(request: CompilerRequest) -> CompilerResult:
    return MotionCompiler().compile(request)
