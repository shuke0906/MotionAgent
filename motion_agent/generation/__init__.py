"""Phase 4 minimal pure-text GEM generation package."""

from motion_agent.generation.normal_generator import generate_motion
from motion_agent.generation.request_builder import build_generation_request
from motion_agent.generation.schemas import (
    CandidateMetadata,
    CandidateStoreRecord,
    GenerationConditionBundle,
    GenerationPreflightReport,
    GenerationRequest,
    GenerationResult,
    MotionCandidate,
    SamplerMetadata,
)

__all__ = [
    "CandidateMetadata",
    "CandidateStoreRecord",
    "GenerationConditionBundle",
    "GenerationPreflightReport",
    "GenerationRequest",
    "GenerationResult",
    "MotionCandidate",
    "SamplerMetadata",
    "build_generation_request",
    "generate_motion",
]
