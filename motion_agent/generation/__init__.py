"""Phase 7 GEM generation package."""

from motion_agent.generation.condition_assembler import assemble_generation_conditions
from motion_agent.generation.normal_generator import generate_motion
from motion_agent.generation.request_builder import build_generation_request
from motion_agent.generation.segment_inpaint import build_segment_inpaint_condition
from motion_agent.generation.schemas import (
    CandidateMetadata,
    CandidateStoreRecord,
    GPUGenerationJob,
    GenerationConditionBundle,
    GenerationPreflightReport,
    GenerationRequest,
    GenerationResult,
    MotionCandidate,
    SamplerMetadata,
    WorkerHealth,
)
from motion_agent.generation.worker import GEMGenerationWorker, InProcessGenerationQueue

__all__ = [
    "CandidateMetadata",
    "CandidateStoreRecord",
    "GEMGenerationWorker",
    "GPUGenerationJob",
    "GenerationConditionBundle",
    "GenerationPreflightReport",
    "GenerationRequest",
    "GenerationResult",
    "InProcessGenerationQueue",
    "MotionCandidate",
    "SamplerMetadata",
    "WorkerHealth",
    "assemble_generation_conditions",
    "build_generation_request",
    "build_segment_inpaint_condition",
    "generate_motion",
]
