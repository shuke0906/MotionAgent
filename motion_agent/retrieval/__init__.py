"""Phase 5 retrieval tool domain package."""

from motion_agent.retrieval.retriever import RetrievalService, build_default_retrieval_service
from motion_agent.retrieval.schemas import (
    CaptionRecord,
    MotionIdMap,
    MotionRecord,
    RetrievalRequest,
    RetrievalResult,
    RetrievedReference,
)

__all__ = [
    "CaptionRecord",
    "MotionIdMap",
    "MotionRecord",
    "RetrievalRequest",
    "RetrievalResult",
    "RetrievedReference",
    "RetrievalService",
    "build_default_retrieval_service",
]
