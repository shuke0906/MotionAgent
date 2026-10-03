"""Derived handle builders for pose and trajectory retrieval views."""

from __future__ import annotations

from motion_agent.common.fingerprints import stable_fingerprint
from motion_agent.retrieval.schemas import RankedReference


def pose_handle_for(reference: RankedReference, *, query_text: str) -> str:
    key = stable_fingerprint(
        {
            "kind": "pose_view",
            "motion_id": reference.candidate.motion_id,
            "rank": reference.rank,
            "query": query_text,
        },
        namespace="retrieval_view",
    )
    return f"pose_view:{key[:16]}"


def trajectory_handle_for(reference: RankedReference, *, query_text: str) -> str:
    key = stable_fingerprint(
        {
            "kind": "trajectory_view",
            "motion_id": reference.candidate.motion_id,
            "rank": reference.rank,
            "query": query_text,
        },
        namespace="retrieval_view",
    )
    return f"trajectory_view:{key[:16]}"
