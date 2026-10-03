"""Metadata filters for retrieval candidates."""

from __future__ import annotations

from motion_agent.retrieval.schemas import RetrievalCandidate, RetrievalQuery


class CandidateFilter:
    def __init__(self, *, allowed_splits: list[str]) -> None:
        self.allowed_splits = set(allowed_splits)

    def apply(
        self,
        candidates: list[RetrievalCandidate],
        *,
        query: RetrievalQuery,
        target_duration_s: float | None,
    ) -> list[RetrievalCandidate]:
        kept: list[RetrievalCandidate] = []
        seen_motion_ids: set[str] = set()
        for candidate in candidates:
            if candidate.split not in self.allowed_splits:
                continue
            if candidate.metadata.get("invalid_motion"):
                continue
            if candidate.motion_id and candidate.motion_id in seen_motion_ids:
                continue
            if not self._duration_ok(candidate, query=query, target_duration_s=target_duration_s):
                continue
            if candidate.motion_id:
                seen_motion_ids.add(candidate.motion_id)
            kept.append(candidate)
        return kept

    def _duration_ok(
        self,
        candidate: RetrievalCandidate,
        *,
        query: RetrievalQuery,
        target_duration_s: float | None,
    ) -> bool:
        if candidate.duration_s is None or target_duration_s is None:
            return True
        if query.purpose == "prompt_grounding":
            return True
        ratio = candidate.duration_s / max(target_duration_s, 1e-6)
        if query.purpose in {"constraint_source"}:
            return 0.25 <= ratio <= 4.0
        if query.purpose == "motion_prior":
            return 0.20 <= ratio <= 5.0
        return 0.10 <= ratio <= 8.0
