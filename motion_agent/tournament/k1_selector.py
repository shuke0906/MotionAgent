"""K=1 tournament bypass selector for current V1 execution mode."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from motion_agent.generation.schemas import GenerationResult, MotionCandidate
from motion_agent.state.schemas import StrictModel


class SingleCandidateSelectionResult(StrictModel):
    status: Literal["selected", "no_successful_candidate", "tournament_required"]
    phase8_status: Literal["BYPASSED_FOR_K1_MODE"]
    generation_id: str
    champion_candidate_id: str | None = None
    candidate: MotionCandidate | None = None
    candidate_ids: list[str] = Field(default_factory=list)
    pairwise_evidence: list[dict] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)


class SingleCandidateSelector:
    def select(self, generation_result: GenerationResult) -> SingleCandidateSelectionResult:
        successful = list(generation_result.candidates)
        if len(successful) == 0:
            return SingleCandidateSelectionResult(
                status="no_successful_candidate",
                phase8_status="BYPASSED_FOR_K1_MODE",
                generation_id=generation_result.generation_id,
                candidate_ids=[],
                metadata={"failure": "NO_SUCCESSFUL_CANDIDATES"},
            )
        if len(successful) > 1:
            return SingleCandidateSelectionResult(
                status="tournament_required",
                phase8_status="BYPASSED_FOR_K1_MODE",
                generation_id=generation_result.generation_id,
                candidate_ids=[candidate.candidate_id for candidate in successful],
                metadata={"failure": "K_GT_1_UNSUPPORTED_IN_CURRENT_MODE"},
            )
        candidate = successful[0]
        return SingleCandidateSelectionResult(
            status="selected",
            phase8_status="BYPASSED_FOR_K1_MODE",
            generation_id=generation_result.generation_id,
            champion_candidate_id=candidate.candidate_id,
            candidate=candidate,
            candidate_ids=[candidate.candidate_id],
            pairwise_evidence=[],
            metadata={
                "generation_metadata": candidate.metadata.model_dump(mode="json"),
                "artifact_handles_preserved": True,
            },
        )
