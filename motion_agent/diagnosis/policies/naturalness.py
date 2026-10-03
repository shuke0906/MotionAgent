from __future__ import annotations

from motion_agent.agent.actions import PlannerAction
from motion_agent.diagnosis.policies.base import RepairProposalBuilder
from motion_agent.diagnosis.schemas import FailureCluster, RootCauseHypothesis


class NaturalnessRepairProposalBuilder(RepairProposalBuilder):
    repair_family = "REGENERATE"
    planner_action = PlannerAction.GENERATE

    def build(self, *, cluster: FailureCluster, cause: RootCauseHypothesis, preserve: list[str]):
        return self._proposal(
            cluster=cluster,
            cause=cause,
            reason_code="NATURALNESS_FAILURE",
            focus=["naturalness"],
            parameters={"strategy": "normal", "scope": "segment" if cluster.target_segments else "full", "num_candidates": 1, "reward_targets": []},
            expected_effect="Regenerate the failed scope to reduce artifacts while preserving validated semantics.",
            preserve=preserve,
        )
