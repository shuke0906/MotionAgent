from __future__ import annotations

from motion_agent.agent.actions import PlannerAction
from motion_agent.diagnosis.policies.base import RepairProposalBuilder
from motion_agent.diagnosis.schemas import FailureCluster, RootCauseHypothesis


class PhysicalRepairProposalBuilder(RepairProposalBuilder):
    repair_family = "PHYSICAL_REGENERATE"
    planner_action = PlannerAction.GENERATE

    def build(self, *, cluster: FailureCluster, cause: RootCauseHypothesis, preserve: list[str]):
        return self._proposal(
            cluster=cluster,
            cause=cause,
            reason_code="NATURALNESS_FAILURE",
            focus=["physical_artifact"],
            parameters={"strategy": "normal", "scope": "segment" if cluster.target_segments else "full", "num_candidates": 1, "reward_targets": []},
            expected_effect="Regenerate to remove physical artifacts such as skating or penetration.",
            preserve=preserve,
        )
