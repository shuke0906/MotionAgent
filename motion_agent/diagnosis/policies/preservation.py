from __future__ import annotations

from motion_agent.agent.actions import PlannerAction
from motion_agent.diagnosis.policies.base import RepairProposalBuilder
from motion_agent.diagnosis.schemas import FailureCluster, RootCauseHypothesis


class PreservationRepairProposalBuilder(RepairProposalBuilder):
    repair_family = "PRESERVATION_REPAIR"
    planner_action = PlannerAction.GENERATE

    def build(self, *, cluster: FailureCluster, cause: RootCauseHypothesis, preserve: list[str]):
        return self._proposal(
            cluster=cluster,
            cause=cause,
            reason_code="PERSISTENT_FAILURE",
            focus=["preservation", "outside_target_regions"],
            parameters={"strategy": "normal", "scope": "segment", "num_candidates": 1, "reward_targets": ["preservation"]},
            expected_effect="Retry localized generation while explicitly preserving already validated outside-target regions.",
            preserve=preserve,
        )
