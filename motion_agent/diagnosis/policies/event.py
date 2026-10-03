from __future__ import annotations

from motion_agent.agent.actions import PlannerAction
from motion_agent.diagnosis.policies.base import RepairProposalBuilder
from motion_agent.diagnosis.schemas import FailureCluster, RootCauseHypothesis


class EventRepairProposalBuilder(RepairProposalBuilder):
    repair_family = "REGENERATE"
    planner_action = PlannerAction.GENERATE

    def build(self, *, cluster: FailureCluster, cause: RootCauseHypothesis, preserve: list[str]):
        focus = ["event_execution"]
        if cluster.target_event:
            focus.append(cluster.target_event)
        return self._proposal(
            cluster=cluster,
            cause=cause,
            reason_code="TEMPORAL_FAILURE" if cause.cause_code == "TEMPORAL_RELATION_EXECUTION_FAILURE" else "SEMANTIC_FAILURE",
            focus=focus,
            parameters={"strategy": "normal", "scope": "segment" if cluster.target_segments else "full", "num_candidates": 1, "reward_targets": []},
            expected_effect="Regenerate while preserving the verified MotionSpecification and failed event requirements.",
            preserve=preserve,
        )
