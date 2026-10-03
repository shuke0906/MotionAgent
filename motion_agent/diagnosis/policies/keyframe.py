from __future__ import annotations

from motion_agent.agent.actions import PlannerAction
from motion_agent.diagnosis.policies.base import RepairProposalBuilder
from motion_agent.diagnosis.schemas import FailureCluster, RootCauseHypothesis


class KeyframeRepairProposalBuilder(RepairProposalBuilder):
    repair_family = "KEYFRAME_REBUILD"
    planner_action = PlannerAction.BUILD_KEYFRAME

    def build(self, *, cluster: FailureCluster, cause: RootCauseHypothesis, preserve: list[str]):
        return self._proposal(
            cluster=cluster,
            cause=cause,
            reason_code="WHOLE_BODY_STATE_REQUIRED",
            focus=["keyframe_pose"],
            parameters={"mode": "replace", "time": 0.0, "description": "replace failed keyframe pose", "source_preference": "auto"},
            expected_effect="Replace the failed whole-body keyframe without rebuilding unrelated constraints.",
            preserve=preserve,
        )
