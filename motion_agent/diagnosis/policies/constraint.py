from __future__ import annotations

from motion_agent.agent.actions import PlannerAction
from motion_agent.diagnosis.policies.base import RepairProposalBuilder
from motion_agent.diagnosis.schemas import FailureCluster, RootCauseHypothesis


class ConstraintRepairProposalBuilder(RepairProposalBuilder):
    repair_family = "CONSTRAINT_REBUILD"
    planner_action = PlannerAction.BUILD_CONSTRAINT

    def build(self, *, cluster: FailureCluster, cause: RootCauseHypothesis, preserve: list[str]):
        constraint_type = "contact" if "CONTACT" in cluster.failure_signature else "root_trajectory" if "TRAJECTORY" in cluster.failure_signature else "joint_target"
        return self._proposal(
            cluster=cluster,
            cause=cause,
            reason_code="CONSTRAINT_FAILURE",
            focus=[constraint_type],
            parameters={"mode": "replace", "constraint_type": constraint_type, "body_part": _body_part(cluster), "target": None, "strength": "soft"},
            expected_effect="Rebuild only the failed measurable constraint and preserve unrelated passed conditions.",
            preserve=preserve,
        )


def _body_part(cluster: FailureCluster) -> str | None:
    return cluster.target_body_parts[0] if cluster.target_body_parts else None
