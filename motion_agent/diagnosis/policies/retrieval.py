from __future__ import annotations

from motion_agent.agent.actions import PlannerAction
from motion_agent.diagnosis.policies.base import RepairProposalBuilder
from motion_agent.diagnosis.schemas import FailureCluster, RootCauseHypothesis


class RetrievalRepairProposalBuilder(RepairProposalBuilder):
    repair_family = "REFERENCE_GROUNDING"
    planner_action = PlannerAction.RETRIEVE_REFERENCE

    def build(self, *, cluster: FailureCluster, cause: RootCauseHypothesis, preserve: list[str]):
        query = " ".join([cluster.target_event or "motion", *cluster.target_body_parts]).strip()
        return self._proposal(
            cluster=cluster,
            cause=cause,
            reason_code="RARE_MOTION",
            focus=["motion_prior"],
            parameters={"query": query or "motion prior", "retrieval_type": "motion", "purpose": "motion_prior", "top_k": 5},
            expected_effect="Retrieve a stronger motion prior for a requirement that generation repeatedly missed.",
            preserve=preserve,
        )
