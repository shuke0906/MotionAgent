from __future__ import annotations

from motion_agent.agent.actions import PlannerAction
from motion_agent.diagnosis.policies.base import RepairProposalBuilder
from motion_agent.diagnosis.schemas import FailureCluster, RootCauseHypothesis


class StopUnresolvableProposalBuilder(RepairProposalBuilder):
    repair_family = "STOP_UNRESOLVABLE"
    planner_action = PlannerAction.STOP_FAILED

    def build(self, *, cluster: FailureCluster, cause: RootCauseHypothesis, preserve: list[str]):
        return self._proposal(
            cluster=cluster,
            cause=cause,
            reason_code="UNRECOVERABLE_TOOL_FAILURE" if "TOOL" in cause.cause_code or "VERIFIER" in cause.cause_code else "BUDGET_EXHAUSTED",
            focus=["unresolvable"],
            parameters={},
            expected_effect="Surface unresolved failure to the Planner for terminal handling.",
            preserve=preserve,
            confidence_delta=-0.05,
        )
