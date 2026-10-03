from __future__ import annotations

from motion_agent.agent.actions import PlannerAction
from motion_agent.diagnosis.policies.base import RepairProposalBuilder
from motion_agent.diagnosis.schemas import FailureCluster, RootCauseHypothesis


class SemanticRepairProposalBuilder(RepairProposalBuilder):
    repair_family = "SEMANTIC_RECOMPILE"
    planner_action = PlannerAction.COMPILE_MOTION

    def build(self, *, cluster: FailureCluster, cause: RootCauseHypothesis, preserve: list[str]):
        focus = _focus_for_cluster(cluster, cause)
        return self._proposal(
            cluster=cluster,
            cause=cause,
            reason_code="TEMPORAL_FAILURE" if any(x in focus for x in ["timeline", "frequency"]) else "SEMANTIC_FAILURE",
            focus=focus,
            parameters={"mode": "revise", "focus": focus},
            expected_effect="Revise only the missing or incorrect structured semantic fields.",
            preserve=preserve,
        )


def _focus_for_cluster(cluster: FailureCluster, cause: RootCauseHypothesis) -> list[str]:
    focus: list[str] = []
    if cause.cause_code == "CAPTION_MISMATCH":
        focus.append("gem_caption")
    if cluster.target_event or "FREQUENCY" in cluster.failure_signature:
        focus.extend(["frequency", "gem_caption"])
    if "DIRECTION" in cluster.failure_signature:
        focus.append("semantic")
    if cluster.target_body_parts:
        focus.append("body_part")
    if not focus:
        focus = ["semantic", "gem_caption"]
    return list(dict.fromkeys(focus))
