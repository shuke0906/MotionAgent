"""Specialized repair proposal builders."""

from __future__ import annotations

from abc import ABC, abstractmethod

from motion_agent.common.fingerprints import repair_signature
from motion_agent.common.ids import new_id
from motion_agent.diagnosis.schemas import FailureCluster, RepairFamily, RepairProposal, RootCauseHypothesis
from motion_agent.agent.actions import PlannerAction


class RepairProposalBuilder(ABC):
    repair_family: RepairFamily
    planner_action: PlannerAction

    @abstractmethod
    def build(self, *, cluster: FailureCluster, cause: RootCauseHypothesis, preserve: list[str]) -> RepairProposal:
        ...

    def _proposal(
        self,
        *,
        cluster: FailureCluster,
        cause: RootCauseHypothesis,
        reason_code: str,
        focus: list[str],
        parameters: dict,
        expected_effect: str,
        preserve: list[str],
        confidence_delta: float = 0.0,
    ) -> RepairProposal:
        fingerprint_payload = {
            "family": self.repair_family,
            "action": self.planner_action.value,
            "cluster": cluster.failure_signature,
            "focus": focus,
            "parameters": parameters,
        }
        fingerprint = repair_signature(fingerprint_payload)
        return RepairProposal(
            proposal_id=new_id("repair"),
            repair_family=self.repair_family,
            planner_action=self.planner_action,
            reason_code=reason_code,
            target_segments=cluster.target_segments or None,
            target_body_parts=cluster.target_body_parts,
            focus=focus,
            parameters=parameters,
            evidence_refs=cause.evidence_failure_ids,
            expected_effect=expected_effect,
            preserve_requirements=preserve,
            confidence=max(0.0, min(1.0, cause.confidence + confidence_delta)),
            failure_signature=cluster.failure_signature,
            proposal_fingerprint=fingerprint,
        )
