"""Build specialized repair proposals from routed families."""

from __future__ import annotations

from motion_agent.diagnosis.policies import (
    ConstraintRepairProposalBuilder,
    EventRepairProposalBuilder,
    KeyframeRepairProposalBuilder,
    NaturalnessRepairProposalBuilder,
    PhysicalRepairProposalBuilder,
    PreservationRepairProposalBuilder,
    RetrievalRepairProposalBuilder,
    SemanticRepairProposalBuilder,
    StopUnresolvableProposalBuilder,
)
from motion_agent.diagnosis.schemas import FailureCluster, RepairFamily, RepairProposal, RootCauseHypothesis


BUILDER_REGISTRY = {
    "SEMANTIC_RECOMPILE": SemanticRepairProposalBuilder(),
    "REFERENCE_GROUNDING": RetrievalRepairProposalBuilder(),
    "CONSTRAINT_REBUILD": ConstraintRepairProposalBuilder(),
    "KEYFRAME_REBUILD": KeyframeRepairProposalBuilder(),
    "REGENERATE": EventRepairProposalBuilder(),
    "PHYSICAL_REGENERATE": PhysicalRepairProposalBuilder(),
    "PRESERVATION_REPAIR": PreservationRepairProposalBuilder(),
    "STOP_UNRESOLVABLE": StopUnresolvableProposalBuilder(),
}


def build_repair_proposals(
    *,
    cluster: FailureCluster,
    cause: RootCauseHypothesis,
    repair_families: list[RepairFamily],
    preserve: list[str],
) -> list[RepairProposal]:
    proposals: list[RepairProposal] = []
    for family in repair_families:
        builder = BUILDER_REGISTRY[family]
        if family == "REGENERATE" and cluster.failure_family == "NATURALNESS":
            builder = NaturalnessRepairProposalBuilder()
        proposals.append(builder.build(cluster=cluster, cause=cause, preserve=preserve))
    return proposals


def deduplicate_proposals(proposals: list[RepairProposal]) -> list[RepairProposal]:
    seen: set[str] = set()
    unique: list[RepairProposal] = []
    for proposal in proposals:
        if proposal.proposal_fingerprint in seen:
            continue
        seen.add(proposal.proposal_fingerprint)
        unique.append(proposal)
    return unique
