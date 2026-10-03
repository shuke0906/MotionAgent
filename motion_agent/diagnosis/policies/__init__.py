"""Specialized Phase 10 repair proposal builders."""

from motion_agent.diagnosis.policies.base import RepairProposalBuilder
from motion_agent.diagnosis.policies.constraint import ConstraintRepairProposalBuilder
from motion_agent.diagnosis.policies.event import EventRepairProposalBuilder
from motion_agent.diagnosis.policies.keyframe import KeyframeRepairProposalBuilder
from motion_agent.diagnosis.policies.naturalness import NaturalnessRepairProposalBuilder
from motion_agent.diagnosis.policies.physical import PhysicalRepairProposalBuilder
from motion_agent.diagnosis.policies.preservation import PreservationRepairProposalBuilder
from motion_agent.diagnosis.policies.retrieval import RetrievalRepairProposalBuilder
from motion_agent.diagnosis.policies.semantic import SemanticRepairProposalBuilder
from motion_agent.diagnosis.policies.stop import StopUnresolvableProposalBuilder

__all__ = [
    "RepairProposalBuilder",
    "SemanticRepairProposalBuilder",
    "EventRepairProposalBuilder",
    "ConstraintRepairProposalBuilder",
    "KeyframeRepairProposalBuilder",
    "NaturalnessRepairProposalBuilder",
    "PhysicalRepairProposalBuilder",
    "PreservationRepairProposalBuilder",
    "RetrievalRepairProposalBuilder",
    "StopUnresolvableProposalBuilder",
]
