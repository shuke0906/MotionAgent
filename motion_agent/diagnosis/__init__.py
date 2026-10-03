"""Phase 10 diagnosis and targeted repair planning."""

from motion_agent.diagnosis.cache import DiagnosisCache, diagnosis_cache_key
from motion_agent.diagnosis.clustering import cluster_failures
from motion_agent.diagnosis.history import (
    RepairHistory,
    classify_metric_progression,
    detect_regressions,
    targeted_repair_success,
)
from motion_agent.diagnosis.normalizer import normalize_failures
from motion_agent.diagnosis.proposal_validator import validate_repair_proposal
from motion_agent.diagnosis.root_cause import diagnose_root_causes
from motion_agent.diagnosis.schemas import (
    DiagnosisRequest,
    DiagnosisResult,
    FailureCase,
    FailureCluster,
    RepairHistoryEntry,
    RepairProposal,
    RepairTarget,
    RootCauseHypothesis,
)
from motion_agent.diagnosis.service import DiagnosisService
from motion_agent.diagnosis.store import DiagnosisStore

__all__ = [
    "DiagnosisCache",
    "DiagnosisRequest",
    "DiagnosisResult",
    "DiagnosisService",
    "DiagnosisStore",
    "FailureCase",
    "FailureCluster",
    "RepairHistory",
    "RepairHistoryEntry",
    "RepairProposal",
    "RepairTarget",
    "RootCauseHypothesis",
    "classify_metric_progression",
    "cluster_failures",
    "detect_regressions",
    "diagnose_root_causes",
    "diagnosis_cache_key",
    "normalize_failures",
    "targeted_repair_success",
    "validate_repair_proposal",
]
