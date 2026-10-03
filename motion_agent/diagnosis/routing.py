"""Map diagnosed root causes to repair families and existing Planner actions."""

from __future__ import annotations

from motion_agent.diagnosis.schemas import FailureCluster, RepairFamily, RootCauseHypothesis, RepairHistoryEntry


def route_repair_families(
    *,
    cluster: FailureCluster,
    cause: RootCauseHypothesis,
    history: list[RepairHistoryEntry],
) -> list[RepairFamily]:
    blocked = {
        entry.repair_family
        for entry in history
        if entry.failure_signature == cluster.failure_signature and entry.outcome in {"unchanged", "worse"}
    }
    families = _base_families(cluster, cause)
    if len(blocked) < 2:
        return families
    return [family for family in families if family not in blocked] or ["STOP_UNRESOLVABLE"]


def _base_families(cluster: FailureCluster, cause: RootCauseHypothesis) -> list[RepairFamily]:
    code = cause.cause_code
    if code in {"SEMANTIC_COMPILATION_LOSS", "SPECIFICATION_INCOMPLETE", "CAPTION_MISMATCH"}:
        return ["SEMANTIC_RECOMPILE"]
    if code == "MISSING_MOTION_PRIOR":
        return ["REFERENCE_GROUNDING", "REGENERATE"]
    if code in {"CONSTRAINT_TARGET_INADEQUATE", "CONSTRAINT_TOO_STRONG", "GUIDANCE_NEEDED"}:
        return ["CONSTRAINT_REBUILD", "REGENERATE"]
    if code == "CONSTRAINT_TOO_WEAK":
        return ["REGENERATE", "CONSTRAINT_REBUILD"]
    if code == "KEYFRAME_INADEQUATE":
        return ["KEYFRAME_REBUILD", "REGENERATE"]
    if code in {"PHYSICAL_SAMPLING_ARTIFACT", "SAMPLING_VARIANCE"}:
        return ["PHYSICAL_REGENERATE" if cluster.failure_family == "PHYSICAL" else "REGENERATE"]
    if code == "SEGMENT_BOUNDARY_ARTIFACT":
        return ["PRESERVATION_REPAIR"]
    if code in {"TECHNICAL_TOOL_FAILURE", "VERIFIER_INFRASTRUCTURE_FAILURE"}:
        return ["STOP_UNRESOLVABLE"]
    if code in {"GENERATION_SEMANTIC_EXECUTION_FAILURE", "TEMPORAL_RELATION_EXECUTION_FAILURE", "REPETITION_EXECUTION_INSUFFICIENT"}:
        return ["REGENERATE"]
    if cluster.failure_family == "CONSTRAINT":
        return ["CONSTRAINT_REBUILD", "REGENERATE"]
    if cluster.failure_family == "KEYFRAME":
        return ["KEYFRAME_REBUILD", "REGENERATE"]
    if cluster.failure_family == "NATURALNESS":
        return ["REGENERATE"]
    return ["STOP_UNRESOLVABLE"]
