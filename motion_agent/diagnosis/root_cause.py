"""Rule-first root cause analysis for Phase 10."""

from __future__ import annotations

from typing import Any

from motion_agent.common.ids import new_id
from motion_agent.diagnosis.schemas import FailureCase, FailureCluster, RepairHistoryEntry, RootCauseHypothesis


def diagnose_root_causes(
    *,
    clusters: list[FailureCluster],
    failures: list[FailureCase],
    motion_spec: Any | None,
    generation_request: Any | None,
    history: list[RepairHistoryEntry],
) -> list[RootCauseHypothesis]:
    failures_by_id = {failure.failure_id: failure for failure in failures}
    return [
        _diagnose_cluster(
            cluster=cluster,
            failures_by_id=failures_by_id,
            motion_spec=motion_spec,
            generation_request=generation_request,
            history=history,
        )
        for cluster in clusters
    ]


def _diagnose_cluster(
    *,
    cluster: FailureCluster,
    failures_by_id: dict[str, FailureCase],
    motion_spec: Any | None,
    generation_request: Any | None,
    history: list[RepairHistoryEntry],
) -> RootCauseHypothesis:
    failure_ids = [cluster.primary_failure_id, *cluster.supporting_failure_ids]
    failures = [failures_by_id[fid] for fid in failure_ids if fid in failures_by_id]
    primary = failures[0]
    code = primary.diagnostic_code

    if primary.status in {"error", "uncertain"} or primary.failure_family == "VERIFICATION_INFRA":
        return _hypothesis(cluster, failures, "VERIFIER_INFRASTRUCTURE_FAILURE", 0.85, ["Verifier returned no reliable motion evidence."])

    if primary.failure_family == "TECHNICAL":
        return _hypothesis(cluster, failures, "TECHNICAL_TOOL_FAILURE", 0.8, ["Candidate or generation artifact is technically invalid."])

    if code == "UNCOMMANDED_HEADING_DRIFT":
        if not _spec_has_heading_policy(motion_spec, primary.target_segments):
            return _hypothesis(cluster, failures, "SPECIFICATION_INCOMPLETE", 0.9, ["MotionSpecification lacks a heading-continuity policy."])
        if _persistent(history, cluster.failure_signature, {"REGENERATE", "PHYSICAL_REGENERATE"}):
            return _hypothesis(cluster, failures, "CONSTRAINT_TOO_WEAK", 0.78, ["Heading drift persisted after sampling repair."])
        return _hypothesis(cluster, failures, "SAMPLING_VARIANCE", 0.72, ["Heading policy exists; current evidence supports a sampling failure."])

    if primary.failure_family in {"EVENT", "SEMANTIC"}:
        return _diagnose_semantic_event(cluster, failures, motion_spec, generation_request)

    if primary.failure_family == "CONSTRAINT":
        if _constraint_spec_invalid(primary, motion_spec, generation_request):
            return _hypothesis(cluster, failures, "CONSTRAINT_TARGET_INADEQUATE", 0.86, ["Constraint evidence says the active verification spec is invalid or inconsistent."])
        return _hypothesis(cluster, failures, "CONSTRAINT_TOO_WEAK", 0.82, ["A valid constraint reached generation, but the candidate did not satisfy it."])

    if primary.failure_family == "KEYFRAME":
        if _keyframe_spec_invalid(primary, motion_spec, generation_request):
            return _hypothesis(cluster, failures, "KEYFRAME_INADEQUATE", 0.86, ["Keyframe evidence says the active keyframe spec is invalid or incomplete."])
        return _hypothesis(cluster, failures, "SAMPLING_VARIANCE", 0.78, ["A valid keyframe reached generation, but the candidate missed it."])

    if primary.failure_family == "NATURALNESS":
        if _recent_hard_constraint(history):
            return _hypothesis(cluster, failures, "CONSTRAINT_TOO_STRONG", 0.72, ["Naturalness failure followed a hard/measurable repair attempt."])
        return _hypothesis(cluster, failures, "SAMPLING_VARIANCE", 0.74, ["Naturalness failed without evidence that semantic compilation changed."])

    if primary.failure_family == "PHYSICAL":
        return _hypothesis(cluster, failures, "PHYSICAL_SAMPLING_ARTIFACT", 0.78, ["Physical artifact was observed in the generated candidate."])

    if primary.failure_family == "PRESERVATION":
        return _hypothesis(cluster, failures, "SEGMENT_BOUNDARY_ARTIFACT", 0.8, ["Segment repair regressed a previously preserved region."])

    return _hypothesis(cluster, failures, "UNKNOWN", 0.25, ["Evidence is insufficient to select a root cause."], ambiguous=True)


def _diagnose_semantic_event(
    cluster: FailureCluster,
    failures: list[FailureCase],
    motion_spec: Any | None,
    generation_request: Any | None,
) -> RootCauseHypothesis:
    primary = failures[0]
    code = primary.diagnostic_code
    requirement = _requirement_from_failure(primary)
    spec_has = _spec_preserves_requirement(motion_spec, primary, requirement)
    gen_has = _generation_preserves_requirement(generation_request, primary, requirement)

    if not spec_has:
        return _hypothesis(
            cluster,
            failures,
            "SEMANTIC_COMPILATION_LOSS",
            0.9,
            [f"MotionSpecification does not preserve {requirement['kind']}={requirement.get('value')}."],
        )
    if gen_has is False:
        return _hypothesis(
            cluster,
            failures,
            "CAPTION_MISMATCH",
            0.78,
            ["MotionSpecification preserved the requirement, but GenerationRequest evidence does not."],
        )
    if code == "EVENT_FREQUENCY_MISMATCH":
        return _hypothesis(
            cluster,
            failures,
            "GENERATION_SEMANTIC_EXECUTION_FAILURE",
            0.88,
            ["Count was preserved through compiler and generation request, but observed candidate count differs."],
        )
    if code == "EVENT_ORDER_MISMATCH":
        return _hypothesis(cluster, failures, "TEMPORAL_RELATION_EXECUTION_FAILURE", 0.84, ["Expected order survived into structured semantics."])
    if code in {"BODY_PART_MISMATCH", "SEMANTIC_BODY_PART_MISMATCH", "DIRECTION_MISMATCH", "SEMANTIC_DIRECTION_MISMATCH"}:
        return _hypothesis(cluster, failures, "GENERATION_SEMANTIC_EXECUTION_FAILURE", 0.86, ["Structured semantics preserve the target, but generated motion differs."])
    if code == "EVENT_MISSING":
        return _hypothesis(cluster, failures, "GENERATION_SEMANTIC_EXECUTION_FAILURE", 0.84, ["Required event exists in structured semantics but was absent from the candidate."])
    return _hypothesis(cluster, failures, "UNKNOWN", 0.35, ["Semantic/event evidence is ambiguous."], ambiguous=True)


def _hypothesis(
    cluster: FailureCluster,
    failures: list[FailureCase],
    cause_code: str,
    confidence: float,
    supporting_facts: list[str],
    *,
    contradicting_facts: list[str] | None = None,
    ambiguous: bool = False,
) -> RootCauseHypothesis:
    return RootCauseHypothesis(
        hypothesis_id=new_id("cause"),
        cluster_id=cluster.cluster_id,
        cause_code=cause_code,  # type: ignore[arg-type]
        confidence=confidence,
        evidence_failure_ids=[failure.failure_id for failure in failures],
        supporting_facts=supporting_facts,
        contradicting_facts=contradicting_facts or [],
        affected_segments=cluster.target_segments,
        ambiguous=ambiguous,
    )


def _requirement_from_failure(failure: FailureCase) -> dict[str, Any]:
    expected = failure.expected
    if failure.diagnostic_code == "EVENT_FREQUENCY_MISMATCH":
        return {"kind": "count", "value": expected.get("count") or expected.get("expected_count"), "event": failure.target_event}
    if "direction" in expected:
        return {"kind": "direction", "value": expected.get("direction"), "event": failure.target_event}
    if failure.target_body_parts:
        return {"kind": "body_part", "value": failure.target_body_parts[0], "event": failure.target_event}
    if failure.target_event:
        return {"kind": "event", "value": failure.target_event, "event": failure.target_event}
    return {"kind": "semantic", "value": expected, "event": failure.target_event}


def _spec_preserves_requirement(motion_spec: Any | None, failure: FailureCase, requirement: dict[str, Any]) -> bool:
    if motion_spec is None:
        return False
    segments = _segments(motion_spec)
    relevant = _relevant_segments(segments, failure)
    kind = requirement["kind"]
    value = requirement.get("value")
    if kind == "count":
        if value is None:
            return False
        return any(_segment_count(segment) == int(value) for segment in relevant)
    if kind == "direction":
        return any(_get(segment, "direction") == value for segment in relevant)
    if kind == "body_part":
        return any(value in (_get(segment, "body_parts") or []) for segment in relevant)
    if kind == "event":
        return any(_action_matches(segment, str(value)) for segment in relevant)
    return bool(relevant)


def _generation_preserves_requirement(generation_request: Any | None, failure: FailureCase, requirement: dict[str, Any]) -> bool | None:
    if generation_request is None:
        return None
    motion_spec = _get(generation_request, "motion_spec")
    if motion_spec is None and isinstance(generation_request, dict):
        motion_spec = generation_request.get("motion_spec")
    if motion_spec is None:
        return None
    return _spec_preserves_requirement(motion_spec, failure, requirement)


def _segments(motion_spec: Any) -> list[Any]:
    if isinstance(motion_spec, dict):
        return list(motion_spec.get("segments", []))
    return list(getattr(motion_spec, "segments", []) or [])


def _relevant_segments(segments: list[Any], failure: FailureCase) -> list[Any]:
    if failure.target_segments:
        selected = [segment for segment in segments if _get(segment, "segment_id") in failure.target_segments]
        if selected:
            return selected
    if failure.target_event:
        selected = [segment for segment in segments if _action_matches(segment, failure.target_event)]
        if selected:
            return selected
    return segments


def _action_matches(segment: Any, event: str) -> bool:
    event = event.lower().replace("_", " ")
    action = str(_get(segment, "action") or "").lower().replace("_", " ")
    if event in action or action in event:
        return True
    return any(event in str(item).lower().replace("_", " ") for item in (_get(segment, "secondary_actions") or []))


def _segment_count(segment: Any) -> int | None:
    tc = _get(segment, "temporal_constraint")
    if tc is not None:
        count = _get(tc, "count")
        if count is not None:
            return int(count)
    repetition = _get(segment, "repetition")
    return int(repetition) if repetition is not None else None


def _get(value: Any, field: str) -> Any:
    if isinstance(value, dict):
        return value.get(field)
    return getattr(value, field, None)


def _constraint_spec_invalid(failure: FailureCase, motion_spec: Any | None, generation_request: Any | None) -> bool:
    return bool(
        failure.expected.get("spec_valid") is False
        or failure.expected.get("constraint_spec_valid") is False
        or failure.observed.get("spec_invalid")
        or failure.observed.get("constraint_missing")
    )


def _keyframe_spec_invalid(failure: FailureCase, motion_spec: Any | None, generation_request: Any | None) -> bool:
    return bool(
        failure.expected.get("spec_valid") is False
        or failure.expected.get("keyframe_spec_valid") is False
        or failure.observed.get("keyframe_missing")
    )


def _spec_has_heading_policy(motion_spec: Any | None, target_segments: list[int]) -> bool:
    if motion_spec is None:
        return False
    for segment in _relevant_segments(_segments(motion_spec), _synthetic_failure(target_segments)):
        heading = _get(segment, "heading_continuity")
        if heading is not None and _get(heading, "mode") in {"inherit_previous", "explicit", "follow_trajectory"}:
            return True
    return False


def _synthetic_failure(target_segments: list[int]):
    class _Failure:
        pass

    failure = _Failure()
    failure.target_segments = target_segments
    failure.target_event = None
    return failure


def _persistent(history: list[RepairHistoryEntry], failure_signature: str, families: set[str]) -> bool:
    return sum(
        1
        for entry in history
        if entry.failure_signature == failure_signature and entry.repair_family in families and entry.outcome in {"unchanged", "worse"}
    ) >= 1


def _recent_hard_constraint(history: list[RepairHistoryEntry]) -> bool:
    return any(entry.repair_family == "CONSTRAINT_REBUILD" for entry in history[-2:])
