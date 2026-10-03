"""Failure clustering for Phase 10."""

from __future__ import annotations

from collections import defaultdict

from motion_agent.common.fingerprints import failure_signature
from motion_agent.common.ids import new_id
from motion_agent.diagnosis.schemas import FailureCase, FailureCluster


def cluster_failures(failures: list[FailureCase]) -> list[FailureCluster]:
    grouped: dict[tuple, list[FailureCase]] = defaultdict(list)
    for failure in failures:
        grouped[_cluster_key(failure)].append(failure)

    clusters: list[FailureCluster] = []
    for items in grouped.values():
        ordered = sorted(items, key=_failure_priority)
        primary = ordered[0]
        support = [item.failure_id for item in ordered[1:]]
        target_segments = sorted({segment for item in ordered for segment in item.target_segments})
        body_parts = sorted({part for item in ordered for part in item.target_body_parts})
        target_event = primary.target_event or next((item.target_event for item in ordered if item.target_event), None)
        sig = failure_signature(
            {
                "family": primary.failure_family,
                "target_segments": target_segments,
                "body_parts": body_parts,
                "target_event": target_event,
                "primary_signature": primary.failure_signature,
            }
        )
        clusters.append(
            FailureCluster(
                cluster_id=new_id("cluster"),
                failure_family=primary.failure_family,
                target_segments=target_segments,
                target_body_parts=body_parts,
                target_event=target_event,
                primary_failure_id=primary.failure_id,
                supporting_failure_ids=support,
                dependent_failure_ids=_dependent_failures(ordered),
                failure_signature=sig,
            )
        )
    return clusters


def _cluster_key(failure: FailureCase) -> tuple:
    family = failure.failure_family
    if family == "SEMANTIC" and failure.diagnostic_code in {
        "BODY_PART_MISMATCH",
        "SEMANTIC_BODY_PART_MISMATCH",
        "DIRECTION_MISMATCH",
        "SEMANTIC_DIRECTION_MISMATCH",
    }:
        family = "EVENT"
    return (
        family,
        tuple(failure.target_segments),
        tuple(failure.target_body_parts),
        failure.target_event,
        tuple(failure.source_spec_ids),
    )


def _failure_priority(failure: FailureCase) -> tuple[int, str]:
    severity_rank = {"critical": 0, "major": 1, "minor": 2, "info": 3}
    status_rank = {"fail": 0, "error": 1, "uncertain": 2}
    return (severity_rank.get(failure.severity, 3), status_rank.get(failure.status, 2), failure.failure_id)


def _dependent_failures(items: list[FailureCase]) -> list[str]:
    if not any(item.diagnostic_code == "EVENT_MISSING" for item in items):
        return []
    return [
        item.failure_id
        for item in items
        if item.diagnostic_code in {"EVENT_FREQUENCY_MISMATCH", "EVENT_ORDER_MISMATCH"}
    ]
