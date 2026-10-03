"""Repair history, improvement, and regression tracking."""

from __future__ import annotations

from motion_agent.diagnosis.schemas import (
    FailureCase,
    NoImprovementState,
    ProgressStatus,
    RepairFamily,
    RepairHistoryEntry,
)


NO_IMPROVEMENT_BLOCK_AFTER = 2


class RepairHistory:
    def __init__(self, entries: list[RepairHistoryEntry] | None = None) -> None:
        self.entries = entries or []

    def blocked_families(self) -> set[tuple[str, str]]:
        return {
            (state.failure_signature, state.repair_family)
            for state in self.no_improvement_states()
            if state.blocked
        }

    def no_improvement_states(self) -> list[NoImprovementState]:
        counters: dict[tuple[str, RepairFamily], int] = {}
        for entry in self.entries:
            key = (entry.failure_signature, entry.repair_family)
            if entry.outcome in {"unchanged", "worse"}:
                counters[key] = counters.get(key, 0) + 1
            elif entry.outcome in {"improved", "resolved"}:
                counters[key] = 0
        return [
            NoImprovementState(
                failure_signature=signature,
                repair_family=family,
                unchanged_or_worse_attempts=count,
                blocked=count >= NO_IMPROVEMENT_BLOCK_AFTER,
            )
            for (signature, family), count in counters.items()
        ]

    def duplicate_fingerprints(self) -> set[str]:
        return {entry.proposal_fingerprint for entry in self.entries}


def classify_metric_progression(
    *,
    previous_metric: float | None,
    current_metric: float | None,
    threshold: float | None,
) -> ProgressStatus:
    if current_metric is None:
        return "unchanged"
    if threshold is not None and current_metric <= threshold:
        return "resolved"
    if previous_metric is None:
        return "failed"
    if current_metric < previous_metric:
        return "improved"
    if current_metric > previous_metric:
        return "worse"
    return "unchanged"


def detect_regressions(before: list[FailureCase], after: list[FailureCase]) -> list[str]:
    before_failures = {failure.failure_signature for failure in before}
    return [
        failure.failure_signature
        for failure in after
        if failure.failure_signature not in before_failures and failure.severity in {"major", "critical"}
    ]


def targeted_repair_success(*, target_resolved: bool, regressions: list[str]) -> bool:
    return target_resolved and not regressions
