"""In-memory verification report store for local Phase 9 runs."""

from __future__ import annotations

from motion_agent.verification.schemas import VerificationReport


class VerificationStore:
    def __init__(self) -> None:
        self._reports: dict[str, VerificationReport] = {}

    def put(self, report: VerificationReport) -> VerificationReport:
        self._reports[report.verification_id] = report
        return report

    def get(self, verification_id: str) -> VerificationReport:
        return self._reports[verification_id]
