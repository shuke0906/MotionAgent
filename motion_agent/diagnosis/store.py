"""In-memory store for Phase 10 diagnosis results."""

from __future__ import annotations

from motion_agent.diagnosis.schemas import DiagnosisResult


class DiagnosisStore:
    def __init__(self) -> None:
        self._results: dict[str, DiagnosisResult] = {}

    def put(self, result: DiagnosisResult) -> DiagnosisResult:
        self._results[result.diagnosis_id] = result
        return result

    def get(self, diagnosis_id: str) -> DiagnosisResult:
        return self._results[diagnosis_id]
