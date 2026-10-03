"""Selective diagnosis LLM interfaces."""

from __future__ import annotations

from typing import Protocol

from motion_agent.state.schemas import StrictModel


class DiagnosisLLMOutput(StrictModel):
    root_cause: str
    confidence: float
    evidence_refs: list[str]
    candidate_repair_families: list[str]


class DiagnosisLLMBackend(Protocol):
    status: str

    def diagnose(self, context: dict) -> DiagnosisLLMOutput:
        ...


class UnavailableDiagnosisLLMBackend:
    status = "IMPLEMENTED_NOT_EXECUTED_NO_CREDENTIALS"

    def diagnose(self, context: dict) -> DiagnosisLLMOutput:
        raise RuntimeError(self.status)
