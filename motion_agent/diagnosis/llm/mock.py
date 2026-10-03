from __future__ import annotations

from motion_agent.diagnosis.llm.backend import DiagnosisLLMOutput


class MockDiagnosisLLMBackend:
    status = "MOCK"

    def __init__(self, output: DiagnosisLLMOutput | None = None) -> None:
        self.output = output or DiagnosisLLMOutput(
            root_cause="UNKNOWN",
            confidence=0.25,
            evidence_refs=[],
            candidate_repair_families=["STOP_UNRESOLVABLE"],
        )

    def diagnose(self, context: dict) -> DiagnosisLLMOutput:
        return self.output
