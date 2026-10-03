from __future__ import annotations

import json

from motion_agent.diagnosis.llm.backend import DiagnosisLLMOutput
from motion_agent.llm import LLMClient, LLMConfig, PromptEnvelope
from motion_agent.llm.errors import LLMError


DIAGNOSIS_SYSTEM_PROMPT = """You are the selective MotionAgent diagnosis reasoner.

Interpret structured verification failures only when deterministic rules are ambiguous.
Do not execute repairs, modify state, choose final planner actions, or invent evidence.
Return one root cause code, confidence, evidence references, and candidate repair families."""


class RealDiagnosisLLMBackend:
    """Selective diagnosis adapter using the shared provider layer."""

    def __init__(self, client: LLMClient | None = None) -> None:
        self.client = client or LLMClient(LLMConfig.from_env())
        self.status = (
            "READY"
            if self.client.config.api_key_found
            else "IMPLEMENTED_NOT_EXECUTED_NO_CREDENTIALS"
        )
        self.last_metadata = None

    def diagnose(self, context: dict) -> DiagnosisLLMOutput:
        envelope = PromptEnvelope(
            prompt_version=self.client.config.prompt_version,
            system=DIAGNOSIS_SYSTEM_PROMPT,
            user=json.dumps({"diagnosis_context": context}, sort_keys=True),
            metadata={"component": "diagnosis.selective_llm"},
        )
        try:
            output, metadata = self.client.complete_structured(envelope, DiagnosisLLMOutput)
        except LLMError as exc:
            self.status = exc.code
            raise RuntimeError(exc.code) from None
        self.last_metadata = metadata
        self.status = "READY"
        return output
