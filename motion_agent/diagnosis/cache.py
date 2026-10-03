"""Diagnosis cache and idempotency keys."""

from __future__ import annotations

from motion_agent.common.fingerprints import stable_fingerprint
from motion_agent.diagnosis.schemas import DiagnosisRequest, DiagnosisResult


def diagnosis_cache_key(request: DiagnosisRequest) -> str:
    return stable_fingerprint(
        {
            "verification_report": request.verification_report.model_dump(mode="json"),
            "motion_spec": request.motion_spec,
            "generation_request": request.generation_request,
            "policy": request.diagnosis_policy_version,
            "rule": request.rule_version,
            "llm_model": request.llm_model_version,
            "prompt": request.prompt_version,
            "repair_history": [entry.model_dump(mode="json") for entry in request.repair_history],
        },
        namespace="diagnosis_cache",
    )


class DiagnosisCache:
    def __init__(self) -> None:
        self._items: dict[str, DiagnosisResult] = {}

    def get(self, key: str) -> DiagnosisResult | None:
        return self._items.get(key)

    def put(self, key: str, result: DiagnosisResult) -> None:
        self._items[key] = result
