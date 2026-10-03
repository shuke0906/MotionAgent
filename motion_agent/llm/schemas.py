"""Shared LLM call metadata schemas."""

from __future__ import annotations

from typing import Any

from pydantic import Field

from motion_agent.state.schemas import StrictModel


class PromptEnvelope(StrictModel):
    prompt_version: str
    system: str
    user: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class LLMCallMetadata(StrictModel):
    provider: str
    model: str
    prompt_version: str
    schema_name: str
    latency_ms: float
    success: bool
    error_type: str | None = None
    usage: dict[str, Any] | None = None
    response_id: str | None = None
