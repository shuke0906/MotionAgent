"""Provider-neutral shared LLM client facade."""

from __future__ import annotations

from typing import TypeVar

from pydantic import BaseModel

from motion_agent.llm.config import LLMConfig
from motion_agent.llm.schemas import LLMCallMetadata, PromptEnvelope

T = TypeVar("T", bound=BaseModel)


class LLMClient:
    def __init__(self, config: LLMConfig | None = None) -> None:
        self.config = config or LLMConfig.from_env()
        if self.config.provider != "openai":
            from motion_agent.llm.errors import LLMConfigurationError

            raise LLMConfigurationError("LLM_PROVIDER_UNSUPPORTED")
        from motion_agent.llm.providers.openai_backend import OpenAIBackend

        self.backend = OpenAIBackend(self.config)
        self.last_metadata: LLMCallMetadata | None = None

    def complete_structured(self, envelope: PromptEnvelope, schema: type[T]) -> tuple[T, LLMCallMetadata]:
        result, metadata = self.backend.complete_structured(envelope, schema)
        self.last_metadata = metadata
        return result, metadata
