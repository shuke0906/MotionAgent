"""OpenAI Responses structured-output backend."""

from __future__ import annotations

import json
import os
import time
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from motion_agent.llm.config import LLMConfig
from motion_agent.llm.errors import LLMConfigurationError, LLMProviderError, LLMSchemaError
from motion_agent.llm.schemas import LLMCallMetadata, PromptEnvelope
from motion_agent.llm.json_schema import strict_output_schema

T = TypeVar("T", bound=BaseModel)


class OpenAIBackend:
    def __init__(self, config: LLMConfig) -> None:
        self.config = config

    def complete_structured(self, envelope: PromptEnvelope, schema: type[T]) -> tuple[T, LLMCallMetadata]:
        self.config.require_ready()
        started = time.perf_counter()
        try:
            response = self._call(envelope, schema)
            output_text = getattr(response, "output_text", None)
            if not output_text:
                raise LLMProviderError("OPENAI_EMPTY_OUTPUT", retryable=False)
            try:
                parsed = schema.model_validate_json(output_text)
            except ValidationError:
                try:
                    parsed = schema.model_validate(json.loads(output_text))
                except Exception as exc:
                    raise LLMSchemaError("OPENAI_SCHEMA_VALIDATION_FAILED") from exc
            metadata = self._metadata(envelope, schema, started, True, response=response)
            return parsed, metadata
        except (LLMConfigurationError, LLMProviderError, LLMSchemaError) as exc:
            metadata = self._metadata(envelope, schema, started, False, error_type=exc.code)
            self._last_failed_metadata = metadata
            raise
        except Exception as exc:
            code = type(exc).__name__
            metadata = self._metadata(envelope, schema, started, False, error_type="OPENAI_" + code.upper())
            self._last_failed_metadata = metadata
            raise LLMProviderError("OPENAI_" + code.upper(), retryable=True) from None

    def _call(self, envelope: PromptEnvelope, schema: type[T]):
        return self.create_raw_response(
            instructions=envelope.system,
            content=[{"type": "input_text", "text": envelope.user}],
            schema=schema,
        )

    def create_raw_response(self, *, instructions: str, content: list[dict], schema, max_output_tokens: int = 1200):
        """Shared transport for text and image adapters with their own validation."""
        from openai import OpenAI

        client = OpenAI(
            api_key=os.environ[self.config.api_key_env],
            base_url=self.config.base_url,
            timeout=self.config.timeout_s,
            max_retries=self.config.max_retries,
        )
        # Reasoning models reject temperature with their default reasoning effort.
        sampling = {} if self.config.model.startswith(("gpt-5", "gpt-6", "o1", "o3", "o4")) else {"temperature": 0}
        return client.responses.create(
            model=self.config.model,
            instructions=instructions,
            input=[{"role": "user", "content": content}],
            **sampling,
            max_output_tokens=max_output_tokens,
            store=False,
            text={
                "format": {
                    "type": "json_schema",
                    "name": schema.__name__,
                    "strict": True,
                    "schema": strict_output_schema(schema),
                }
            },
        )

    def _metadata(
        self,
        envelope: PromptEnvelope,
        schema: type[BaseModel],
        started: float,
        success: bool,
        *,
        response=None,
        error_type: str | None = None,
    ) -> LLMCallMetadata:
        usage = None
        if response is not None and getattr(response, "usage", None) is not None:
            try:
                usage = response.usage.model_dump(mode="json")
            except AttributeError:
                usage = dict(response.usage)
        return LLMCallMetadata(
            provider=self.config.provider,
            model=getattr(response, "model", None) or self.config.model,
            prompt_version=envelope.prompt_version,
            schema_name=schema.__name__,
            latency_ms=(time.perf_counter() - started) * 1000,
            success=success,
            error_type=error_type,
            usage=usage,
            response_id=getattr(response, "id", None) if response is not None else None,
        )
