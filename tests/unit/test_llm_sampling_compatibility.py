from unittest.mock import MagicMock, patch

import pytest

from motion_agent.llm.config import LLMConfig
from motion_agent.llm.providers.openai_backend import OpenAIBackend
from motion_agent.llm.schemas import PromptEnvelope
from motion_agent.state.schemas import StrictModel


class SmokeResponse(StrictModel):
    status: str


@pytest.mark.parametrize("model", ["gpt-6-sol", "gpt-5", "o3", "gpt-4o-mini"])
def test_sampling_is_compatible_with_model(model, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-only-placeholder")
    sdk = MagicMock()
    with patch("openai.OpenAI", return_value=sdk):
        OpenAIBackend(LLMConfig(model=model))._call(
            PromptEnvelope(prompt_version="test", system="Return JSON.", user="Check."),
            SmokeResponse,
        )
    kwargs = sdk.responses.create.call_args.kwargs
    if model == "gpt-4o-mini":
        assert kwargs["temperature"] == 0
    else:
        assert "temperature" not in kwargs
    assert kwargs["text"]["format"]["strict"] is True
    assert kwargs["store"] is False
