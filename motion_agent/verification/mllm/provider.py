"""OpenAI Responses provider; keys never enter artifacts, messages or cache keys."""

from __future__ import annotations

import base64
import os
import re
import time
from pathlib import Path

from motion_agent.llm.config import LLMConfig, load_dotenv, normalize_base_url
from motion_agent.llm.providers.openai_backend import OpenAIBackend
from motion_agent.verification.backends import BackendUnavailable


class ProviderError(RuntimeError):
    def __init__(self, code: str, retryable: bool = False):
        self.code, self.retryable = code, retryable
        super().__init__(code)


def api_base_url() -> str:
    """Accept an API root or a mistakenly configured endpoint, without changing host."""
    try:
        return normalize_base_url(os.environ.get("MLLM_BASE_URL") or os.environ.get("OPENAI_BASE_URL") or
                                  "https://api.openai.com/v1")
    except Exception as exc:
        raise BackendUnavailable("MLLM_INVALID_API_BASE_URL") from exc


class OpenAIVisualProvider:
    def __init__(self, config):
        self.config = config
        self.unavailable_reason = None
        self.network_calls = 0
        self.transport_errors = []

    def check_ready(self):
        load_dotenv()
        if self.unavailable_reason:
            raise BackendUnavailable(self.unavailable_reason)
        if not os.environ.get(self.config.mllm_api_key_env):
            if os.environ.get("OPENAI_API_KEY"):
                self.config.mllm_api_key_env = "OPENAI_API_KEY"
            else:
                raise BackendUnavailable("IMPLEMENTED_NOT_EXECUTED_NO_CREDENTIALS")
        api_base_url()

    def observe(self, prompt: str, requirements: str, storyboard: dict, schema):
        self.check_ready()
        content = [{"type": "input_text", "text": requirements}]
        for frame in storyboard.get("images") or storyboard["frames"]:
            label = ("2x2 tiles in row-major time order (blank cells are not observations): " +
                     "; ".join(f"Frame {t['frame_index']} at {t['timestamp_s']:.4f}s" for t in frame["tiles"])
                     if "tiles" in frame else f"Frame {frame['frame_index']}; timestamp {frame['timestamp_s']:.4f}s")
            content.append({"type": "input_text", "text": label})
            data = base64.b64encode(Path(frame["path"]).read_bytes()).decode("ascii")
            mime = "image/png" if Path(frame["path"]).suffix.lower() == ".png" else "image/jpeg"
            content.append({"type": "input_image", "image_url": "data:" + mime + ";base64," + data,
                            "detail": self.config.mllm_image_detail})
        try:
            backend = OpenAIBackend(LLMConfig(
                model=self.config.mllm_model, api_key_env=self.config.mllm_api_key_env,
                base_url=api_base_url(), timeout_s=self.config.mllm_timeout_s, max_retries=0,
            ))
            for attempt in range(2):
                if self.network_calls >= self.config.mllm_max_calls:
                    raise BackendUnavailable("MLLM_API_CALL_BUDGET_EXHAUSTED")
                self.network_calls += 1
                try:
                    response = backend.create_raw_response(instructions=prompt, content=content,
                                                           schema=schema, max_output_tokens=2200)
                    break
                except Exception as exc:
                    if type(exc).__name__ == "RateLimitError":
                        body = getattr(exc, "body", None)
                        error = body.get("error", body) if isinstance(body, dict) else {}
                        message = error.get("message", "") if isinstance(error, dict) else ""
                        counts = re.search(r"Limit[: ]+(\d+).*Requested[: ]+(\d+)", message)
                        self.transport_errors.append({"status_code": 429,
                            "kind": "insufficient_quota" if getattr(exc, "code", None) == "insufficient_quota" else "rate_limit",
                            "token_limit": int(counts[1]) if counts else None,
                            "requested_tokens": int(counts[2]) if counts else None})
                        if counts and int(counts[2]) > int(counts[1]):
                            raise ProviderError("MLLM_REQUEST_EXCEEDS_TOKEN_LIMIT") from None
                        if getattr(exc, "code", None) == "insufficient_quota":
                            self.unavailable_reason = "MLLM_INSUFFICIENT_QUOTA"
                            raise BackendUnavailable(self.unavailable_reason) from None
                        if attempt == 0 and self.network_calls < self.config.mllm_max_calls:
                            time.sleep(self.config.mllm_rate_limit_retry_s)
                            continue
                        raise ProviderError("MLLM_RATE_LIMIT_EXCEEDED", retryable=True) from None
                    raise
            if response.status != "completed" or not response.output_text:
                raise ProviderError("MLLM_REFUSAL_OR_INCOMPLETE")
            return {"response_id": response.id, "model": response.model, "output_text": response.output_text,
                    "usage": response.usage.model_dump(mode="json") if response.usage else None}
        except (ProviderError, BackendUnavailable):
            raise
        except Exception as exc:
            # Never persist provider exception text, which can contain credential/request data.
            code = type(exc).__name__
            if code == "AuthenticationError":
                self.unavailable_reason = "IMPLEMENTED_NOT_EXECUTED_NO_CREDENTIALS: credential rejected"
                raise BackendUnavailable(self.unavailable_reason) from None
            raise ProviderError("MLLM_" + code.upper()) from None
