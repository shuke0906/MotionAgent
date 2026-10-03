"""Environment-backed LLM configuration.

Only variable names and non-secret metadata are exposed by this module.
"""

from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from pydantic import Field

from motion_agent.llm.errors import LLMConfigurationError
from motion_agent.state.schemas import StrictModel


def load_dotenv(path: str | Path = ".env") -> None:
    """Load simple KEY=VALUE pairs into the process environment if absent.

    Values are never returned or logged. Existing environment variables win over
    local files so deployment-level configuration remains authoritative.
    """

    dotenv = Path(path)
    if not dotenv.exists():
        return
    for raw_line in dotenv.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or key in os.environ:
            continue
        value = value.strip().strip('"').strip("'")
        os.environ[key] = value


class LLMConfig(StrictModel):
    provider: str = "openai"
    model: str = "gpt-4o-mini"
    api_key_env: str = "OPENAI_API_KEY"
    base_url: str = "https://api.openai.com/v1"
    timeout_s: float = Field(default=30.0, gt=0)
    max_retries: int = Field(default=1, ge=0, le=3)
    prompt_version: str = "phase11_local_v1"

    @classmethod
    def from_env(cls, *, load_local_dotenv: bool = True) -> "LLMConfig":
        if load_local_dotenv:
            load_dotenv()
        base_url = normalize_base_url(os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1")
        return cls(
            provider="openai",
            model=os.environ.get("OPENAI_MODEL") or "gpt-4o-mini",
            api_key_env="OPENAI_API_KEY",
            base_url=base_url,
            timeout_s=float(os.environ.get("OPENAI_TIMEOUT_S", "30")),
            max_retries=int(os.environ.get("OPENAI_MAX_RETRIES", "1")),
            prompt_version=os.environ.get("MOTIONAGENT_PROMPT_VERSION", "phase11_local_v1"),
        )

    @property
    def api_key_found(self) -> bool:
        return bool(os.environ.get(self.api_key_env))

    @property
    def model_found(self) -> bool:
        return bool(os.environ.get("OPENAI_MODEL"))

    def require_ready(self) -> None:
        if not self.api_key_found:
            raise LLMConfigurationError("OPENAI_API_KEY_MISSING")

    def safe_snapshot(self) -> dict:
        return {
            "provider": self.provider,
            "model": self.model,
            "api_key_env": self.api_key_env,
            "api_key_found": self.api_key_found,
            "model_env_found": self.model_found,
            "base_url_host": urlsplit(self.base_url).netloc,
            "timeout_s": self.timeout_s,
            "max_retries": self.max_retries,
            "prompt_version": self.prompt_version,
        }


def normalize_base_url(value: str) -> str:
    parts = urlsplit(value)
    if parts.scheme not in {"http", "https"} or not parts.hostname or parts.username or parts.password:
        raise LLMConfigurationError("OPENAI_BASE_URL_INVALID")
    path = parts.path.rstrip("/")
    for endpoint in ("/responses", "/chat/completions", "/embeddings"):
        if path.endswith(endpoint):
            path = path[: -len(endpoint)]
            break
    return urlunsplit((parts.scheme, parts.netloc, path or "/v1", "", ""))
