"""Run one small structured API request without logging credentials."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Literal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from motion_agent.llm.client import LLMClient
from motion_agent.llm.config import LLMConfig, load_dotenv
from motion_agent.llm.providers.openai_backend import OpenAIBackend
from motion_agent.llm.schemas import PromptEnvelope
from motion_agent.state.schemas import StrictModel


class ConnectionCheck(StrictModel):
    status: Literal["ok"]


class DiagnosticBackend(OpenAIBackend):
    def _call(self, envelope, schema):
        try:
            return super()._call(envelope, schema)
        except Exception as exc:
            details = {}
            for name in ("code", "param"):
                value = getattr(exc, name, None)
                if isinstance(value, str) and re.fullmatch(r"[a-zA-Z0-9_.-]{1,80}", value):
                    details[name] = value
            status = getattr(exc, "status_code", None)
            if isinstance(status, int):
                details["http_status"] = status
            if details:
                print("SAFE_API_ERROR=" + json.dumps(details))
            raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--use-dotenv", action="store_true")
    args = parser.parse_args()
    if args.use_dotenv:
        # Override inherited LLM settings only inside this test process.
        for name in (
            "OPENAI_API_KEY", "OPENAI_MODEL", "OPENAI_BASE_URL",
            "OPENAI_TIMEOUT_S", "OPENAI_MAX_RETRIES", "MOTIONAGENT_PROMPT_VERSION",
        ):
            os.environ.pop(name, None)
    load_dotenv(ROOT / ".env")
    print("OPENAI_API_KEY=" + ("FOUND" if os.environ.get("OPENAI_API_KEY") else "MISSING"))
    print("OPENAI_MODEL=" + ("FOUND" if os.environ.get("OPENAI_MODEL") else "MISSING"))
    result = {
        "provider": "openai", "model": None, "latency_ms": 0.0,
        "success": False, "error_type": None,
    }
    started = time.perf_counter()
    try:
        config = LLMConfig.from_env(load_local_dotenv=False)
        config = config.model_copy(update={"max_retries": 0})
        result["model"] = config.model
        client = LLMClient(config)
        client.backend = DiagnosticBackend(config)
        _, metadata = client.complete_structured(
            PromptEnvelope(
                prompt_version="phase11_api_smoke_v1",
                system="Return exactly the structured object with status ok.",
                user="Check connection.",
            ),
            ConnectionCheck,
        )
        result.update(success=True, model=metadata.model)
    except Exception as exc:
        # Exception messages may contain provider responses or credentials.
        result["error_type"] = getattr(exc, "code", type(exc).__name__)
    result["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
    path = ROOT / "outputs" / "phase11_local_readiness" / "api_smoke_test.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
