"""Append-only execution event log."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import Field

from motion_agent.common.ids import new_id
from motion_agent.state.schemas import StrictModel


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class AgentEvent(StrictModel):
    event_id: str = Field(default_factory=lambda: new_id("event"))
    run_id: str
    event_type: str
    state_version_before: int
    state_version_after: int | None
    input_artifact_ids: list[str] = Field(default_factory=list)
    output_artifact_ids: list[str] = Field(default_factory=list)
    status: str = "success"
    trace_id: str
    span_id: str = Field(default_factory=lambda: new_id("span"))
    timestamp: str = Field(default_factory=utc_now)
    summary: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class EventLog:
    def __init__(self, root: str | Path = "data/events") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, run_id: str) -> Path:
        return self.root / f"{run_id}.jsonl"

    def append(self, event: AgentEvent) -> None:
        with self._path(event.run_id).open("a", encoding="utf-8") as f:
            f.write(event.model_dump_json() + "\n")

    def list(self, run_id: str) -> list[AgentEvent]:
        path = self._path(run_id)
        if not path.exists():
            return []
        return [AgentEvent.model_validate(json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line]

