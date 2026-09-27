"""Optimistic-concurrency state persistence."""

from __future__ import annotations

import json
from pathlib import Path

from motion_agent.common.errors import StateConflictError
from motion_agent.state.events import AgentEvent, EventLog
from motion_agent.state.invariants import validate_state_invariants
from motion_agent.state.schemas import MotionAgentState


class StateStore:
    def __init__(self, root: str | Path = "data/states", event_log: EventLog | None = None, artifact_store=None) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.event_log = event_log or EventLog()
        self.artifact_store = artifact_store

    def _run_dir(self, run_id: str) -> Path:
        path = self.root / run_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def latest_path(self, run_id: str) -> Path:
        return self._run_dir(run_id) / "latest.json"

    def load_latest(self, run_id: str) -> MotionAgentState:
        return MotionAgentState.model_validate(json.loads(self.latest_path(run_id).read_text(encoding="utf-8")))

    def current_version(self, run_id: str) -> int | None:
        path = self.latest_path(run_id)
        if not path.exists():
            return None
        return self.load_latest(run_id).state_version

    def commit(
        self,
        new_state: MotionAgentState,
        *,
        expected_version: int | None,
        event_type: str,
        input_artifact_ids: list[str] | None = None,
        output_artifact_ids: list[str] | None = None,
        summary: str | None = None,
    ) -> MotionAgentState:
        run_id = new_state.run.run_id
        current = self.current_version(run_id)
        if current != expected_version:
            raise StateConflictError(
                f"version conflict for {run_id}: expected {expected_version}, current {current}"
            )
        previous_version = current if current is not None else -1
        validate_state_invariants(
            new_state,
            artifact_store=self.artifact_store,
            previous_version=previous_version,
        )

        run_dir = self._run_dir(run_id)
        payload = new_state.model_dump_json(indent=2)
        (run_dir / f"v{new_state.state_version:06d}.json").write_text(payload, encoding="utf-8")
        self.latest_path(run_id).write_text(payload, encoding="utf-8")
        self.event_log.append(
            AgentEvent(
                run_id=run_id,
                event_type=event_type,
                state_version_before=previous_version,
                state_version_after=new_state.state_version,
                input_artifact_ids=input_artifact_ids or [],
                output_artifact_ids=output_artifact_ids or [],
                trace_id=new_state.run.trace_id,
                summary=summary,
            )
        )
        return new_state

