"""Phase 2 application orchestrator and graph runner."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver

from motion_agent.agent.mock_tools import ToolRuntime
from motion_agent.agent.planner import Planner
from motion_agent.graph.builder import build_motion_graph
from motion_agent.graph.state import GraphState, graph_state_from_motion_state
from motion_agent.state.artifacts import ArtifactStore
from motion_agent.state.events import EventLog
from motion_agent.state.reducer import StateReducer
from motion_agent.state.schemas import MotionAgentState
from motion_agent.state.versioning import StateStore


class MotionAgentOrchestrator:
    """Create, invoke, interrupt, and resume the Phase 2 LangGraph runtime."""

    def __init__(self, root: str | Path, planner: Planner) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.artifact_store = ArtifactStore(self.root / "artifacts")
        self.event_log = EventLog(self.root / "events")
        self.state_store = StateStore(
            self.root / "states",
            event_log=self.event_log,
            artifact_store=self.artifact_store,
        )
        self.reducer = StateReducer()
        self.runtime = ToolRuntime(
            state_store=self.state_store,
            artifact_store=self.artifact_store,
            reducer=self.reducer,
        )
        self.planner = planner
        self.checkpoint_db = self.root / "checkpoints" / "langgraph.sqlite3"
        self.checkpoint_db.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def config(run_id: str) -> dict:
        return {"configurable": {"thread_id": run_id}, "recursion_limit": 100}

    @contextmanager
    def _graph(self) -> Iterator:
        connection = sqlite3.connect(self.checkpoint_db, check_same_thread=False)
        serializer = JsonPlusSerializer(allowed_msgpack_modules=())
        try:
            checkpointer = SqliteSaver(connection, serde=serializer)
            yield build_motion_graph(checkpointer=checkpointer, runtime=self.runtime, planner=self.planner)
        finally:
            connection.close()

    def create_run(
        self,
        original_request: str,
        *,
        run_id: str | None = None,
        generations_left: int | None = None,
        iterations_left: int | None = None,
    ) -> MotionAgentState:
        state = self.reducer.create_run(original_request, run_id=run_id)
        if generations_left is not None:
            state.control.budgets.generations_left = generations_left
        state.control.budgets.iterations_left = iterations_left if iterations_left is not None else 30
        return self.state_store.commit(state, expected_version=None, event_type="run_created")

    def invoke(
        self,
        state: MotionAgentState,
        *,
        interrupt_before: list[str] | None = None,
        interrupt_after: list[str] | None = None,
    ) -> MotionAgentState:
        graph_state = graph_state_from_motion_state(state, next_node="planner")
        with self._graph() as graph:
            graph.invoke(
                graph_state.model_dump(mode="python"),
                self.config(state.run.run_id),
                interrupt_before=interrupt_before,
                interrupt_after=interrupt_after,
                durability="sync",
            )
        return self.state_store.load_latest(state.run.run_id)

    def resume(
        self,
        run_id: str,
        *,
        interrupt_before: list[str] | None = None,
        interrupt_after: list[str] | None = None,
    ) -> MotionAgentState:
        with self._graph() as graph:
            graph.invoke(
                None,
                self.config(run_id),
                interrupt_before=interrupt_before,
                interrupt_after=interrupt_after,
                durability="sync",
            )
        return self.state_store.load_latest(run_id)

    def run(
        self,
        original_request: str,
        *,
        run_id: str | None = None,
        generations_left: int | None = None,
        iterations_left: int | None = None,
    ) -> MotionAgentState:
        state = self.create_run(
            original_request,
            run_id=run_id,
            generations_left=generations_left,
            iterations_left=iterations_left,
        )
        return self.invoke(state)

    def load_graph_state(self, run_id: str) -> GraphState | None:
        with self._graph() as graph:
            snapshot = graph.get_state(self.config(run_id))
        if not snapshot.values:
            return None
        return GraphState.model_validate(snapshot.values)
