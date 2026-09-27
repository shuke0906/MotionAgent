"""Persistent local LangGraph runtime for the Phase 1 state boundary."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from motion_agent.common.errors import CheckpointError
from motion_agent.graph.state import GraphState, graph_state_from_motion_state, reconcile_graph_state
from motion_agent.state.schemas import MotionAgentState
from motion_agent.state.versioning import StateStore


@dataclass(frozen=True)
class ReconciledRuntimeState:
    """State recovered from the independent graph and domain stores."""

    graph_state: GraphState
    canonical_state: MotionAgentState


def _persist_checkpoint(state: GraphState) -> dict:
    """Validate the lightweight envelope before LangGraph commits it."""

    return {
        "run_id": state.run_id,
        "thread_id": state.thread_id,
        "state_version": state.state_version,
        "next_node": state.next_node,
        "planner_decision": state.planner_decision,
        "committed_operation_ids": state.committed_operation_ids,
        "metadata": state.metadata,
    }


def build_persistence_graph(checkpointer: SqliteSaver):
    """Compile the Phase 1 persistence graph without Phase 2 routing."""

    builder = StateGraph(GraphState)
    builder.add_node("persist_checkpoint", _persist_checkpoint)
    builder.add_edge(START, "persist_checkpoint")
    builder.add_edge("persist_checkpoint", END)
    return builder.compile(checkpointer=checkpointer, name="motion_agent_phase1_persistence")


class LocalGraphCheckpointer:
    """Real LangGraph StateGraph backed by a persistent local SQLite saver."""

    def __init__(self, root: str | Path = "data/checkpoints") -> None:
        self.root = Path(root)
        if self.root.suffix.lower() in {".db", ".sqlite", ".sqlite3"}:
            self.db_path = self.root
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            self.root.mkdir(parents=True, exist_ok=True)
            self.db_path = self.root / "langgraph.sqlite3"

    @staticmethod
    def _config(thread_id: str) -> dict:
        return {"configurable": {"thread_id": thread_id}}

    @contextmanager
    def _runtime(self) -> Iterator:
        connection = sqlite3.connect(self.db_path, check_same_thread=False)
        serializer = JsonPlusSerializer(allowed_msgpack_modules=())
        try:
            checkpointer = SqliteSaver(connection, serde=serializer)
            yield build_persistence_graph(checkpointer)
        finally:
            connection.close()

    def save(self, graph_state: GraphState) -> GraphState:
        if graph_state.thread_id != graph_state.run_id:
            raise CheckpointError("thread_id must equal run_id")
        with self._runtime() as graph:
            persisted = graph.invoke(
                graph_state.model_dump(mode="python"),
                self._config(graph_state.thread_id),
                durability="sync",
            )
        return GraphState.model_validate(persisted)

    def checkpoint_canonical_state(
        self,
        state: MotionAgentState,
        *,
        next_node: str | None = None,
        committed_operation_ids: list[str] | None = None,
    ) -> GraphState:
        graph_state = graph_state_from_motion_state(
            state,
            next_node=next_node,
            committed_operation_ids=committed_operation_ids,
        )
        return self.save(graph_state)

    def load(self, thread_id: str) -> GraphState | None:
        with self._runtime() as graph:
            snapshot = graph.get_state(self._config(thread_id))
        if not snapshot.values:
            return None
        return GraphState.model_validate(snapshot.values)

    def resume(self, thread_id: str) -> GraphState | None:
        """Resume a persisted LangGraph thread without replaying completed nodes."""

        with self._runtime() as graph:
            resumed = graph.invoke(None, self._config(thread_id), durability="sync")
        if not resumed:
            return None
        return GraphState.model_validate(resumed)

    def resume_and_reconcile(self, thread_id: str, state_store: StateStore) -> ReconciledRuntimeState:
        """Resume graph progress and align its marker to canonical domain state."""

        canonical_state = state_store.load_latest(thread_id)
        if canonical_state.run.run_id != thread_id:
            raise CheckpointError("thread_id must equal canonical MotionAgent run_id")

        with self._runtime() as graph:
            resumed = graph.invoke(None, self._config(thread_id), durability="sync")
            if not resumed:
                raise CheckpointError(f"no LangGraph checkpoint for thread_id={thread_id}")
            graph_state = GraphState.model_validate(resumed)
            reconciled = reconcile_graph_state(graph_state, canonical_state)
            if reconciled != graph_state:
                graph.update_state(
                    self._config(thread_id),
                    reconciled.model_dump(mode="python"),
                    as_node="persist_checkpoint",
                )

        return ReconciledRuntimeState(
            graph_state=reconciled,
            canonical_state=canonical_state,
        )

    def exists(self, thread_id: str) -> bool:
        return self.load(thread_id) is not None
