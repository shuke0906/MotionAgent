"""LangGraph execution state and canonical-state reconciliation helpers.

GraphState is deliberately separate from MotionAgentState. It stores only
runtime/routing metadata and references that LangGraph may checkpoint.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field, model_validator

from motion_agent.state.schemas import MotionAgentState, StrictModel


class GraphState(StrictModel):
    run_id: str
    thread_id: str
    state_version: int
    next_node: str | None = None
    planner_decision: dict[str, Any] | None = None
    committed_operation_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_thread_identity(self) -> "GraphState":
        if self.thread_id != self.run_id:
            raise ValueError("thread_id must equal run_id")
        if len(self.committed_operation_ids) != len(set(self.committed_operation_ids)):
            raise ValueError("committed_operation_ids must be unique")
        return self


def graph_state_from_motion_state(
    state: MotionAgentState,
    *,
    next_node: str | None = None,
    committed_operation_ids: list[str] | None = None,
) -> GraphState:
    return GraphState(
        run_id=state.run.run_id,
        thread_id=state.run.run_id,
        state_version=state.state_version,
        next_node=next_node,
        committed_operation_ids=committed_operation_ids or [],
        metadata={"canonical_schema_version": state.schema_version},
    )


def reconcile_graph_state(graph_state: GraphState, state: MotionAgentState) -> GraphState:
    if graph_state.thread_id != state.run.run_id:
        raise ValueError("thread_id must equal MotionAgent run_id")
    metadata = dict(graph_state.metadata)
    metadata["canonical_schema_version"] = state.schema_version
    if graph_state.state_version != state.state_version:
        metadata["reconciled_from_state_version"] = graph_state.state_version
    return graph_state.model_copy(
        update={
            "run_id": state.run.run_id,
            "thread_id": state.run.run_id,
            "state_version": state.state_version,
            "metadata": metadata,
        }
    )
