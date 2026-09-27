"""Executable action and terminal node adapters."""

from __future__ import annotations

from motion_agent.agent.actions import PlannerDecision
from motion_agent.agent.mock_tools import ACTION_EXECUTORS, ToolRuntime
from motion_agent.graph.state import GraphState


def _commit_envelope(envelope: GraphState, *, node_name: str, state_version: int) -> dict:
    operation_id = f"{node_name}:{envelope.run_id}:v{state_version}"
    committed = [*envelope.committed_operation_ids, operation_id]
    return envelope.model_copy(
        update={
            "state_version": state_version,
            "next_node": "planner",
            "committed_operation_ids": committed,
        }
    ).model_dump(mode="python")


def make_action_node(runtime: ToolRuntime, node_name: str):
    def action_node(graph_state: GraphState | dict) -> dict:
        envelope = GraphState.model_validate(graph_state)
        if envelope.planner_decision is None:
            raise ValueError(f"{node_name} requires planner_decision")
        decision = PlannerDecision.model_validate(envelope.planner_decision)
        state = runtime.state_store.load_latest(envelope.run_id)
        executor = ACTION_EXECUTORS[decision.action]
        new_state = executor(runtime, state, decision)
        return _commit_envelope(envelope, node_name=node_name, state_version=new_state.state_version)

    return action_node


def make_accept_node(runtime: ToolRuntime):
    def accept_node(graph_state: GraphState | dict) -> dict:
        envelope = GraphState.model_validate(graph_state)
        state = runtime.state_store.load_latest(envelope.run_id)
        new_state = runtime.reducer.accept(state)
        new_state = runtime.state_store.commit(
            new_state,
            expected_version=state.state_version,
            event_type="accepted",
        )
        return envelope.model_copy(
            update={
                "state_version": new_state.state_version,
                "next_node": None,
                "committed_operation_ids": [
                    *envelope.committed_operation_ids,
                    f"accept:{envelope.run_id}:v{new_state.state_version}",
                ],
            }
        ).model_dump(mode="python")

    return accept_node


def make_stop_failed_node(runtime: ToolRuntime):
    def stop_failed_node(graph_state: GraphState | dict) -> dict:
        envelope = GraphState.model_validate(graph_state)
        state = runtime.state_store.load_latest(envelope.run_id)
        reason = "failed"
        if state.evaluation.diagnosis_summary and state.evaluation.diagnosis_summary.terminal_hint:
            reason = state.evaluation.diagnosis_summary.terminal_hint
        new_state = runtime.reducer.stop_failed(state, reason=reason)
        new_state = runtime.state_store.commit(
            new_state,
            expected_version=state.state_version,
            event_type="stopped_failed",
        )
        return envelope.model_copy(
            update={
                "state_version": new_state.state_version,
                "next_node": None,
                "committed_operation_ids": [
                    *envelope.committed_operation_ids,
                    f"stop_failed:{envelope.run_id}:v{new_state.state_version}",
                ],
            }
        ).model_dump(mode="python")

    return stop_failed_node

