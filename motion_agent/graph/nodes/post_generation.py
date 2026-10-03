"""Automatic post-generation subgraph nodes."""

from __future__ import annotations

from motion_agent.agent.mock_tools import execute_diagnosis, execute_tournament, execute_verifier, ToolRuntime
from motion_agent.graph.state import GraphState


def _post_node(runtime: ToolRuntime, node_name: str, executor):
    def node(graph_state: GraphState | dict) -> dict:
        envelope = GraphState.model_validate(graph_state)
        state = runtime.state_store.load_latest(envelope.run_id)
        new_state = executor(runtime, state)
        return envelope.model_copy(
            update={
                "state_version": new_state.state_version,
                "next_node": "planner",
                "committed_operation_ids": [
                    *envelope.committed_operation_ids,
                    f"{node_name}:{envelope.run_id}:v{new_state.state_version}",
                ],
            }
        ).model_dump(mode="python")

    return node


def make_tournament_node(runtime: ToolRuntime):
    return _post_node(runtime, "k1_selector" if runtime.phase11 else "tournament", runtime.selection_executor or execute_tournament)


def make_verifier_node(runtime: ToolRuntime):
    return _post_node(runtime, "verifier", runtime.verifier_executor or execute_verifier)


def make_diagnosis_node(runtime: ToolRuntime):
    return _post_node(runtime, "diagnosis", runtime.diagnosis_executor or execute_diagnosis)
