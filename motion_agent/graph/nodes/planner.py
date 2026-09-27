"""Planner node adapter."""

from __future__ import annotations

from motion_agent.agent.context_builder import PlannerContextBuilder
from motion_agent.agent.guards import validate_planner_decision
from motion_agent.agent.planner import Planner
from motion_agent.graph.routing import action_to_node
from motion_agent.graph.state import GraphState
from motion_agent.state.versioning import StateStore


def make_planner_node(*, state_store: StateStore, planner: Planner, context_builder: PlannerContextBuilder | None = None):
    builder = context_builder or PlannerContextBuilder()

    def planner_node(graph_state: GraphState | dict) -> dict:
        envelope = GraphState.model_validate(graph_state)
        state = state_store.load_latest(envelope.run_id)
        context = builder.build(state)
        decision = planner.step(context, state)
        validate_planner_decision(state, decision)
        metadata = dict(envelope.metadata)
        metadata["planner_context_size"] = len(context.model_dump_json())
        metadata["planner_decisions"] = int(metadata.get("planner_decisions", 0)) + 1
        return envelope.model_copy(
            update={
                "state_version": state.state_version,
                "planner_decision": decision.model_dump(mode="json"),
                "next_node": action_to_node(decision.action),
                "metadata": metadata,
            }
        ).model_dump(mode="python")

    return planner_node

