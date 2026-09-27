"""Deterministic LangGraph routing for Phase 2."""

from __future__ import annotations

from motion_agent.agent.actions import ACTION_TO_NODE, PlannerAction, PlannerDecision
from motion_agent.graph.state import GraphState


ROUTE_TO_NODE = {action.value: node for action, node in ACTION_TO_NODE.items()}


def route_planner_action(graph_state: GraphState | dict) -> str:
    state = GraphState.model_validate(graph_state)
    if state.planner_decision is None:
        raise ValueError("planner_decision is required for routing")
    decision = PlannerDecision.model_validate(state.planner_decision)
    return decision.action.value


def action_to_node(action: PlannerAction | str) -> str:
    planner_action = PlannerAction(action)
    return ACTION_TO_NODE[planner_action]

