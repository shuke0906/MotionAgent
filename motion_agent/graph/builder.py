"""Real LangGraph Phase 2 topology builder."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from motion_agent.agent.planner import Planner
from motion_agent.agent.mock_tools import ToolRuntime
from motion_agent.graph.nodes.planner import make_planner_node
from motion_agent.graph.nodes.post_generation import make_diagnosis_node, make_tournament_node, make_verifier_node
from motion_agent.graph.nodes.tools import make_accept_node, make_action_node, make_stop_failed_node
from motion_agent.graph.routing import ROUTE_TO_NODE, route_planner_action
from motion_agent.graph.state import GraphState


PHASE2_NODE_NAMES = {
    "planner",
    "compile_motion",
    "retrieval",
    "constraint",
    "keyframe",
    "generation",
    "tournament",
    "verifier",
    "diagnosis",
    "accept",
    "stop_failed",
}


def build_motion_graph(*, checkpointer, runtime: ToolRuntime, planner: Planner):
    builder = StateGraph(GraphState)

    builder.add_node("planner", make_planner_node(state_store=runtime.state_store, planner=planner, runtime=runtime))
    builder.add_node("compile_motion", make_action_node(runtime, "compile_motion"))
    builder.add_node("retrieval", make_action_node(runtime, "retrieval"))
    builder.add_node("constraint", make_action_node(runtime, "constraint"))
    builder.add_node("keyframe", make_action_node(runtime, "keyframe"))
    builder.add_node("generation", make_action_node(runtime, "generation"))
    selector = "k1_selector" if runtime.phase11 else "tournament"
    builder.add_node(selector, make_tournament_node(runtime))
    builder.add_node("verifier", make_verifier_node(runtime))
    builder.add_node("diagnosis", make_diagnosis_node(runtime))
    builder.add_node("accept", make_accept_node(runtime))
    builder.add_node("stop_failed", make_stop_failed_node(runtime))

    builder.add_edge(START, "planner")
    builder.add_conditional_edges("planner", route_planner_action, ROUTE_TO_NODE)

    for node_name in ["compile_motion", "retrieval", "constraint", "keyframe"]:
        builder.add_edge(node_name, "planner")

    builder.add_edge("generation", selector)
    builder.add_edge(selector, "verifier")
    if runtime.phase11:
        def route_verification(graph_state):
            envelope = GraphState.model_validate(graph_state)
            summary = runtime.state_store.load_latest(envelope.run_id).evaluation.verification_summary
            return "planner" if summary and summary.status == "complete" and summary.overall_pass else "diagnosis"
        builder.add_conditional_edges("verifier", route_verification, {"planner": "planner", "diagnosis": "diagnosis"})
    else:
        builder.add_edge("verifier", "diagnosis")
    builder.add_edge("diagnosis", "planner")
    builder.add_edge("accept", END)
    builder.add_edge("stop_failed", END)

    return builder.compile(checkpointer=checkpointer, name="motion_agent_phase2")
