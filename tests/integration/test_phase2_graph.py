from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from motion_agent.app.orchestrator import MotionAgentOrchestrator
from motion_agent.agent.actions import PlannerAction
from motion_agent.agent.planner import ScriptedPlanner
from motion_agent.graph.routing import ROUTE_TO_NODE


def compile_decision():
    return {
        "action": "COMPILE_MOTION",
        "reason_code": "INITIAL_COMPILE",
        "target_segments": None,
        "reason_summary": "Compile the user request.",
        "payload": {"mode": "initial", "focus": ["semantic", "timeline", "gem_caption"]},
    }


def generate_decision(*, fail: bool = False):
    return {
        "action": "GENERATE",
        "reason_code": "CONDITIONS_READY",
        "target_segments": None,
        "reason_summary": "Generate mock candidates.",
        "payload": {
            "strategy": "normal",
            "scope": "full",
            "num_candidates": 2,
            "reward_targets": ["mock_fail"] if fail else [],
        },
    }


def accept_decision():
    return {
        "action": "ACCEPT",
        "reason_code": "ALL_REQUIRED_CHECKS_PASSED",
        "target_segments": None,
        "reason_summary": "Accept verified motion.",
        "payload": {},
    }


def corrective_constraint_decision():
    return {
        "action": "BUILD_CONSTRAINT",
        "reason_code": "CONSTRAINT_FAILURE",
        "target_segments": [0],
        "reason_summary": "Add a corrective mock constraint.",
        "payload": {"mode": "add", "constraint_type": "contact", "body_part": "right_wrist", "strength": "soft"},
    }


def stop_failed_decision():
    return {
        "action": "STOP_FAILED",
        "reason_code": "BUDGET_EXHAUSTED",
        "target_segments": None,
        "reason_summary": "Stop after budget exhaustion.",
        "payload": {},
    }


class Phase2GraphTests(unittest.TestCase):
    def test_graph_topology_and_routing_table(self):
        expected = {
            "COMPILE_MOTION": "compile_motion",
            "RETRIEVE_REFERENCE": "retrieval",
            "BUILD_CONSTRAINT": "constraint",
            "BUILD_KEYFRAME": "keyframe",
            "GENERATE": "generation",
            "ACCEPT": "accept",
            "STOP_FAILED": "stop_failed",
        }
        self.assertEqual(ROUTE_TO_NODE, expected)

        with tempfile.TemporaryDirectory() as tmp:
            orch = MotionAgentOrchestrator(Path(tmp), ScriptedPlanner([]))
            with orch._graph() as graph:
                rendered = graph.get_graph()
            edges = {(edge.source, edge.target, edge.data) for edge in rendered.edges}
            self.assertIn(("__start__", "planner", None), edges)
            self.assertIn(("generation", "tournament", None), edges)
            self.assertIn(("tournament", "verifier", None), edges)
            self.assertIn(("verifier", "diagnosis", None), edges)
            self.assertIn(("diagnosis", "planner", None), edges)
            self.assertIn(("accept", "__end__", None), edges)
            self.assertIn(("stop_failed", "__end__", None), edges)
            for action, node in expected.items():
                self.assertIn(("planner", node, action), edges)

    def test_mock_e2e_success_reaches_accept_with_automatic_post_generation(self):
        with tempfile.TemporaryDirectory() as tmp:
            planner = ScriptedPlanner([compile_decision(), generate_decision(), accept_decision()])
            orch = MotionAgentOrchestrator(Path(tmp), planner)
            state = orch.run("user walks forward", run_id="run_success")
            self.assertEqual(state.run.status, "accepted")
            self.assertTrue(state.control.accepted)
            self.assertEqual(
                [event.event_type for event in orch.event_log.list("run_success")],
                [
                    "run_created",
                    "compile_motion_committed",
                    "generation_committed",
                    "tournament_committed",
                    "verification_committed",
                    "diagnosis_committed",
                    "accepted",
                ],
            )
            self.assertEqual(planner.decisions_made, 3)

    def test_failure_recovery_loop_generates_again_then_accepts(self):
        with tempfile.TemporaryDirectory() as tmp:
            planner = ScriptedPlanner(
                [
                    compile_decision(),
                    generate_decision(fail=True),
                    corrective_constraint_decision(),
                    generate_decision(),
                    accept_decision(),
                ]
            )
            orch = MotionAgentOrchestrator(Path(tmp), planner)
            state = orch.run("user walks forward with recovery", run_id="run_recovery")
            self.assertEqual(state.run.status, "accepted")
            self.assertEqual(state.generation.generation_round, 2)
            self.assertEqual(state.evaluation.verification_summary.overall_pass, True)
            events = [event.event_type for event in orch.event_log.list("run_recovery")]
            self.assertEqual(events.count("diagnosis_committed"), 2)
            self.assertIn("constraint_committed", events)

    def test_stop_failed_after_budget_exhaustion_reaches_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            planner = ScriptedPlanner([compile_decision(), generate_decision(fail=True), stop_failed_decision()])
            orch = MotionAgentOrchestrator(Path(tmp), planner)
            initial = orch.create_run("persistent failure", run_id="run_stop", generations_left=1)
            state = orch.invoke(initial)
            self.assertEqual(state.run.status, "failed")
            self.assertEqual(state.control.terminal_reason, "BUDGET_EXHAUSTED")
            self.assertFalse(state.control.accepted)

    def test_checkpoint_resume_continues_from_interrupted_generation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = MotionAgentOrchestrator(root, ScriptedPlanner([compile_decision(), generate_decision()]))
            initial = first.create_run("walk forward", run_id="run_resume")
            partial = first.invoke(initial, interrupt_after=["generation"])
            self.assertEqual(partial.generation.generation_round, 1)
            self.assertIsNone(partial.generation.champion_candidate_id)

            restarted = MotionAgentOrchestrator(root, ScriptedPlanner([accept_decision()]))
            final = restarted.resume("run_resume")
            self.assertEqual(final.run.status, "accepted")
            self.assertEqual(final.generation.generation_round, 1)
            events = [event.event_type for event in restarted.event_log.list("run_resume")]
            self.assertEqual(events.count("generation_committed"), 1)
            self.assertIn("tournament_committed", events)
            self.assertIn("accepted", events)


if __name__ == "__main__":
    unittest.main()

