from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from motion_agent.agent.planner import ScriptedPlanner
from motion_agent.app.orchestrator import MotionAgentOrchestrator


def compile_decision():
    return {
        "action": "COMPILE_MOTION",
        "reason_code": "INITIAL_COMPILE",
        "target_segments": None,
        "reason_summary": "Compile the user request.",
        "payload": {"mode": "initial", "focus": ["semantic", "timeline", "gem_caption"]},
    }


def generate_decision():
    return {
        "action": "GENERATE",
        "reason_code": "CONDITIONS_READY",
        "target_segments": None,
        "reason_summary": "Generate mock candidates.",
        "payload": {"strategy": "normal", "scope": "full", "num_candidates": 2, "reward_targets": []},
    }


def accept_decision():
    return {
        "action": "ACCEPT",
        "reason_code": "ALL_REQUIRED_CHECKS_PASSED",
        "target_segments": None,
        "reason_summary": "Accept verified motion.",
        "payload": {},
    }


class Phase3CompileNodeTests(unittest.TestCase):
    def test_compile_motion_node_commits_real_compiler_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            planner = ScriptedPlanner([compile_decision()])
            orch = MotionAgentOrchestrator(Path(tmp), planner)
            state = orch.create_run("wave the right hand three times and sit down", run_id="phase3_compile")
            compiled = orch.invoke(state, interrupt_after=["compile_motion"])

            self.assertEqual(compiled.plan.status, "ready")
            self.assertEqual(len(compiled.plan.segment_summaries), 2)
            self.assertEqual(compiled.plan.segment_summaries[0].action, "wave")
            self.assertEqual(compiled.plan.segment_summaries[0].repetition, 3)
            self.assertEqual(compiled.plan.segment_summaries[1].action, "sit_down")
            self.assertIsNotNone(compiled.plan.segment_summaries[0].heading_continuity)
            self.assertEqual(compiled.plan.segment_summaries[0].heading_continuity.mode, "free")
            self.assertEqual(compiled.plan.segment_summaries[1].heading_continuity.mode, "free")

            spec_payload = orch.artifact_store.get(compiled.plan.motion_spec_id)
            text_payload = orch.artifact_store.get(compiled.plan.gem_text_condition_id)
            self.assertEqual(spec_payload["schema"], "MotionSpecification")
            self.assertEqual(text_payload["schema"], "GEMTextCondition")
            self.assertIn("motion_spec", spec_payload)
            self.assertIn("gem_text_condition", text_payload)
            self.assertNotEqual(spec_payload["schema"], "MockMotionSpecification")

    def test_phase2_orchestrator_regression_still_accepts(self):
        with tempfile.TemporaryDirectory() as tmp:
            planner = ScriptedPlanner([compile_decision(), generate_decision(), accept_decision()])
            orch = MotionAgentOrchestrator(Path(tmp), planner)
            final = orch.run("walk forward", run_id="phase3_regression")
            self.assertEqual(final.run.status, "accepted")
            self.assertEqual(final.plan.segment_summaries[0].action, "walk")
            events = [event.event_type for event in orch.event_log.list("phase3_regression")]
            self.assertEqual(
                events,
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


if __name__ == "__main__":
    unittest.main()
