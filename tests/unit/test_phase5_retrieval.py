from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from motion_agent.agent.actions import PlannerAction
from motion_agent.agent.guards import PlannerGuardError, validate_planner_decision
from motion_agent.agent.mock_tools import execute_compile_motion, execute_retrieval
from motion_agent.retrieval import RetrievalRequest
from motion_agent.retrieval.id_mapping import mapping_coverage, validate_motion_mapping
from motion_agent.retrieval.retriever import build_default_retrieval_service
from motion_agent.state.artifacts import ArtifactStore
from motion_agent.state.events import EventLog
from motion_agent.state.reducer import StateReducer
from motion_agent.state.schemas import MotionSegmentSummary
from motion_agent.state.versioning import StateStore
from tests.unit.test_phase2_actions_guards import decision_for


def prepared_state():
    reducer = StateReducer()
    state = reducer.create_run("walk forward with an unusual asymmetric limping gait", run_id="phase5_unit")
    state = reducer.commit_motion_plan(
        state,
        motion_spec_id="spec_fixture",
        gem_text_condition_id="text_fixture",
        segments=[
            MotionSegmentSummary(
                segment_id=0,
                start_s=0.0,
                end_s=4.0,
                action="walk",
                body_parts=["full_body"],
                style=["limping", "asymmetric"],
            )
        ],
    )
    return state


class Phase5RetrievalUnitTests(unittest.TestCase):
    def test_common_motion_retrieval(self):
        result = self._retrieve("walk forward", retrieval_type="motion")
        self.assertEqual(result.status, "success")
        self.assertIn("walk", result.references[0].caption)

    def test_rare_style_retrieval(self):
        result = self._retrieve("asymmetric limping gait", retrieval_type="motion")
        self.assertEqual(result.status, "success")
        self.assertEqual(result.references[0].motion_id, "hml_train_walk_forward_limp")

    def test_body_part_specific_query(self):
        result = self._retrieve("wave right hand", retrieval_type="caption", purpose="prompt_grounding")
        self.assertEqual(result.status, "success")
        self.assertIn("right hand", result.references[0].caption)

    def test_direction_sensitive_query(self):
        result = self._retrieve("walk backward carefully", retrieval_type="motion")
        self.assertEqual(result.references[0].motion_id, "hml_train_walk_backward")

    def test_duration_sensitive_query_filters_extreme_mismatch(self):
        state = prepared_state()
        state.plan.segment_summaries[0].end_s = 0.2
        result = build_default_retrieval_service().retrieve_reference(
            state,
            RetrievalRequest(
                target_segment=0,
                query="walk forward with a limp",
                retrieval_type="motion",
                purpose="constraint_source",
                top_k=5,
            ),
        )
        self.assertNotEqual(result.status, "error")

    def test_caption_and_motion_retrieval(self):
        caption = self._retrieve("limping walk", retrieval_type="caption", purpose="prompt_grounding")
        motion = self._retrieve("limping walk", retrieval_type="motion")
        self.assertTrue(caption.references[0].caption_id)
        self.assertTrue(motion.references[0].motion_handle)

    def test_pose_and_trajectory_views(self):
        pose = self._retrieve("stable seated pose", retrieval_type="pose", purpose="keyframe_source")
        trajectory = self._retrieve("walk backward trajectory", retrieval_type="trajectory", purpose="constraint_source")
        self.assertTrue(pose.references[0].pose_handle)
        self.assertTrue(trajectory.references[0].trajectory_handle)
        self.assertIsNone(trajectory.references[0].pose_handle)

    def test_low_confidence_retrieval(self):
        state = prepared_state()
        state.plan.segment_summaries[0].action = "unknown_action"
        state.plan.segment_summaries[0].style = []
        result = build_default_retrieval_service().retrieve_reference(
            state,
            RetrievalRequest(
                target_segment=0,
                query="cartwheel moonwalk violin",
                retrieval_type="motion",
                purpose="motion_prior",
                top_k=5,
            ),
        )
        self.assertIn(result.status, {"empty", "low_confidence"})

    def test_train_test_leakage_guard(self):
        result = self._retrieve("jump upward", retrieval_type="motion")
        self.assertNotIn("hml_test_jump", [reference.motion_id for reference in result.references])

    def test_id_mapping_failure_is_reported(self):
        service = build_default_retrieval_service()
        coverage = mapping_coverage(service.index.id_map)
        self.assertGreater(coverage["unmapped"], 0)
        self.assertEqual(validate_motion_mapping(service.index.motions, service.index.id_map), [])

    def _retrieve(self, query: str, *, retrieval_type: str, purpose: str = "motion_prior"):
        return build_default_retrieval_service().retrieve_reference(
            prepared_state(),
            RetrievalRequest(
                target_segment=0,
                query=query,
                retrieval_type=retrieval_type,
                purpose=purpose,
                top_k=5,
            ),
        )


class Phase5RetrievalIntegrationTests(unittest.TestCase):
    def test_duplicate_query_guard(self):
        state = prepared_state()
        decision = decision_for(
            PlannerAction.RETRIEVE_REFERENCE,
            payload={
                "query": "limping gait",
                "retrieval_type": "motion",
                "purpose": "motion_prior",
                "top_k": 5,
            },
        )
        with tempfile.TemporaryDirectory() as tmp:
            runtime = _runtime(Path(tmp))
            runtime.artifact_store.put("motion_spec", {"schema": "fixture"}, artifact_id="spec_fixture")
            runtime.artifact_store.put("gem_text_condition", {"schema": "fixture"}, artifact_id="text_fixture")
            state = runtime.state_store.commit(state, expected_version=None, event_type="run_created")
            state = execute_retrieval(runtime, state, decision)
            with self.assertRaises(PlannerGuardError):
                validate_planner_decision(state, decision)

    def test_retrieval_to_compiler_caption_revision_preserves_continuity(self):
        with tempfile.TemporaryDirectory() as tmp:
            runtime = _runtime(Path(tmp))
            state = runtime.reducer.create_run("walk forward, wave the right hand, then sit down", run_id="phase5_flow")
            state = runtime.state_store.commit(state, expected_version=None, event_type="run_created")
            compile_initial = decision_for(
                PlannerAction.COMPILE_MOTION,
                payload={"mode": "initial", "focus": ["semantic", "timeline", "gem_caption"]},
            )
            state = execute_compile_motion(runtime, state, compile_initial)
            original_actions = [segment.action for segment in state.plan.segment_summaries]
            original_heading = [segment.heading_continuity for segment in state.plan.segment_summaries]
            retrieve = decision_for(
                PlannerAction.RETRIEVE_REFERENCE,
                payload={
                    "query": "asymmetric limping gait",
                    "retrieval_type": "caption",
                    "purpose": "prompt_grounding",
                    "top_k": 5,
                },
            ).model_copy(update={"target_segments": [0]})
            state = execute_retrieval(runtime, state, retrieve)
            revise = decision_for(
                PlannerAction.COMPILE_MOTION,
                payload={"mode": "revise", "focus": ["gem_caption"]},
            ).model_copy(update={"target_segments": [0]})
            state = execute_compile_motion(runtime, state, revise)
            revised_payload = runtime.artifact_store.get(state.plan.motion_spec_id)
            revised_segments = revised_payload["motion_spec"]["segments"]
            self.assertEqual([segment["action"] for segment in revised_segments], original_actions)
            self.assertEqual([summary.heading_continuity for summary in state.plan.segment_summaries], original_heading)
            self.assertIn("limping", revised_segments[0]["gem_caption"])


def _runtime(root: Path):
    from motion_agent.agent.mock_tools import ToolRuntime

    artifacts = ArtifactStore(root / "artifacts")
    return ToolRuntime(
        state_store=StateStore(root / "states", event_log=EventLog(root / "events"), artifact_store=artifacts),
        artifact_store=artifacts,
        reducer=StateReducer(),
    )


if __name__ == "__main__":
    unittest.main()
