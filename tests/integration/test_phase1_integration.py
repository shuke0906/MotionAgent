from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from motion_agent.graph.checkpoint import LocalGraphCheckpointer
from motion_agent.graph.state import graph_state_from_motion_state
from motion_agent.state.artifacts import ArtifactStore
from motion_agent.state.context_builder import PlannerContextBuilder
from motion_agent.state.events import EventLog
from motion_agent.state.reducer import StateReducer
from motion_agent.state.schemas import ConstraintSummary, MotionSegmentSummary
from motion_agent.state.versioning import StateStore


class Phase1IntegrationTests(unittest.TestCase):
    def test_state_artifact_event_checkpoint_integration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = ArtifactStore(root / "artifacts")
            events = EventLog(root / "events")
            store = StateStore(root / "states", event_log=events, artifact_store=artifacts)
            reducer = StateReducer()

            state = reducer.create_run("walk forward", run_id="run_integration")
            state = store.commit(state, expected_version=None, event_type="run_created")

            spec = artifacts.put("motion_spec", {"task": "walk forward"})
            text = artifacts.put("gem_text_condition", {"captions": ["walk forward"]})
            state = reducer.commit_motion_plan(
                state,
                motion_spec_id=spec.artifact_id,
                gem_text_condition_id=text.artifact_id,
                segments=[
                    MotionSegmentSummary(
                        segment_id=0,
                        start_s=0.0,
                        end_s=6.0,
                        action="walk",
                        body_parts=["full_body"],
                    )
                ],
            )
            state = store.commit(state, expected_version=0, event_type="plan_committed")

            checkpoint = LocalGraphCheckpointer(root / "checkpoints")
            checkpoint.save(graph_state_from_motion_state(state, next_node="planner"))

            resumed_graph_state = checkpoint.load("run_integration")
            resumed_state = store.load_latest("run_integration")
            context = PlannerContextBuilder().build(resumed_state)

            self.assertEqual(resumed_graph_state.thread_id, "run_integration")
            self.assertEqual(resumed_graph_state.state_version, resumed_state.state_version)
            self.assertEqual(context.plan.status, "ready")
            self.assertEqual(context.task.original_request, "walk forward")
            self.assertEqual(
                [event.event_type for event in events.list("run_integration")],
                ["run_created", "plan_committed"],
            )

    def test_real_langgraph_restart_reconciles_without_replaying_committed_operation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = ArtifactStore(root / "artifacts")
            events = EventLog(root / "events")
            store = StateStore(root / "states", event_log=events, artifact_store=artifacts)
            reducer = StateReducer()

            state = reducer.create_run("walk and stop", run_id="run_restart")
            state = store.commit(state, expected_version=None, event_type="run_created")

            checkpoint = LocalGraphCheckpointer(root / "checkpoints")
            checkpoint.checkpoint_canonical_state(
                state,
                next_node="planner",
                committed_operation_ids=["run_created:run_restart"],
            )

            spec = artifacts.put("motion_spec", {"task": "walk and stop"})
            text = artifacts.put("gem_text_condition", {"captions": ["walk", "stop"]})
            state = reducer.commit_motion_plan(
                state,
                motion_spec_id=spec.artifact_id,
                gem_text_condition_id=text.artifact_id,
                segments=[
                    MotionSegmentSummary(
                        segment_id=0,
                        start_s=0.0,
                        end_s=6.0,
                        action="walk_and_stop",
                        body_parts=["full_body"],
                    )
                ],
            )
            state = store.commit(state, expected_version=0, event_type="plan_committed")
            checkpoint.checkpoint_canonical_state(
                state,
                next_node="planner",
                committed_operation_ids=[
                    "run_created:run_restart",
                    "compile_motion:run_restart:v1",
                ],
            )

            constraint = artifacts.put("constraint", {"type": "stop_position"})
            verification = artifacts.put("verification_spec", {"tolerance": 0.05})
            state = reducer.add_or_replace_constraint(
                state,
                ConstraintSummary(
                    constraint_id=constraint.artifact_id,
                    segment_id=0,
                    constraint_type="stop_position",
                    verification_spec_id=verification.artifact_id,
                ),
            )
            state = store.commit(state, expected_version=1, event_type="constraint_committed")

            event_count_before_restart = len(events.list("run_restart"))
            action_count_before_restart = len(state.history.recent_actions)

            restarted_events = EventLog(root / "events")
            restarted_store = StateStore(
                root / "states",
                event_log=restarted_events,
                artifact_store=ArtifactStore(root / "artifacts"),
            )
            restarted_checkpoint = LocalGraphCheckpointer(root / "checkpoints")
            recovered = restarted_checkpoint.resume_and_reconcile("run_restart", restarted_store)

            self.assertEqual(recovered.canonical_state.state_version, 2)
            self.assertEqual(recovered.graph_state.state_version, 2)
            self.assertEqual(recovered.graph_state.metadata["reconciled_from_state_version"], 1)
            self.assertEqual(
                recovered.graph_state.committed_operation_ids.count("compile_motion:run_restart:v1"),
                1,
            )

            resumed_again = restarted_checkpoint.resume_and_reconcile("run_restart", restarted_store)
            self.assertEqual(resumed_again.graph_state.state_version, 2)
            self.assertEqual(len(restarted_events.list("run_restart")), event_count_before_restart)
            self.assertEqual(len(resumed_again.canonical_state.history.recent_actions), action_count_before_restart)
            self.assertEqual(
                resumed_again.canonical_state.history.repeated_action_counts["COMPILE_MOTION"],
                1,
            )

    def test_real_langgraph_sqlite_keeps_recreated_threads_isolated(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = StateStore(root / "states", event_log=EventLog(root / "events"))
            reducer = StateReducer()
            state_a = store.commit(
                reducer.create_run("walk", run_id="thread_A"),
                expected_version=None,
                event_type="run_created",
            )
            state_b = store.commit(
                reducer.create_run("sit", run_id="thread_B"),
                expected_version=None,
                event_type="run_created",
            )

            checkpoint = LocalGraphCheckpointer(root / "checkpoints")
            checkpoint.checkpoint_canonical_state(
                state_a,
                next_node="planner",
                committed_operation_ids=["create:A"],
            )
            checkpoint.checkpoint_canonical_state(
                state_b,
                next_node="generation",
                committed_operation_ids=["create:B"],
            )

            restarted = LocalGraphCheckpointer(root / "checkpoints")
            recovered_a = restarted.resume_and_reconcile("thread_A", store)
            recovered_b = restarted.resume_and_reconcile("thread_B", store)

            self.assertEqual(recovered_a.graph_state.thread_id, "thread_A")
            self.assertEqual(recovered_b.graph_state.thread_id, "thread_B")
            self.assertEqual(recovered_a.graph_state.next_node, "planner")
            self.assertEqual(recovered_b.graph_state.next_node, "generation")
            self.assertEqual(recovered_a.graph_state.committed_operation_ids, ["create:A"])
            self.assertEqual(recovered_b.graph_state.committed_operation_ids, ["create:B"])
            self.assertEqual(recovered_a.canonical_state.task.original_request, "walk")
            self.assertEqual(recovered_b.canonical_state.task.original_request, "sit")
