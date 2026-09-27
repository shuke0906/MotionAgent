from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from motion_agent.common.errors import MissingArtifactError, StateConflictError, StateInvariantError
from motion_agent.graph.checkpoint import LocalGraphCheckpointer
from motion_agent.graph.state import graph_state_from_motion_state, reconcile_graph_state
from motion_agent.state.artifacts import ArtifactStore
from motion_agent.state.context_builder import PlannerContextBuilder
from motion_agent.state.events import EventLog
from motion_agent.state.invariants import validate_state_invariants
from motion_agent.state.reducer import StateReducer
from motion_agent.state.schemas import (
    ConstraintSummary,
    DiagnosisSummary,
    KeyframeSummary,
    MotionSegmentSummary,
    RepairProposalSummary,
    VerificationSummary,
)
from motion_agent.state.versioning import StateStore


def make_runtime(root: Path):
    artifacts = ArtifactStore(root / "artifacts")
    events = EventLog(root / "events")
    store = StateStore(root / "states", event_log=events, artifact_store=artifacts)
    reducer = StateReducer()
    return reducer, artifacts, events, store


def commit_initial(store, reducer):
    state = reducer.create_run("walk forward then sit down", run_id="run_phase1_test", total_frames=180)
    state.control.budgets.iterations_left = 100
    return store.commit(state, expected_version=None, event_type="run_created")


def make_plan(reducer, artifacts, state):
    spec = artifacts.put("motion_spec", {"segments": ["walk", "sit"]})
    text = artifacts.put("gem_text_condition", {"captions": ["walk forward", "sit down"]})
    segments = [
        MotionSegmentSummary(segment_id=0, start_s=0.0, end_s=3.0, action="walk", body_parts=["full_body"]),
        MotionSegmentSummary(segment_id=1, start_s=3.0, end_s=6.0, action="sit", body_parts=["full_body"]),
    ]
    return reducer.commit_motion_plan(
        state,
        motion_spec_id=spec.artifact_id,
        gem_text_condition_id=text.artifact_id,
        segments=segments,
    )


class Phase1StateTests(unittest.TestCase):
    def test_basic_state_lifecycle(self):
        with tempfile.TemporaryDirectory() as tmp:
            reducer, artifacts, events, store = make_runtime(Path(tmp))
            state = commit_initial(store, reducer)

            state = store.commit(make_plan(reducer, artifacts, state), expected_version=0, event_type="plan_committed")

            constraint = artifacts.put("constraint", {"type": "contact"})
            constraint_spec = artifacts.put("verification_spec", {"threshold": 0.05})
            state = reducer.add_or_replace_constraint(
                state,
                ConstraintSummary(
                    constraint_id=constraint.artifact_id,
                    segment_id=1,
                    constraint_type="contact",
                    verification_spec_id=constraint_spec.artifact_id,
                ),
            )
            state = store.commit(state, expected_version=1, event_type="constraint_added")
            self.assertEqual([c.status for c in state.conditions.constraints], ["active"])

            replacement = artifacts.put("constraint", {"type": "contact", "target": "table"})
            replacement_spec = artifacts.put("verification_spec", {"threshold": 0.03})
            state = reducer.add_or_replace_constraint(
                state,
                ConstraintSummary(
                    constraint_id=replacement.artifact_id,
                    segment_id=1,
                    constraint_type="contact",
                    verification_spec_id=replacement_spec.artifact_id,
                ),
            )
            state = store.commit(state, expected_version=2, event_type="constraint_replaced")
            self.assertEqual([c.status for c in state.conditions.constraints], ["superseded", "active"])

            state = reducer.remove_constraint(state, replacement.artifact_id)
            state = store.commit(state, expected_version=3, event_type="constraint_removed")
            self.assertEqual(state.conditions.constraints[-1].status, "removed")

            keyframe = artifacts.put("keyframe", {"pose": "seated"})
            keyframe_spec = artifacts.put("keyframe_verification_spec", {"position_threshold": 0.1})
            state = reducer.add_or_replace_keyframe(
                state,
                KeyframeSummary(
                    keyframe_id=keyframe.artifact_id,
                    segment_id=1,
                    target_time_s=5.8,
                    source_type="retrieval",
                    verification_spec_id=keyframe_spec.artifact_id,
                ),
            )
            state = store.commit(state, expected_version=4, event_type="keyframe_added")

            replacement_keyframe = artifacts.put("keyframe", {"pose": "stable seated"})
            replacement_keyframe_spec = artifacts.put("keyframe_verification_spec", {"position_threshold": 0.08})
            state = reducer.add_or_replace_keyframe(
                state,
                KeyframeSummary(
                    keyframe_id=replacement_keyframe.artifact_id,
                    segment_id=1,
                    target_time_s=5.9,
                    source_type="retrieval",
                    verification_spec_id=replacement_keyframe_spec.artifact_id,
                ),
            )
            state = store.commit(state, expected_version=5, event_type="keyframe_replaced")
            self.assertEqual([k.status for k in state.conditions.keyframes], ["superseded", "active"])

            state = reducer.remove_keyframe(state, replacement_keyframe.artifact_id)
            state = store.commit(state, expected_version=6, event_type="keyframe_removed")
            self.assertEqual(state.conditions.keyframes[-1].status, "removed")

            generation = artifacts.put("generation", {"status": "success"})
            cand_a = artifacts.put("candidate", {"seed": 1})
            cand_b = artifacts.put("candidate", {"seed": 2})
            state = reducer.commit_generation_result(
                state,
                generation_id=generation.artifact_id,
                candidate_ids=[cand_a.artifact_id, cand_b.artifact_id],
            )
            state = store.commit(state, expected_version=7, event_type="generation_committed")

            tournament = artifacts.put("tournament", {"winner": cand_b.artifact_id})
            state = reducer.set_champion(state, cand_b.artifact_id, tournament_id=tournament.artifact_id)
            state = store.commit(state, expected_version=8, event_type="champion_set")
            self.assertEqual(state.generation.champion_candidate_id, cand_b.artifact_id)

            verification = artifacts.put("verification", {"overall_pass": False})
            state = reducer.commit_verification_report(
                state,
                verification_id=verification.artifact_id,
                summary=VerificationSummary(
                    status="complete",
                    overall_pass=False,
                    failed_check_ids=["semantic_core"],
                    critical_failures=["SEMANTIC_FAILURE"],
                    affected_segments=[0],
                    diagnostic_codes=["SEMANTIC_FAILURE"],
                ),
            )
            state = store.commit(state, expected_version=9, event_type="verification_committed")

            diagnosis = artifacts.put("diagnosis", {"root": "caption_mismatch"})
            state = reducer.commit_diagnosis_result(
                state,
                diagnosis_id=diagnosis.artifact_id,
                summary=DiagnosisSummary(
                    status="diagnosed",
                    primary_failures=["SEMANTIC_FAILURE"],
                    root_causes=["CAPTION_MISMATCH"],
                    target_segments=[0],
                    proposal_summaries=[
                        RepairProposalSummary(
                            proposal_id="repair_001",
                            action="COMPILE_MOTION",
                            reason_code="SEMANTIC_FAILURE",
                            target_segments=[0],
                            confidence=0.9,
                        )
                    ],
                ),
            )
            state = store.commit(state, expected_version=10, event_type="diagnosis_committed")
            self.assertEqual(state.evaluation.latest_diagnosis_id, diagnosis.artifact_id)

            verification_pass = artifacts.put("verification", {"overall_pass": True})
            state = reducer.commit_verification_report(
                state,
                verification_id=verification_pass.artifact_id,
                summary=VerificationSummary(status="complete", overall_pass=True, passed_check_ids=["semantic_core"]),
            )
            state = store.commit(state, expected_version=11, event_type="verification_passed")
            state = reducer.accept(state)
            state = store.commit(state, expected_version=12, event_type="accepted")

            self.assertTrue(state.control.accepted)
            self.assertEqual(state.run.status, "accepted")
            self.assertEqual([event.state_version_after for event in events.list(state.run.run_id)], list(range(14)))

    def test_state_invariants_reject_invalid_states(self):
        with tempfile.TemporaryDirectory() as tmp:
            reducer, artifacts, _, store = make_runtime(Path(tmp))
            state = commit_initial(store, reducer)

            invalid_budget = state.model_copy(deep=True)
            invalid_budget.control.budgets.generations_left = -1
            invalid_budget.state_version += 1
            with self.assertRaisesRegex(StateInvariantError, "negative"):
                validate_state_invariants(invalid_budget, previous_version=state.state_version)

            accepted_without_verification = state.model_copy(deep=True)
            accepted_without_verification.control.accepted = True
            accepted_without_verification.state_version += 1
            with self.assertRaisesRegex(StateInvariantError, "accepted=true"):
                validate_state_invariants(accepted_without_verification, previous_version=state.state_version)

            missing_artifact_state = state.model_copy(deep=True)
            missing_artifact_state.plan.motion_spec_id = "spec_missing"
            missing_artifact_state.plan.status = "ready"
            missing_artifact_state.state_version += 1
            with self.assertRaises(MissingArtifactError):
                store.commit(missing_artifact_state, expected_version=0, event_type="bad_missing_artifact")

            cand = artifacts.put("candidate", {"seed": 1})
            generation = artifacts.put("generation", {"status": "success"})
            invalid_champion = reducer.commit_generation_result(
                state,
                generation_id=generation.artifact_id,
                candidate_ids=[cand.artifact_id],
            )
            invalid_champion.generation.champion_candidate_id = "cand_missing"
            with self.assertRaisesRegex(StateInvariantError, "champion"):
                validate_state_invariants(invalid_champion, artifact_store=artifacts, previous_version=state.state_version)

    def test_tensor_isolation_artifact_state_and_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            reducer, artifacts, _, store = make_runtime(Path(tmp))
            state = commit_initial(store, reducer)

            motion_tensor = torch.zeros(12, 151)
            candidate = artifacts.put("candidate", motion_tensor, metadata={"shape": [12, 151]})
            generation = artifacts.put("generation", {"candidate_ids": [candidate.artifact_id]})
            state = reducer.commit_generation_result(
                state,
                generation_id=generation.artifact_id,
                candidate_ids=[candidate.artifact_id],
            )
            state = store.commit(state, expected_version=0, event_type="generation_with_tensor_artifact")

            self.assertTrue(artifacts.exists(candidate.artifact_id))
            self.assertEqual(tuple(artifacts.get(candidate.artifact_id).shape), (12, 151))
            state_json = state.model_dump_json()
            self.assertNotIn("tensor", state_json.lower())
            self.assertNotIn("[0.0", state_json)

            context = PlannerContextBuilder().build(state)
            context_json = context.model_dump_json()
            self.assertIn(candidate.artifact_id, context_json)
            self.assertNotIn("tensor", context_json.lower())
            self.assertNotIn("[0.0", context_json)

    def test_optimistic_concurrency_allows_exactly_one_commit(self):
        with tempfile.TemporaryDirectory() as tmp:
            reducer, artifacts, _, store = make_runtime(Path(tmp))
            state = commit_initial(store, reducer)
            update_a = make_plan(reducer, artifacts, state)
            update_b = make_plan(reducer, artifacts, state)
            store.commit(update_a, expected_version=0, event_type="plan_a")
            with self.assertRaises(StateConflictError):
                store.commit(update_b, expected_version=0, event_type="plan_b")

    def test_event_log_append_only_and_ordered(self):
        with tempfile.TemporaryDirectory() as tmp:
            reducer, artifacts, events, store = make_runtime(Path(tmp))
            state = commit_initial(store, reducer)
            state = store.commit(make_plan(reducer, artifacts, state), expected_version=0, event_type="plan_committed")
            logged = events.list(state.run.run_id)
            self.assertEqual([event.event_type for event in logged], ["run_created", "plan_committed"])
            self.assertEqual([event.state_version_before for event in logged], [-1, 0])
            self.assertEqual([event.state_version_after for event in logged], [0, 1])

    def test_local_graph_checkpoint_resume_and_reconcile(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reducer, artifacts, _, store = make_runtime(root)
            state = commit_initial(store, reducer)
            state = store.commit(make_plan(reducer, artifacts, state), expected_version=0, event_type="plan_committed")

            checkpointer = LocalGraphCheckpointer(root / "checkpoints")
            graph_state = graph_state_from_motion_state(state, next_node="planner")
            checkpointer.save(graph_state)

            restarted = LocalGraphCheckpointer(root / "checkpoints")
            recovered = restarted.load(state.run.run_id)
            self.assertEqual(recovered, graph_state)
            self.assertEqual(recovered.thread_id, state.run.run_id)

            canonical = store.load_latest(state.run.run_id)
            reconciled = reconcile_graph_state(recovered, canonical)
            self.assertEqual(reconciled.state_version, canonical.state_version)

    def test_thread_isolation_between_run_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reducer, _, _, store = make_runtime(root)
            state_a = store.commit(reducer.create_run("walk", run_id="run_A"), expected_version=None, event_type="run_created")
            state_b = store.commit(reducer.create_run("sit", run_id="run_B"), expected_version=None, event_type="run_created")

            checkpointer = LocalGraphCheckpointer(root / "checkpoints")
            checkpointer.save(graph_state_from_motion_state(state_a, next_node="planner"))
            checkpointer.save(graph_state_from_motion_state(state_b, next_node="generation"))

            self.assertEqual(checkpointer.load("run_A").next_node, "planner")
            self.assertEqual(checkpointer.load("run_B").next_node, "generation")
            self.assertEqual(store.load_latest("run_A").task.original_request, "walk")
            self.assertEqual(store.load_latest("run_B").task.original_request, "sit")

    def test_context_growth_is_bounded_while_event_log_grows(self):
        with tempfile.TemporaryDirectory() as tmp:
            reducer, artifacts, events, store = make_runtime(Path(tmp))
            state = reducer.create_run("long repair sequence", run_id="run_context_growth")
            state.control.budgets.iterations_left = 100
            state = store.commit(state, expected_version=None, event_type="run_created")

            for index in range(30):
                handle = artifacts.put("constraint", {"index": index})
                spec = artifacts.put("verification_spec", {"index": index})
                state = reducer.add_or_replace_constraint(
                    state,
                    ConstraintSummary(
                        constraint_id=handle.artifact_id,
                        segment_id=index,
                        constraint_type=f"target_{index}",
                        verification_spec_id=spec.artifact_id,
                    ),
                )
                state = store.commit(state, expected_version=index, event_type="constraint_added")

            context = PlannerContextBuilder(max_recent_actions=6, max_items_per_kind=3).build(state)
            self.assertEqual(len(events.list(state.run.run_id)), 31)
            self.assertEqual(len(context.history.recent_actions), 6)
            self.assertEqual(len(context.conditions.constraints), 3)
            self.assertLess(len(context.model_dump_json()), 8000)
