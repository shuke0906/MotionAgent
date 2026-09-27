from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from motion_agent.agent.actions import PlannerAction, PlannerDecision, parse_planner_decision
from motion_agent.agent.guards import PlannerGuardError, validate_planner_decision
from motion_agent.state.artifacts import ArtifactStore
from motion_agent.state.events import EventLog
from motion_agent.state.reducer import StateReducer
from motion_agent.state.schemas import MotionSegmentSummary, VerificationSummary
from motion_agent.state.versioning import StateStore


VALID_DECISIONS = {
    PlannerAction.COMPILE_MOTION: {
        "reason_code": "INITIAL_COMPILE",
        "payload": {"mode": "initial", "focus": ["semantic", "timeline", "gem_caption"]},
    },
    PlannerAction.RETRIEVE_REFERENCE: {
        "reason_code": "RARE_MOTION",
        "payload": {"query": "limping gait", "retrieval_type": "motion", "purpose": "motion_prior", "top_k": 5},
    },
    PlannerAction.BUILD_CONSTRAINT: {
        "reason_code": "CONTACT_REQUIREMENT",
        "payload": {"mode": "add", "constraint_type": "contact", "body_part": "right_wrist", "strength": "soft"},
    },
    PlannerAction.BUILD_KEYFRAME: {
        "reason_code": "WHOLE_BODY_STATE_REQUIRED",
        "payload": {"mode": "add", "time": 5.8, "description": "stable seated pose", "source_preference": "auto"},
    },
    PlannerAction.GENERATE: {
        "reason_code": "CONDITIONS_READY",
        "payload": {"strategy": "normal", "scope": "full", "num_candidates": 4, "reward_targets": []},
    },
    PlannerAction.ACCEPT: {"reason_code": "ALL_REQUIRED_CHECKS_PASSED", "payload": {}},
    PlannerAction.STOP_FAILED: {"reason_code": "BUDGET_EXHAUSTED", "payload": {}},
}

INVALID_PAYLOADS = {
    PlannerAction.COMPILE_MOTION: {"mode": "initial", "focus": ["invalid"]},
    PlannerAction.RETRIEVE_REFERENCE: {"query": "x", "retrieval_type": "motion", "purpose": "motion_prior", "top_k": 0},
    PlannerAction.BUILD_CONSTRAINT: {"mode": "add", "constraint_type": "temporal_order"},
    PlannerAction.BUILD_KEYFRAME: {"mode": "add", "time": 1.0, "description": "pose", "source_preference": "text_to_smpl"},
    PlannerAction.GENERATE: {"strategy": "normal", "scope": "full", "num_candidates": 0, "reward_targets": []},
    PlannerAction.ACCEPT: {"unexpected": True},
    PlannerAction.STOP_FAILED: {"unexpected": True},
}


def decision_for(action: PlannerAction, *, payload=None) -> PlannerDecision:
    data = VALID_DECISIONS[action]
    return PlannerDecision(
        action=action,
        reason_code=data["reason_code"],
        target_segments=None,
        reason_summary=f"{action.value} for test",
        payload=data["payload"] if payload is None else payload,
    )


class Phase2ActionGuardTests(unittest.TestCase):
    def test_all_7_action_schemas_accept_valid_and_reject_invalid_payloads(self):
        for action in PlannerAction:
            self.assertEqual(decision_for(action).action, action)
            with self.assertRaises(ValueError, msg=action.value):
                decision_for(action, payload=INVALID_PAYLOADS[action])

    def test_one_action_rule_rejects_multi_action_shapes(self):
        with self.assertRaisesRegex(ValueError, "exactly one"):
            parse_planner_decision([VALID_DECISIONS[PlannerAction.COMPILE_MOTION]])
        with self.assertRaisesRegex(ValueError, "exactly one"):
            parse_planner_decision({"actions": [VALID_DECISIONS[PlannerAction.COMPILE_MOTION]]})

    def test_hard_guards_reject_generate_without_plan_and_budget(self):
        reducer = StateReducer()
        state = reducer.create_run("walk", run_id="guard_run")
        with self.assertRaisesRegex(PlannerGuardError, "ready Motion Plan"):
            validate_planner_decision(state, decision_for(PlannerAction.GENERATE))

        state.plan.status = "ready"
        state.plan.motion_spec_id = "spec_mock"
        state.plan.gem_text_condition_id = "text_mock"
        state.conditions.text.ready = True
        state.control.budgets.generations_left = 0
        with self.assertRaisesRegex(PlannerGuardError, "generation_budget"):
            validate_planner_decision(state, decision_for(PlannerAction.GENERATE))

    def test_accept_guard_requires_fresh_successful_verification(self):
        reducer = StateReducer()
        state = reducer.create_run("walk", run_id="accept_guard")
        with self.assertRaisesRegex(PlannerGuardError, "VerificationReport"):
            validate_planner_decision(state, decision_for(PlannerAction.ACCEPT))

        state.evaluation.verification_summary = VerificationSummary(
            status="complete",
            overall_pass=False,
            failed_check_ids=["semantic"],
            critical_failures=["SEMANTIC_FAILURE"],
            diagnostic_codes=["SEMANTIC_FAILURE"],
        )
        with self.assertRaisesRegex(PlannerGuardError, "overall_pass"):
            validate_planner_decision(state, decision_for(PlannerAction.ACCEPT))

    def test_accept_reducer_invariant_rejects_illegal_terminal_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = ArtifactStore(root / "artifacts")
            store = StateStore(root / "states", event_log=EventLog(root / "events"), artifact_store=artifacts)
            reducer = StateReducer()
            state = store.commit(reducer.create_run("walk", run_id="bad_accept"), expected_version=None, event_type="run_created")
            bad_state = state.model_copy(deep=True)
            bad_state.control.accepted = True
            bad_state.state_version += 1
            with self.assertRaises(Exception):
                store.commit(bad_state, expected_version=0, event_type="bad_accept")

    def test_generate_guard_accepts_ready_plan(self):
        reducer = StateReducer()
        state = reducer.create_run("walk", run_id="ready_generate")
        state = reducer.commit_motion_plan(
            state,
            motion_spec_id="spec_mock",
            gem_text_condition_id="text_mock",
            segments=[MotionSegmentSummary(segment_id=0, start_s=0.0, end_s=6.0, action="walk")],
        )
        validate_planner_decision(state, decision_for(PlannerAction.GENERATE))


if __name__ == "__main__":
    unittest.main()

