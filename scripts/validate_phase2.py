from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from motion_agent.agent.context_builder import PlannerContextBuilder
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


def generate_decision(*, fail: bool = False):
    return {
        "action": "GENERATE",
        "reason_code": "CONDITIONS_READY",
        "target_segments": None,
        "reason_summary": "Generate deterministic mock candidates.",
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


def main() -> int:
    tests = [
        "tests/unit/test_phase2_actions_guards.py",
        "tests/integration/test_phase2_graph.py",
    ]
    pytest_start = time.perf_counter()
    pytest_code = pytest.main([*tests, "-q"])
    pytest_ms = (time.perf_counter() - pytest_start) * 1000

    with tempfile.TemporaryDirectory() as tmp:
        planner = ScriptedPlanner([compile_decision(), generate_decision(), accept_decision()])
        orch = MotionAgentOrchestrator(Path(tmp), planner)
        start = time.perf_counter()
        final_state = orch.run("user walks forward", run_id="validate_phase2")
        latency_ms = (time.perf_counter() - start) * 1000
        events = orch.event_log.list("validate_phase2")
        context_size = len(PlannerContextBuilder().build(final_state).model_dump_json())
        tool_call_count = len(
            [
                event
                for event in events
                if event.event_type
                in {
                    "compile_motion_committed",
                    "generation_committed",
                    "tournament_committed",
                    "verification_committed",
                    "diagnosis_committed",
                }
            ]
        )

        resume_planner = ScriptedPlanner([compile_decision(), generate_decision()])
        resume_orch = MotionAgentOrchestrator(Path(tmp) / "resume", resume_planner)
        resume_initial = resume_orch.create_run("resume walk", run_id="validate_resume")
        resume_orch.invoke(resume_initial, interrupt_after=["generation"])
        resume_start = time.perf_counter()
        MotionAgentOrchestrator(Path(tmp) / "resume", ScriptedPlanner([accept_decision()])).resume("validate_resume")
        checkpoint_overhead_ms = (time.perf_counter() - resume_start) * 1000

    gates = [
        ("Graph topology", pytest_code == 0),
        ("7 action schemas", pytest_code == 0),
        ("Hard guards", pytest_code == 0),
        ("One-action rule", pytest_code == 0),
        ("Automatic post-generation", pytest_code == 0),
        ("Mock E2E success", final_state.run.status == "accepted"),
        ("Failure/recovery loop", pytest_code == 0),
        ("STOP_FAILED", pytest_code == 0),
        ("Checkpoint regression", pytest_code == 0),
    ]

    for label, passed in gates:
        print(f"{label:<32} {'PASS' if passed else 'FAIL'}")
    print()
    print(f"pytest phase2 tests          {'PASS' if pytest_code == 0 else 'FAIL'} ({pytest_ms:.1f} ms)")
    print(f"graph execution latency      {latency_ms:.1f} ms")
    print(f"node transition count        {len(events) - 1}")
    print(f"PlannerContext size          {context_size} bytes")
    print(f"checkpoint resume overhead   {checkpoint_overhead_ms:.1f} ms")
    print(f"Planner decisions            {planner.decisions_made}")
    print(f"tool call count              {tool_call_count}")
    print()
    print(f"PHASE 2 GATE                 {'PASS' if pytest_code == 0 and final_state.run.status == 'accepted' else 'FAIL'}")
    return int(pytest_code)


if __name__ == "__main__":
    sys.exit(main())
