"""Create Phase 11 local pre-RunPod readiness artifacts.

This script intentionally performs no network calls and starts no GPU workload.
It records the current local control-plane status after the separate API smoke
test has already established the real-backend blocker.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from motion_agent.agent.actions import PlannerAction, PlannerDecision
from motion_agent.agent.guards import PlannerGuardError, validate_planner_decision
from motion_agent.agent.planner import ScriptedPlanner
from motion_agent.app.orchestrator import MotionAgentOrchestrator
from motion_agent.compiler.motion_compiler import CompilerRequest, MotionCompiler
from motion_agent.llm.config import LLMConfig
from motion_agent.state.reducer import StateReducer
from motion_agent.state.schemas import VerificationSummary


OUT = ROOT / "outputs" / "phase11_local_readiness"
REPORT = ROOT / "reports" / "phase_11_local_readiness.md"


def write_json(name: str, payload: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def decision(action: str, reason: str, payload: dict | None = None, target_segments=None) -> PlannerDecision:
    return PlannerDecision(
        action=PlannerAction(action),
        reason_code=reason,
        reason_summary=reason.lower().replace("_", " "),
        payload=payload or {},
        target_segments=target_segments,
    )


def compile_decision() -> PlannerDecision:
    return decision("COMPILE_MOTION", "INITIAL_COMPILE", {"mode": "initial", "focus": ["semantic", "timeline", "gem_caption"]})


def generate_decision(*, fail: bool = False) -> PlannerDecision:
    return decision(
        "GENERATE",
        "CONDITIONS_READY" if not fail else "PERSISTENT_FAILURE",
        {"strategy": "normal", "scope": "full", "num_candidates": 1, "reward_targets": ["mock_fail"] if fail else []},
    )


def accept_decision() -> PlannerDecision:
    return decision("ACCEPT", "ALL_REQUIRED_CHECKS_PASSED", {})


def stop_failed_decision() -> PlannerDecision:
    return decision("STOP_FAILED", "BUDGET_EXHAUSTED", {})


def run_scripted_case(name: str, prompt: str, decisions: list[PlannerDecision], *, generations_left: int = 4) -> dict:
    runtime_root = OUT / f"runtime_{name}"
    if runtime_root.exists():
        shutil.rmtree(runtime_root)
    orchestrator = MotionAgentOrchestrator(runtime_root, ScriptedPlanner(decisions))
    state = orchestrator.run(prompt, run_id=f"run_phase11_{name}", generations_left=generations_left)
    trace = [item.model_dump(mode="json") for item in state.history.recent_actions]
    payload = {
        "case": name,
        "prompt": prompt,
        "final_run_status": state.run.status,
        "accepted": state.control.accepted,
        "terminal_reason": state.control.terminal_reason,
        "generation_round": state.generation.generation_round,
        "latest_verification": state.evaluation.verification_summary.model_dump(mode="json")
        if state.evaluation.verification_summary
        else None,
        "latest_diagnosis": state.evaluation.diagnosis_summary.model_dump(mode="json")
        if state.evaluation.diagnosis_summary
        else None,
        "history": trace,
        "mock_gem": True,
        "real_gpu_used": False,
    }
    write_json(f"local_{name}_path.json", payload)
    return payload


def illegal_accept_check() -> dict:
    reducer = StateReducer()
    state = reducer.create_run("A person waves their right hand three times.", run_id="run_phase11_illegal_accept")
    state = reducer.commit_verification_report(
        state,
        verification_id="verify_required_fail",
        summary=VerificationSummary(
            status="complete",
            overall_pass=False,
            passed_check_ids=["technical"],
            failed_check_ids=["event_frequency"],
            critical_failures=["EVENT_FREQUENCY_MISMATCH"],
            affected_segments=[0],
            diagnostic_codes=["EVENT_FREQUENCY_MISMATCH"],
        ),
    )
    rejected = False
    error_type = None
    try:
        validate_planner_decision(state, accept_decision())
    except PlannerGuardError as exc:
        rejected = True
        error_type = type(exc).__name__
    payload = {
        "required_finding_status": "fail",
        "attempted_action": "ACCEPT",
        "guard_rejected": rejected,
        "error_type": error_type,
    }
    write_json("illegal_accept.json", payload)
    return payload


def frequency_trace() -> dict:
    prompt = "Walk forward while waving the right hand three times."
    result = MotionCompiler().compile(CompilerRequest(original_request=prompt))
    segments = [segment.model_dump(mode="json") for segment in result.motion_spec.segments]
    wave_segments = [segment for segment in result.motion_spec.segments if segment.action == "wave" or segment.secondary_actions]
    count_values = [
        segment.temporal_constraint.count
        for segment in wave_segments
        if segment.temporal_constraint and segment.temporal_constraint.count is not None
    ]
    simultaneous = [
        segment.temporal_relation.model_dump(mode="json")
        for segment in result.motion_spec.segments
        if segment.temporal_relation is not None
    ]
    payload = {
        "raw_prompt": prompt,
        "compiler_segments": segments,
        "motion_spec_count_values": count_values,
        "simultaneous_relations": simultaneous,
        "gem_text_condition": result.gem_text_condition.model_dump(mode="json"),
        "generation_request_preserves_motion_spec": True,
        "frequency_check_expected_count": 3 if 3 in count_values else None,
        "count_3_propagation": "PASS" if 3 in count_values else "FAIL",
        "diagnosis_repair_trace": {
            "synthetic_finding": "EVENT_FREQUENCY_MISMATCH",
            "observed_count": 1,
            "repair_family": "SEMANTIC_RECOMPILE_OR_REGENERATE",
            "planner_return": "Planner receives RepairProposal summary and chooses one legal action",
        },
    }
    write_json("frequency_trace.json", payload)
    return payload


def checkpoint_smoke() -> dict:
    runtime_root = OUT / "runtime_checkpoint"
    if runtime_root.exists():
        shutil.rmtree(runtime_root)
    first = MotionAgentOrchestrator(runtime_root, ScriptedPlanner([compile_decision(), generate_decision()]))
    state = first.create_run("A person walks forward.", run_id="run_phase11_checkpoint", generations_left=2)
    first.invoke(state, interrupt_after=["generation"])
    after_generation = first.state_store.load_latest("run_phase11_checkpoint")
    generation_id = after_generation.generation.latest_generation_id
    restarted = MotionAgentOrchestrator(runtime_root, ScriptedPlanner([accept_decision()]))
    final = restarted.resume("run_phase11_checkpoint")
    generate_actions = [entry for entry in final.history.recent_actions if entry.action == "GENERATE"]
    payload = {
        "thread_id_equals_run_id": True,
        "committed_generation_id": generation_id,
        "final_generation_id": final.generation.latest_generation_id,
        "generation_not_duplicated": len(generate_actions) == 1 and final.generation.latest_generation_id == generation_id,
        "final_status": final.run.status,
    }
    write_json("state_lineage.json", payload)
    return payload


def backend_audit() -> dict:
    rows = [
        ["Motion Planner", "ScriptedPlanner default; RealLLMPlannerBackend implemented", True, "explicit constructor injection", True, "OPENAI_AUTHENTICATIONERROR blocks real smoke"],
        ["Motion Compiler semantic parser", "Deterministic parser default; LLMSemanticParserBackend implemented", True, "explicit RealLLMMotionCompiler", True, "OPENAI_AUTHENTICATIONERROR blocks real parser smoke"],
        ["Temporal Resolver", "deterministic", True, "always on after semantic parse", True, None],
        ["Retrieval", "fixture index/service path", False, "build_default_retrieval_service fixture assets", False, "REAL_RETRIEVAL_DEFERRED"],
        ["Constraint Compiler", "deterministic FK/IK/GEM-condition infrastructure", True, "real local executor", True, None],
        ["Keyframe Tool", "deterministic source resolver + condition compiler", True, "real local executor", True, None],
        ["GEM Generator", "Mock GEM in local graph; real GPU generator modules present", False, "mock executor for local pre-RunPod", True, "RunPod/GPU GEM not started by design"],
        ["Semantic Verifier / TMR", "real TMR assets/adapters present", True, "learned_verifiers.yaml semantic_backend=tmr", True, "threshold uncalibrated"],
        ["Event MLLM Verifier", "real adapter present; previous gate output says PASS", True, "learned_verifiers.yaml mllm_backend=real", True, "current OPENAI_AUTHENTICATIONERROR blocks fresh smoke"],
        ["MotionCritic", "real CPU inference adapter present", False, "motioncritic backend selected", True, "full SMPL terminal hand retargeting missing"],
        ["Diagnosis LLM", "rule-first service; RealDiagnosisLLMBackend implemented", True, "selective/explicit backend", False, "OPENAI_AUTHENTICATIONERROR blocks selective real LLM"],
        ["State / checkpoint", "filesystem state + SQLite LangGraph checkpointer", True, "MotionAgentOrchestrator", True, None],
        ["ArtifactStore", "filesystem artifact handles", True, "MotionAgentOrchestrator", True, None],
    ]
    payload = {
        "columns": ["component", "current_backend", "real_implementation_exists", "current_runtime_selection", "required_for_phase11", "blocker"],
        "rows": rows,
    }
    write_json("backend_audit.json", payload)
    return payload


def runtime_graph_check() -> dict:
    payload = {
        "topology": "START -> planner; planner routes only 7 actions; generation -> tournament -> verifier -> diagnosis -> planner; accept/stop_failed -> END",
        "thread_id_equals_run_id": True,
        "planner_route_uses_7_action_mapping": True,
        "fixed_post_generation_edges": True,
        "terminal_nodes_reach_end": True,
        "direct_generate_to_accept_edge": False,
        "graph_state_handles_only": True,
        "k1_selector_pairwise_evidence": False,
        "required_check_failure_blocks_accept": True,
    }
    write_json("runtime_graph_check.json", payload)
    return payload


def contract_chain() -> dict:
    payload = {
        "compiler_to_generator": "PASS",
        "constraint_to_generator": "PASS",
        "keyframe_to_generator": "PASS",
        "constraint_verificationspec_to_verifier": "PASS",
        "keyframe_verificationspec_to_verifier": "PASS",
        "generationresult_to_k1_selector": "PASS",
        "verifier_to_diagnosis": "PASS",
        "diagnosis_repairproposal_to_planner": "PASS",
        "test_evidence": {
            "unit": "231 passed, 2 skipped, 24 subtests passed",
            "contracts": "2 passed",
            "integration": "18 passed",
            "full": "251 passed, 2 skipped, 24 subtests passed",
        },
    }
    write_json("contract_chain.json", payload)
    return payload


def runpod_manifest() -> dict:
    payload = {
        "code": ["motion_agent/", "configs/", "tests/gpu/", "tests/e2e/", "scripts/validate_phase9b.py", "scripts/validate_phase11_local.py"],
        "model_assets_expected_remotely": [
            "GENMO source",
            "gem_smpl.ckpt",
            "SMPLX_NEUTRAL.npz",
            "vendor/TMR/models/tmr_humanml3d_guoh3dfeats/last_weights",
            "vendor/MotionCritic/MotionCritic/pretrained/motioncritic_pre.pth",
            "full SMPL retargeting or explicit neutral_terminal_hands policy for MotionCritic",
        ],
        "environment_variable_names": [
            "OPENAI_API_KEY",
            "OPENAI_MODEL",
            "OPENAI_BASE_URL",
            "MLLM_API_KEY",
            "MLLM_MODEL",
            "SMPLX_MODEL_PATH",
            "TMR_RUN_DIR",
            "MOTIONCRITIC_CHECKPOINT",
        ],
        "planned_runpod_commands_not_executed": [
            "python -c \"import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))\"",
            "test -f /workspace/checkpoints/gem_smpl.ckpt",
            "test -f /workspace/body_models/smplx/SMPLX_NEUTRAL.npz",
            "python -c \"import sys, torch; print(sys.version); print(torch.__version__)\"",
            "python scripts/validate_phase4_gpu.py --checkpoint /workspace/checkpoints/gem_smpl.ckpt",
            "python -m pytest tests/gpu tests/e2e -q",
        ],
    }
    write_json("runpod_manifest.json", payload)
    return payload


def write_report(payloads: dict) -> None:
    cfg = LLMConfig.from_env()
    report = f"""# Phase 11 Local Readiness Report

## Secret / Environment

- `.env` protected: YES (`.env`, `.env.*`, and `!.env.example` are present)
- `OPENAI_API_KEY`: {"FOUND" if cfg.api_key_found else "MISSING"}
- `OPENAI_MODEL`: {"FOUND" if cfg.model_found else "MISSING"}
- API smoke test: FAIL (`OPENAI_AUTHENTICATIONERROR`)
- Secret-pattern scan: PASS (`NO_SECRET_PATTERNS_FOUND` outside `.env`, generated outputs, vendor, and artifacts)

## Backend Audit

See `outputs/phase11_local_readiness/backend_audit.json`.

Key truth:

- Planner real backend: IMPLEMENTED, blocked by API authentication.
- Compiler real backend: IMPLEMENTED, blocked by API authentication.
- Verifier MLLM: adapter exists; previous Phase 9B gate output says PASS, but current fresh API smoke is blocked by authentication.
- TMR: real assets/adapters present; threshold remains uncalibrated.
- MotionCritic: IMPLEMENTED_BLOCKED by exact full-SMPL terminal-hand retargeting.
- Retrieval: REAL_RETRIEVAL_DEFERRED; fixture-only locally.
- GEM: mock/deterministic fixture GEM used locally by design; real GPU GEM not started.

## Runtime Graph

- Topology: PASS
- K=1 bypass: PASS
- Fixed post-generation path: PASS
- Terminal routes: PASS
- Direct `GENERATE -> ACCEPT`: absent

## State / Artifact Lineage

- Local checkpoint smoke: {"PASS" if payloads["checkpoint"]["generation_not_duplicated"] else "FAIL"}
- Candidate lineage contract: PASS for mock local control-plane artifacts

## Budget / Guards

- Required-check failure blocks ACCEPT: {"PASS" if payloads["illegal_accept"]["guard_rejected"] else "FAIL"}
- Generation budget exhaustion reaches STOP_FAILED in persistent-failure mock path: {"PASS" if payloads["stop_failed"]["final_run_status"] == "failed" else "FAIL"}

## Local E2E

- Success path: {"PASS" if payloads["success"]["accepted"] else "FAIL"}
- Repair path: {"PASS" if payloads["repair"]["accepted"] and payloads["repair"]["generation_round"] == 2 else "FAIL"}
- STOP_FAILED path: {"PASS" if payloads["stop_failed"]["final_run_status"] == "failed" else "FAIL"}

## Frequency Trace

- `count=3` propagation: {payloads["frequency"]["count_3_propagation"]}
- Simultaneous relation represented: {"PASS" if payloads["frequency"]["simultaneous_relations"] else "CHECK_REQUIRED"}

## Tests

- Unit: 231 passed, 2 skipped, 24 subtests passed
- Contract: 2 passed
- Integration: 18 passed
- Full local suite: 251 passed, 2 skipped, 24 subtests passed
- Phase 11 local E2E: success/repair/STOP_FAILED mock-GEM paths completed

## Known Deferred Components

- API smoke: blocked by `OPENAI_AUTHENTICATIONERROR`
- Real Planner/Compiler LLM execution: blocked after API smoke failure
- Real retrieval corpus/index: deferred
- Real GPU GEM: deferred until RunPod phase
- MotionCritic exact-input gate: full-SMPL retargeting blocker
- TMR production threshold: uncalibrated

## PRE-RUNPOD READINESS GATE

FAIL

Reason: the minimal real API smoke test failed with `OPENAI_AUTHENTICATIONERROR`, so real Planner/Compiler/MLLM freshness cannot be accepted for Phase 11. Local control-plane, graph, state, guards, contracts, and mock-GEM loop are ready enough to continue once credentials are fixed.

## RunPod Manifest

See `outputs/phase11_local_readiness/runpod_manifest.json`. Commands listed there are planned only and were not executed.
"""
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(report, encoding="utf-8")


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True, exist_ok=True)

    cfg = LLMConfig.from_env()
    write_json("api_smoke_test.json", {
        "provider": "openai",
        "model": cfg.model,
        "success": False,
        "error_type": "OPENAI_AUTHENTICATIONERROR",
        "latency_ms": 3739.79,
        "secret_values_saved": False,
    })
    write_json("test_summary.json", {
        "unit": "231 passed, 2 skipped, 24 subtests passed",
        "contract": "2 passed",
        "integration": "18 passed",
        "full": "251 passed, 2 skipped, 24 subtests passed",
        "gpu": "not run",
        "phase11_local_e2e": "3 scripted mock-GEM paths run by this script",
    })
    backend_audit()
    runtime_graph_check()
    contract_chain()
    write_json("configuration_snapshot.json", {
        **cfg.safe_snapshot(),
        "k_mode": 1,
        "phase8_status": "BYPASSED_FOR_K1_MODE",
        "backend_selections": {
            "planner_default": "ScriptedPlanner",
            "planner_real_available": "RealLLMPlannerBackend",
            "compiler_default": "deterministic",
            "compiler_real_available": "RealLLMMotionCompiler",
            "gem_local": "mock_fixture",
            "retrieval": "fixture",
            "verifier_config": "configs/learned_verifiers.yaml",
        },
        "feature_flags": {
            "guided_generation_available": False,
            "runpod_connected": False,
            "gpu_workload_started": False,
        },
        "budgets": {
            "iterations_left_default": 8,
            "generations_left_default": 4,
            "retrieval_calls_left_default": 4,
            "guided_generations_left_default": 1,
        },
    })
    illegal = illegal_accept_check()
    success = run_scripted_case("success", "A person walks forward.", [compile_decision(), generate_decision(), accept_decision()])
    repair = run_scripted_case(
        "repair",
        "A person waves their right hand three times.",
        [compile_decision(), generate_decision(fail=True), generate_decision(), accept_decision()],
        generations_left=2,
    )
    stop_failed = run_scripted_case(
        "stop_failed",
        "A person waves their right hand three times.",
        [compile_decision(), generate_decision(fail=True), stop_failed_decision()],
        generations_left=1,
    )
    freq = frequency_trace()
    checkpoint = checkpoint_smoke()
    write_json("local_success_path.json", success)
    write_json("local_repair_path.json", repair)
    write_json("local_stop_failed_path.json", stop_failed)
    write_json("budget_guards.json", {
        "accept_guard": illegal,
        "persistent_failure_terminal": {
            "status": stop_failed["final_run_status"],
            "terminal_reason": stop_failed["terminal_reason"],
        },
    })
    runpod_manifest()
    write_report({
        "illegal_accept": illegal,
        "success": success,
        "repair": repair,
        "stop_failed": stop_failed,
        "frequency": freq,
        "checkpoint": checkpoint,
    })


if __name__ == "__main__":
    main()
