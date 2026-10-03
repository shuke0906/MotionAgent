"""Bounded real API validation with CPU-only fixture GEM; no remote GPU access."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "outputs" / "phase11_local_readiness"

from motion_agent.agent.context_builder import PlannerContextBuilder
from motion_agent.agent.guards import validate_planner_decision
from motion_agent.agent.planner import RealLLMPlannerBackend
from motion_agent.compiler.motion_compiler import CompilerRequest, RealLLMMotionCompiler
from motion_agent.llm import LLMClient, LLMConfig
from motion_agent.llm.config import load_dotenv
from motion_agent.state.reducer import StateReducer


def write(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def compiler_checks(client):
    from motion_agent.compiler.llm_semantic_parser import LLMSemanticParserBackend
    parser = LLMSemanticParserBackend(client)
    compiler = RealLLMMotionCompiler(parser)
    prompts = {
        "A": "A person walks forward.",
        "B": "A person waves their right hand three times.",
        "C": "A person walks forward while waving their right hand three times.",
        "D": "A person walks forward, turns left, then sits down.",
    }
    cases = {}
    for name, prompt in prompts.items():
        try:
            result = compiler.compile(CompilerRequest(original_request=prompt))
            segments = result.motion_spec.segments
            actions = [segment.action for segment in segments]
            wave = [segment for segment in segments if segment.action == "wave"]
            forward = any(s.action == "walk" and s.direction == "forward" for s in segments)
            count = any(s.temporal_constraint and s.temporal_constraint.count == 3 and "right_hand" in s.body_parts for s in wave)
            simultaneous = any(s.temporal_relation and s.temporal_relation.type == "simultaneous" for s in segments)
            checks = {"validated": True}
            if name == "A":
                checks["walk_forward"] = forward
            elif name == "B":
                checks["right_hand_count_3"] = count
            elif name == "C":
                checks.update(walk_forward=forward, right_hand_count_3=count, simultaneous=simultaneous)
            else:
                checks.update(ordered_actions=actions == ["walk", "turn", "sit_down"], walk_forward=forward,
                              turn_left=any(s.action == "turn" and s.direction == "left" for s in segments),
                              chronological=all(left.end_s <= right.start_s for left, right in zip(segments, segments[1:])))
            cases[name] = {"prompt": prompt, "success": all(checks.values()), "checks": checks,
                           "result": result.model_dump(mode="json"), "metadata": parser.last_metadata.model_dump(mode="json")}
        except Exception as exc:
            cases[name] = {"prompt": prompt, "success": False, "error_type": getattr(exc, "code", type(exc).__name__)}
        write("real_compiler.json", {"success": all(c["success"] for c in cases.values()), "cases": cases})
        print("Compiler " + name + ": " + ("PASS" if cases[name]["success"] else "FAIL"), flush=True)
    return all(c["success"] for c in cases.values())


def planner_check(client):
    state = StateReducer().create_run("A person walks forward.", run_id="phase11_real_planner_smoke")
    planner = RealLLMPlannerBackend(client)
    decision = planner.step(PlannerContextBuilder().build(state), state)
    validate_planner_decision(state, decision)
    result = {"success": decision.action.value == "COMPILE_MOTION", "decision": decision.model_dump(mode="json"),
              "metadata": planner.last_metadata.model_dump(mode="json")}
    write("real_planner.json", result)
    print("Planner: " + ("PASS" if result["success"] else "FAIL"), flush=True)
    return result["success"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["compiler", "planner", "all"], default="all")
    args = parser.parse_args()
    for name in ("OPENAI_API_KEY", "OPENAI_MODEL", "OPENAI_BASE_URL", "MLLM_API_KEY", "MLLM_MODEL", "MLLM_BASE_URL"):
        os.environ.pop(name, None)
    load_dotenv(ROOT / ".env")
    config = LLMConfig.from_env(load_local_dotenv=False).model_copy(update={"timeout_s": 60, "max_retries": 1})
    client = LLMClient(config)
    success = True
    try:
        if args.stage in {"compiler", "all"}:
            success = compiler_checks(client) and success
        if args.stage in {"planner", "all"}:
            success = planner_check(client) and success
    except Exception as exc:
        result = {"success": False, "stage": args.stage, "error_type": getattr(exc, "code", type(exc).__name__)}
        write("real_validation_error.json", result)
        print(json.dumps(result), flush=True)
        return 1
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
