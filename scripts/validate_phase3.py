from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from motion_agent.compiler import CompilerRequest, MotionCompiler
from motion_agent.compiler.validators import validate_heading_continuity, validate_motion_semantics, validate_timeline


def compile_prompt(prompt: str):
    return MotionCompiler().compile(
        CompilerRequest(original_request=prompt, duration_s=6.0, fps=30, total_frames=180)
    )


def passfail(value: bool) -> str:
    return "PASS" if value else "FAIL"


def main() -> int:
    start = time.perf_counter()
    pytest_code = pytest.main(["tests", "-q"])
    pytest_ms = (time.perf_counter() - start) * 1000

    golden = compile_prompt("wave the right hand three times and sit down")
    inherited = compile_prompt("walk forward, wave the right hand, then sit down")
    explicit_turn = compile_prompt("walk forward, turn right, then sit down")
    no_prior = compile_prompt("wave the right hand")
    trajectory = compile_prompt("walk along a circle")
    simultaneous = compile_prompt("walk forward while waving the right hand")
    explicit_facing = compile_prompt("walk forward while facing left")
    contact = compile_prompt("touch the table with the right hand")
    keyframe = compile_prompt("finish in a stable seated whole-body pose")
    revision_base = compile_prompt("walk forward, wave the right hand, then sit down")
    revised = MotionCompiler().compile(
        CompilerRequest(
            original_request="raise the right arm",
            mode="revise",
            target_segments=[2],
            focus=["semantic", "body_part", "gem_caption"],
            duration_s=6.0,
            fps=30,
            total_frames=180,
            existing_motion_spec=revision_base.motion_spec,
        )
    )

    checks = {
        "Heading inheritance": [s.heading_continuity.mode for s in inherited.motion_spec.segments]
        == ["explicit", "inherit_previous", "inherit_previous"]
        and [s.heading_continuity.anchor_segment_id for s in inherited.motion_spec.segments] == [None, 0, 1],
        "Explicit turn reset": [s.heading_continuity.mode for s in explicit_turn.motion_spec.segments]
        == ["explicit", "explicit", "inherit_previous"]
        and explicit_turn.motion_spec.segments[2].heading_continuity.anchor_segment_id == 1,
        "No-prior free policy": no_prior.motion_spec.segments[0].heading_continuity.mode == "free",
        "Trajectory policy": trajectory.motion_spec.segments[0].heading_continuity.mode == "follow_trajectory",
        "Golden DSL": len(golden.motion_spec.segments) == 2
        and golden.motion_spec.segments[0].action == "wave"
        and golden.motion_spec.segments[0].repetition == 3
        and golden.motion_spec.segments[1].action == "sit_down",
        "Timeline invariants": True,
        "Simultaneous action": len(simultaneous.motion_spec.segments) == 1
        and simultaneous.motion_spec.segments[0].secondary_actions == ["wave right hand"]
        and simultaneous.motion_spec.segments[0].heading_continuity.explicit_facing == "forward",
        "Explicit facing": len(explicit_facing.motion_spec.segments) == 1
        and explicit_facing.motion_spec.segments[0].heading_continuity.explicit_facing == "left",
        "Routing-hint boundaries": any(h.intent_type == "constraint_candidate" for h in contact.routing_hints)
        and not any("CompiledConstraint" in json.dumps(contact.model_dump(mode="json")) for _ in [None]),
        "Caption preservation": True,
        "GEMTextCondition contract": len(golden.gem_text_condition.captions)
        == len(golden.gem_text_condition.window_start)
        == len(golden.gem_text_condition.window_end),
        "Segment revision preservation": revised.changed_segments == [2]
        and revised.motion_spec.segments[0].action == revision_base.motion_spec.segments[0].action
        and revised.motion_spec.segments[1].heading_continuity
        == revision_base.motion_spec.segments[1].heading_continuity,
        "Phase 1-4 regression": pytest_code == 0,
    }

    try:
        for result in [golden, inherited, explicit_turn, no_prior, trajectory, simultaneous, explicit_facing, contact, keyframe, revised]:
            validate_timeline(result.motion_spec.segments, total_frames=result.motion_spec.total_frames)
            validate_heading_continuity(result.motion_spec.segments)
            validate_motion_semantics(result.motion_spec, result.gem_text_condition)
    except Exception:
        checks["Timeline invariants"] = False
        checks["Caption preservation"] = False

    for label, passed in checks.items():
        print(f"{label:<30} {passfail(passed)}")
    print()
    print(f"pytest full suite             {passfail(pytest_code == 0)} ({pytest_ms:.1f} ms)")
    print()
    for prompt, result in [
        ("walk forward, wave the right hand, then sit down", inherited),
        ("walk forward, turn right, then sit down", explicit_turn),
        ("wave the right hand three times and sit down", golden),
        ("walk forward while waving the right hand", simultaneous),
        ("touch the table with the right hand", contact),
    ]:
        print(f"Input: {prompt}")
        print(json.dumps(result.motion_spec.model_dump(mode="json")["segments"], indent=2))
        print(json.dumps(result.gem_text_condition.model_dump(mode="json"), indent=2))
        print()

    gate_pass = pytest_code == 0 and all(checks.values())
    print(f"PHASE 3 GATE                 {passfail(gate_pass)}")
    return 0 if gate_pass else 1


if __name__ == "__main__":
    sys.exit(main())
