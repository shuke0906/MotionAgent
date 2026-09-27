# Phase 3 Continuity Patch

Status: PASS

## Scope

Patched Phase 3 Motion Compiler only. No Phase 5 work was started. No real GEM generation, GPU execution, RunPod, paid API, verifier, repair, constraint compilation, or Phase 4 architecture changes were run.

## Schema changes

- Added `HeadingContinuitySpec` in `motion_agent/compiler/schemas.py`.
- Added `MotionSegment.heading_continuity`.
- Added compact `HeadingContinuitySummary` in `motion_agent/state/schemas.py`.
- Persisted continuity summaries through the existing `CompilerResult -> StateReducer -> MotionAgentState` compile path.

Policy fields:

```json
{
  "mode": "free | inherit_previous | explicit | follow_trajectory",
  "anchor_segment_id": "int | null",
  "preserve_facing": "bool",
  "explicit_facing": "str | null",
  "source": "user_explicit | continuity_default | trajectory | unknown"
}
```

## Compiler rules

- Locomotion with explicit direction establishes heading as `explicit`.
- Later segments with no turn/facing/trajectory instruction inherit prior heading as `inherit_previous`.
- Explicit turn/rotate/spin/facing creates an `explicit` policy and becomes the new anchor.
- Trajectory-oriented segments use `follow_trajectory`.
- Segments with no meaningful prior heading remain `free`.
- Captions receive a lightweight continuity cue only for inherited-heading segments where it helps preserve motion semantics.
- No numeric yaw, root rotation, reward, constraint, verifier, repair, or generation artifact is created by Phase 3.

## Files changed

- `motion_agent/compiler/schemas.py`
- `motion_agent/compiler/continuity.py`
- `motion_agent/compiler/motion_compiler.py`
- `motion_agent/compiler/semantic_parser.py`
- `motion_agent/compiler/caption_optimizer.py`
- `motion_agent/compiler/validators.py`
- `motion_agent/compiler/__init__.py`
- `motion_agent/state/schemas.py`
- `motion_agent/agent/mock_tools.py`
- `tests/unit/test_phase3_motion_compiler.py`
- `tests/integration/test_phase3_compile_node.py`
- `scripts/validate_phase3.py`
- `reports/phase_3_continuity_patch.md`

## Representative outputs

Input:

```text
walk forward, wave the right hand, then sit down
```

Continuity:

```json
[
  {"segment_id": 0, "action": "walk", "mode": "explicit", "anchor_segment_id": null, "explicit_facing": "forward"},
  {"segment_id": 1, "action": "wave", "mode": "inherit_previous", "anchor_segment_id": 0, "explicit_facing": null},
  {"segment_id": 2, "action": "sit_down", "mode": "inherit_previous", "anchor_segment_id": 1, "explicit_facing": null}
]
```

Input:

```text
walk forward, turn right, then sit down
```

Continuity:

```json
[
  {"segment_id": 0, "action": "walk", "mode": "explicit", "anchor_segment_id": null, "explicit_facing": "forward"},
  {"segment_id": 1, "action": "turn", "mode": "explicit", "anchor_segment_id": 0, "explicit_facing": "right"},
  {"segment_id": 2, "action": "sit_down", "mode": "inherit_previous", "anchor_segment_id": 1, "explicit_facing": null}
]
```

## Tests run

```text
python -m pytest tests/unit/test_phase3_motion_compiler.py tests/integration/test_phase3_compile_node.py -v
19 passed, 24 subtests passed

python -m pytest tests -v
50 passed, 24 subtests passed

python scripts/validate_phase3.py
PHASE 3 GATE PASS
```

Validation summary:

```text
Heading inheritance            PASS
Explicit turn reset            PASS
No-prior free policy           PASS
Trajectory policy              PASS
Segment revision preservation  PASS
Simultaneous action            PASS
Explicit facing                PASS
Phase 1-4 regression           PASS
```

Total passed / failed:

```text
50 passed / 0 failed
```

## Remaining limitations

- Continuity is semantic metadata only; it does not enforce numeric yaw or root orientation.
- Relative phrases such as `turn right` are stored as explicit facing metadata, not absolute world yaw.
- The deterministic parser covers the local Phase 3 fixtures and requested phrases; broader language coverage remains a later parser/LLM concern.
- Actual heading drift measurement remains downstream Tournament / Verifier work.
