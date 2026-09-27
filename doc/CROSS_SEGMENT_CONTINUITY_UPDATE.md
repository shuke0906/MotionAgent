# Cross-Segment Continuity Update Summary

## Scope

This update introduces first-class cross-segment motion continuity, starting with heading/facing continuity.

Canonical example:

```text
walk forward
→ wave the right hand
→ sit down
```

If the user does not explicitly request a turn / rotation / facing change, later segments should inherit the heading established by the previous segment.

## Phase impact

### No rework required

```text
Phase 0 — GEM baseline
Phase 1 — State / persistence
Phase 2 — Planner / LangGraph topology
```

No new Planner Action or Graph node is introduced.

### Small retrofit required before continuing

```text
Phase 3 — Motion Compiler
```

Add:

```text
HeadingContinuitySpec
inherit_previous / explicit / follow_trajectory / free
caption materialization
continuity validator
golden tests
```

The already-completed Phase 3 implementation should receive this small code patch and rerun Phase 3 regressions.

### No Phase 4 generator redesign required

Phase 4 keeps the existing:

```text
Compiler → GEMTextCondition → Frozen GEM → MotionCandidate
```

After the Phase 3 retrofit, rerun one Phase 4 contract/GPU regression to confirm continuity metadata/caption propagation. The Generator must not invent heading policy itself.

## Later implementation

```text
Phase 6
→ orientation-only root_trajectory soft heading control

Phase 8
→ heading continuity evidence in Candidate Tournament

Phase 9
→ deterministic-first Heading Continuity Verifier

Phase 10
→ UNCOMMANDED_HEADING_DRIFT diagnosis and targeted repair
```

## Files modified

```text
00_project_overview.md
03_motion_planner.md
05_motion_compiler.md
07_constraint_compiler.md
09_candidate_tournament.md
10_multi_verifier.md
11_diagnosis_repair.md
12_system_integration.md
PROJECT_IMPLEMENTATION_PLAN.md
CONSISTENCY_AUDIT.md
```

## Files reviewed and intentionally unchanged

```text
04_unified_state.md
06_retrieval_tool.md
07b_keyframe_tool.md
08_gem_generation_tool.md
README.md
industrial_harness_design.md
```

Reason: their current contracts already support this extension or they are not responsible for defining/evaluating heading continuity.
