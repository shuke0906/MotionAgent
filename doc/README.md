# MotionAgent Documentation Index

本目录是当前 MotionAgent for GEM 的**最新模块化实现指南**。V1 明确使用 **LangGraph StateGraph** 作为 Agent orchestration/runtime；Motion Planner 仍是唯一拥有全局语义决策权的 Agent。

本目录即最终实现文档集：请优先从 `00_project_overview.md` 开始；实现跨模块服务时同时阅读 `12_system_integration.md` 与 `industrial_harness_design.md`。旧的单体长文档不作为 Source of Truth，避免与最新模块接口混用。

## 建议阅读顺序

1. `00_project_overview.md` — 项目总览、完整数据流、模块职责、代码架构、Definition of Done
2. `04_unified_state.md` — Canonical State / Artifact / Event Log / LangGraph Checkpointer / Planner Context / Versioning
3. `03_motion_planner.md` — 唯一 Global Planner、Action Space、Hard Guards
4. `05_motion_compiler.md` — Natural Language → Motion DSL → GEMTextCondition
5. `06_retrieval_tool.md` — HumanML3D / TMR Retrieval
6. `07_constraint_compiler.md` — Numerical Constraints / Hard & Soft Conditions
7. `07b_keyframe_tool.md` — Whole-Body Keyframe Control
8. `08_gem_generation_tool.md` — Frozen GEM Generation / K Candidates / Segment Inpainting
9. `09_candidate_tournament.md` — VISTA-style Relative Candidate Selection
10. `10_multi_verifier.md` — Dynamic Multi-Verifier / Absolute Acceptance
11. `11_diagnosis_repair.md` — Failure Diagnosis / Planner-compatible Repair Proposal
12. `12_system_integration.md` — LangGraph StateGraph、Routing、Registry、Automatic Subgraph、集成规范
13. `industrial_harness_design.md` — LangGraph Runtime + 工业化 Harness、Context、State、Worker、Queue、Guard、Tracing、Eval
14. `CONSISTENCY_AUDIT.md` — 跨模块接口与口径检查结果、V1 已知边界

## 唯一 Planner Action Space

```text
COMPILE_MOTION
RETRIEVE_REFERENCE
BUILD_CONSTRAINT
BUILD_KEYFRAME
GENERATE
ACCEPT
STOP_FAILED
```

`VERIFY / REPAIR / REGENERATE / GENERATE_GUIDED / TOURNAMENT` 不是 Planner Action。

## 关键控制关系

```text
LangGraph StateGraph
→ Planner
→ ONE Action
→ Conditional Route
→ Tool / Workflow Node
→ Canonical State Commit
→ Planner
```

`GENERATE` 之后固定执行：

```text
Generation
→ Tournament
→ Multi-Verifier
→ Diagnosis
→ Planner
```

## V1 核心研究边界

```text
Frozen GEM
+ LangGraph orchestration runtime
```

V1 不重新训练 GEM。

