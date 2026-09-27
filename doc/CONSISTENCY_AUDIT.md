# MotionAgent Documentation Consistency Audit

本文件记录最终 ZIP 打包前完成的跨模块一致性检查。实现时若多个文档出现解释冲突，以各模块正式 Schema 与本文件列出的 Source-of-Truth 顺序为准。

## 1. 最终 Source of Truth

```text
03 Motion Planner
04 Unified State / Harness State
05 Motion Compiler
06 Retrieval Tool
07 Constraint Compiler
07B Keyframe Tool
08 GEM Generation Tool
09 Candidate Tournament
10 Multi-Verifier
11 Diagnosis & Repair Planning
12 System Integration
```

`00_project_overview.md` 是上述模块的总览和实施导航；`industrial_harness_design.md` 是工业化运行规范，不重新定义业务模块接口。

`PROJECT_IMPLEMENTATION_PLAN.md` 定义 Stage Gate 和实施顺序。

## 2. LangGraph Runtime 口径

V1 orchestration runtime 固定为：

```text
LangGraph StateGraph
+ conditional routing
+ persistent checkpointer / resume
```

边界固定：

```text
Motion Planner = 唯一 Autonomous Agent / Global Controller
LangGraph Node ≠ Agent
LangGraph = execution runtime，不定义 motion-domain policy
```

State 采用双层关系：

```text
LangGraph GraphState / Checkpointer
→ 轻量 execution progress / routing metadata

MotionAgentState + Artifact Store + Event Log
→ canonical business state / heavy artifact / audit
```

Automatic post-generation path 是固定 LangGraph subgraph：

```text
generation → tournament → verifier → diagnosis → planner
```

## 3. 唯一 Planner Action Space

```text
COMPILE_MOTION
RETRIEVE_REFERENCE
BUILD_CONSTRAINT
BUILD_KEYFRAME
GENERATE
ACCEPT
STOP_FAILED
```

以下词可以在文档中作为“旧设计/明确禁止/说明文字”出现，但不是可执行 Planner Action：

```text
PLAN
VERIFY
REPAIR
REGENERATE
GENERATE_GUIDED
TOURNAMENT
```

静态检查确认：最终文档集中不存在把上述旧词写成 `"action": "..."` / `"planner_action": "..."` 的可执行示例。

## 4. Automatic Post-Generation Subgraph

```text
GENERATE
→ GenerationResult
→ Candidate Tournament
→ TournamentResult / Champion
→ Multi-Verifier
→ VerificationReport
→ DiagnosisResult / RepairProposal[]
→ Planner
```

Tournament、Verifier、Diagnosis 均不进入 Planner Action Space。

## 5. 核心 Producer → Consumer Contract

| Producer | Consumer | Contract | 检查 |
|---|---|---|---|
| 05 Motion Compiler | 08 GEM Generation | `GEMTextCondition` | PASS |
| 07 Constraint Compiler | 08 GEM Generation | `ConstraintBundle` | PASS |
| 07 Constraint Compiler | 10 Multi-Verifier | `VerificationSpec` | PASS |
| 07B Keyframe Tool | 08 GEM Generation | `KeyframeSpec` | PASS |
| 07B Keyframe Tool | 10 Multi-Verifier | `KeyframeVerificationSpec` | PASS |
| 08 GEM Generation | 09 Candidate Tournament | `GenerationResult` | PASS |
| 09 Candidate Tournament | 10 Multi-Verifier | `TournamentResult` | PASS |
| 10 Multi-Verifier | 11 Diagnosis | `VerificationReport` | PASS |
| 11 Diagnosis | 03 Motion Planner | `RepairProposal` | PASS |

## 6. State / Artifact 口径

04 已统一为：

```text
LangGraph GraphState / Checkpointer   (execution progress)
+
Current MotionAgent State            (canonical domain state)
+
Artifact Store                       (heavy data)
+
Event Log                            (audit / replay)
```

PlannerContext 只包含摘要、ID、状态、预算和能力信息；不包含大型 SMPL / 151-D / Render Tensor。

## 7. Keyframe 口径

已增加独立 `07b_keyframe_tool.md`。正式链路：

```text
Planner BUILD_KEYFRAME
→ KeyframeRequest
→ KeyframeSpec
→ Keyframe HardMotionCondition / KeyframeVerificationSpec
→ 08 Generation / 10 Verification
```

Whole-body Keyframe 与局部 Constraint 保持语义分离，但底层 GEM encoder / hard-mask infrastructure 可以共享。

## 8. Frozen GEM 边界

V1 不重新训练 GEM。

```text
Motion Compiler / Retrieval / Constraint / Keyframe
→ 编译条件
→ Frozen GEM
→ Candidates
```

`observed_motion_3d + motion_mask_3d` 使用现有 denoiser hard-overwrite hook；V1 只做 Adapter 接入，不新增需要训练的 observed-motion condition embedder。

## 9. 验证与修复口径

```text
09 Tournament
= relative selection

10 Multi-Verifier
= absolute pass/fail

11 Diagnosis
= failure interpretation + RepairProposal

03 Planner
= final next-action decision
```

Verifier 使用 Required-Check Gating，而不是允许严重失败被其他高分抵消的总加权分。

## 9.1 Cross-Segment Continuity 口径

统一定义：

```text
Motion Compiler
= continuity expectation producer

Constraint Compiler
= persistent measurable heading control producer

Tournament
= relative heading-continuity evidence

Multi-Verifier
= absolute heading-continuity PASS/FAIL

Diagnosis
= root-cause + repair proposal

Planner
= choose ONE existing action
```

V1 不新增 Planner Action，也不新增独立 Continuity Agent。

默认优先级：

```text
user explicit orientation
> trajectory explicit orientation policy
> inherit_previous continuity
> free
```

`inherit_previous` 不是 hard freeze。默认只形成 semantic cue / evaluation expectation；持续失败时才升级为 soft root-heading control。

统一 Failure Code：

```text
UNCOMMANDED_HEADING_DRIFT
```

禁止 Tournament / Verifier 在 MotionSpecification 没有 continuity expectation 时自行创造“必须朝前”的要求。

## 10. 实现前仍需实验校准的参数

以下参数明确不在文档中假装有普适固定值：

```text
MotionCritic pass threshold
TMR semantic threshold
Foot-skating threshold
Ground-penetration threshold
IK tolerance default
Keyframe position / rotation threshold
Hard-mask density warning level
Retrieval low-confidence threshold
Candidate K default
Generation / repair budget defaults
```

统一通过 validation / calibration config 版本化确定。

## 11. V1 明确暂缓

```text
GEM retraining / finetuning
semantic ReAlign reward-model adaptation
DNO noise optimization implementation
full physics simulator
arbitrary object collision
learned repair router
learned verifier router
B=K GEM batched sampling
```

这些不是完成 V1 闭环的前置条件。

## 12. 已知需要运行时处理的外部信息缺口

### Scene Geometry

类似“右手接触桌面”如果没有 scene geometry / target XYZ / usable reference，系统禁止凭空生成绝对坐标。V1 必须返回 `needs_geometry`、退化为可验证的 soft relation，或通过 Planner 选择 Retrieval。

### User Clarification

当前 7-action V1 没有独立 `ASK_USER`。因此不可解析的 scene-specific target 在 V1 中应显式失败/降级，而不是让 LLM 假装知道。交互式 authoring 版本可后续增加独立 external-context / user-input capability。

## 13. 文档完整性

最终包预期包含：

- `README.md`
- `00_project_overview.md`
- `PROJECT_IMPLEMENTATION_PLAN.md`
- `03_motion_planner.md`
- `04_unified_state.md`
- `05_motion_compiler.md`
- `06_retrieval_tool.md`
- `07_constraint_compiler.md`
- `07b_keyframe_tool.md`
- `08_gem_generation_tool.md`
- `09_candidate_tournament.md`
- `10_multi_verifier.md`
- `11_diagnosis_repair.md`
- `12_system_integration.md`
- `industrial_harness_design.md`
- `CONSISTENCY_AUDIT.md`

## 14. 文件指纹（便于确认本 ZIP 内版本）

| File | SHA256 前 16 位 | Bytes |
|---|---:|---:|
| `00_project_overview.md` | `b80d9bb870f634c0` | 28700 |
| `PROJECT_IMPLEMENTATION_PLAN.md` | `546163b77fccbf4c` | 42139 |
| `03_motion_planner.md` | `99204dd543f119a2` | 23213 |
| `04_unified_state.md` | `b6a0aa0eb9a5da1c` | 25791 |
| `05_motion_compiler.md` | `a35cde94687d6eab` | 24430 |
| `06_retrieval_tool.md` | `10a7991fd7774405` | 30376 |
| `07_constraint_compiler.md` | `3a3c98d7a77ea1f7` | 48169 |
| `07b_keyframe_tool.md` | `96126bf4a5e6b077` | 25738 |
| `08_gem_generation_tool.md` | `3d95ce722667dbcb` | 42669 |
| `09_candidate_tournament.md` | `804235db021fa737` | 45973 |
| `10_multi_verifier.md` | `5b8d115ed9901440` | 52901 |
| `11_diagnosis_repair.md` | `ac7fc4961c536e7b` | 56393 |
| `12_system_integration.md` | `c697c14e5b26752f` | 13702 |
| `industrial_harness_design.md` | `0447068c881c1785` | 28846 |
| `README.md` | `38fe3e47886e8fa7` | 2447 |
