# MotionAgent for GEM：Phased Project Implementation Plan

> **用途**
>
> 本文件是 MotionAgent V1 的具体实施计划。后续开发应按阶段推进；每一阶段必须完成对应测试 Gate，才能进入下一阶段。
>
> Source of Truth：
>
> - `00_project_overview.md`
> - `03_motion_planner.md`
> - `04_unified_state.md`
> - `05_motion_compiler.md`
> - `06_retrieval_tool.md`
> - `07_constraint_compiler.md`
> - `07b_keyframe_tool.md`
> - `08_gem_generation_tool.md`
> - `09_candidate_tournament.md`
> - `10_multi_verifier.md`
> - `11_diagnosis_repair.md`
> - `12_system_integration.md`
> - `industrial_harness_design.md`
> - `CONSISTENCY_AUDIT.md`
>
> V1 固定边界：
>
> ```text
> Frozen GEM
> Frozen T5
> No GEM finetuning
> ```

> V1 orchestration runtime：
>
> ```text
> LangGraph StateGraph
> + typed MotionAgent domain state
> + persistent checkpointer / resume
> ```
>
> LangGraph 只负责 orchestration/runtime；Planner policy、domain state、Tool Contract、Artifact Store、Guard、GPU Worker 仍由 MotionAgent 自己定义。
>
> 唯一 Planner Action Space：
>
> ```text
> COMPILE_MOTION
> RETRIEVE_REFERENCE
> BUILD_CONSTRAINT
> BUILD_KEYFRAME
> GENERATE
> ACCEPT
> STOP_FAILED
> ```

---

# 1. 使用本计划的规则

每个阶段都按照：

```text
Read
→ Implement
→ Unit Test
→ Contract Test
→ Integration Test
→ Stage Gate
```

执行。

禁止：

```text
上一阶段核心接口还不稳定
→ 直接继续堆下一阶段 Agent 功能
```

任何阶段如果修改了 canonical schema，必须重新执行：

```text
tests/contracts/
+
受影响模块 integration tests
```

每个阶段结束后建议保存：

```text
stage_report.md
test_result.json
config_snapshot.yaml
git commit / tag
known_issues.md
```

---

# 2. 测试分层

整个项目统一使用以下测试层级：

```text
tests/
│
├── unit/
├── contracts/
├── integration/
├── gpu/
├── e2e/
├── regression/
├── failure_injection/
└── fixtures/
```

## 2.1 Unit Test

测试单模块纯逻辑：

```text
Schema
Validator
Reducer
Query Builder
Timeline
Mask
Metric
Router
```

特点：

```text
不调用真实 LLM
不调用真实 GEM
尽可能 deterministic
```

## 2.2 Contract Test

测试 Producer → Consumer 接口。必须长期保持：

```text
05 GEMTextCondition
→ 08 Generator

07 ConstraintBundle
→ 08 Generator

07 VerificationSpec
→ 10 Verifier

07B KeyframeSpec
→ 08 Generator

07B KeyframeVerificationSpec
→ 10 Verifier

08 GenerationResult
→ 09 Tournament

09 TournamentResult
→ 10 Verifier

10 VerificationReport
→ 11 Diagnosis

11 RepairProposal
→ 03 Planner
```

## 2.3 Integration Test

测试两个或多个真实模块连接，例如：

```text
Compiler
→ Generator

Retrieval
→ Compiler revise

Constraint
→ Generator
→ Verifier
```

## 2.4 GPU Test

真实运行 Frozen GEM。GPU Test 不应作为每次普通 CI 的必跑项。

建议：

```text
normal CI
→ unit + contracts + mock integration

GPU CI / manual
→ GEM smoke + generation regression
```

## 2.5 E2E Test

运行完整：

```text
User
→ Planner
→ Tools
→ Generate
→ Tournament
→ Verify
→ Diagnose
→ Planner
```

## 2.6 Regression Test

每次以下内容升级后运行固定 Benchmark：

```text
Prompt
Model
Threshold
Schema
GEM Adapter
```

---

# 3. Phase 0 — Environment / GEM Baseline Freeze

## 目标

先证明：

> **当前 GEM 环境本身可正常工作，并冻结 V1 baseline。**

不要一开始就写 Agent。

## 阅读文件

重点：

```text
00_project_overview.md
→ #0–#5

08_gem_generation_tool.md
→ #8.3
→ #8.8
→ #8.9
→ #8.10
→ #8.19

CONSISTENCY_AUDIT.md
→ Frozen GEM / V1 boundary
```

同时阅读 GEM 官方 repo 的：

```text
README
configs/exp/gem_smpl.yaml
configs/model/gem.yaml
scripts/demo/demo_smpl.py
gem/gem.py
gem/network/endecoder.py
```

## 执行操作

1. 建立 Python / CUDA 环境。
2. 下载 GEM checkpoint。
3. 跑官方 demo。
4. 保存：
   - checkpoint hash；
   - GEM config；
   - CUDA / PyTorch 版本；
   - demo output；
   - baseline latency / VRAM。
5. 写 `configs/model_versions.yaml` 固定 V1：
   - GEM checkpoint；
   - T5 version；
   - sampler config。

## 达成结果

得到明确的：

```text
Frozen GEM Baseline
```

后续 Agent 所有结果都基于这一版本。

## 测试

### Test 0.1 — Official Smoke Test

运行官方 GEM demo。

Pass：

```text
checkpoint 成功加载
inference 无异常
输出 SMPL / pred_x
无 NaN / Inf
```

### Test 0.2 — Deterministic Environment Record

保存：

```text
checkpoint SHA
config hash
torch version
cuda version
GPU model
```

重新启动后能读取相同版本。

### Test 0.3 — Baseline Artifact

至少保存 3 个 baseline：

```text
simple walking
sitting
simple multi-action / official example
```

作为后续 regression reference。

## Stage Gate

只有同时满足：

```text
[ ] GEM 官方 inference 成功
[ ] checkpoint/config 版本已冻结
[ ] baseline artifacts 已保存
[ ] basic latency/VRAM 已记录
```

才进入 Phase 1。

---

# 4. Phase 1 — Canonical Schema + Unified State + LangGraph Persistence Core

## 目标

建立整个项目的数据底座：

```text
ToolResult
→ Artifact
→ StateReducer
→ Commit
```

所有模块禁止直接互相修改 State。

## 阅读文件

```text
04_unified_state.md
→ #4.2–#4.35
→ #4.36–#4.38

12_system_integration.md
→ #12.3–#12.8
→ #12.13–#12.24

industrial_harness_design.md
→ #3–#14
→ #20–#21
→ #28–#30

00_project_overview.md
→ #17–#21
```

## 执行操作

优先实现：

```text
common/schemas/
common/ids.py
common/fingerprints.py

state/schemas.py
state/store.py
state/reducer.py
state/versioning.py
state/summaries.py

stores/artifact_store.py
event_log.py

graph/state.py
graph/checkpoints.py
```

按照 `04_unified_state.md #4.36`：

```text
1. canonical schema
2. artifact store
3. reducer
4. invariants
5. PlannerContextBuilder
6. event log
7. version / concurrency
8. MotionGraphState adapter
9. persistent LangGraph checkpointer
10. canonical-state reconciliation / resume
```

## 达成结果

得到：

```text
MotionAgentState
ArtifactStore
EventLog
StateReducer
PlannerContextBuilder
MotionGraphState adapter
LangGraph persistent checkpointer
```

且大型 Tensor 不进入 State。

## 测试

### Test 1.1 — State Unit Fixtures

覆盖 `04 #4.37`：

```text
Create Run
Plan Commit
Constraint Add / Replace / Remove
Keyframe Add / Replace / Remove
Generation Commit
Champion Update
Verification Commit
Diagnosis Commit
```

所有 fixture 使用固定 JSON。

### Test 1.2 — State Invariant

主动构造非法状态：

```text
budget < 0
accepted without verification
missing artifact
constraint references missing spec
```

Expected：

```text
commit 被拒绝
```

### Test 1.3 — No Tensor in Planner State

构造真实或 fake：

```text
[L,151] motion tensor
```

尝试提交 State。

Expected：

```text
State 只保存 handle
PlannerContext 中不存在 tensor / huge arrays
```

### Test 1.4 — Optimistic Concurrency

两个 updater 同时基于：

```text
state_version = N
```

提交。

Expected：

```text
一个成功
一个 version conflict
```

### Test 1.5 — LangGraph Checkpoint / Resume

流程：

```text
create run (thread_id = run_id)
→ commit plan
→ checkpoint graph
→ 模拟进程退出
→ reload LangGraph checkpoint
→ reload latest canonical MotionAgentState
→ reconcile state_version
→ continue from correct next node
```

Expected：

```text
恢复后的 canonical state 与退出前一致
Graph 从正确 node 继续
已 commit 的昂贵 Tool 不重复执行
```

### Test 1.6 — Context Growth

模拟：

```text
20 rounds
```

Event Log 不断增加。

Expected：

```text
PlannerContext token/字符长度保持在配置预算附近
而不是随 Event Log 线性增长
```

## Stage Gate

```text
[ ] 04 #4.37 全部测试通过
[ ] LangGraph persistent checkpoint / crash-resume 可工作
[ ] GraphState 与 canonical State version 对齐
[ ] State 中无大型 Tensor
[ ] context 不依赖完整 history
[ ] canonical schema 只有一份定义
```

---

# 5. Phase 2 — Planner / LangGraph Orchestrator Skeleton with Mock Tools

## 目标

先让：

```text
LangGraph START
→ Planner
→ ONE Action
→ conditional route
→ Mock Tool Node
→ State Commit
→ Planner
```

完整跑通。

此阶段不依赖真实 GEM。

## 阅读文件

```text
03_motion_planner.md
→ #3.1–#3.11

12_system_integration.md
→ #12.3–#12.10
→ #12.25–#12.28

industrial_harness_design.md
→ #8–#10
→ #15–#19
→ #38–#40
```

## 执行操作

实现：

```text
agent/actions.py
agent/planner.py
agent/context_builder.py
agent/guards.py
agent/budgets.py

app/orchestrator.py

graph/builder.py
graph/state.py
graph/routing.py
graph/nodes/planner.py
graph/nodes/tools.py
graph/nodes/post_generation.py

request_builder_registry
action_executor_registry
```

先给 5 个执行 Action 接 Mock：

```text
COMPILE_MOTION
RETRIEVE_REFERENCE
BUILD_CONSTRAINT
BUILD_KEYFRAME
GENERATE
```

## 达成结果

Planner + LangGraph 能：

```text
读取 PlannerContext
输出严格 PlannerDecision
通过 Action Guard
固定 Enum → Node conditional routing
执行 Mock node
commit canonical state
checkpoint graph state
进入下一轮
```

## 测试

### Test 2.0 — Graph Topology / Routing

验证：

```text
START → planner
5 个执行 Action → 对应 Tool node → planner
GENERATE → generation → tournament → verifier → diagnosis → planner
ACCEPT / STOP_FAILED → END
```

Expected：每个 Action 恰好映射一个合法 route；非法 action / 任意 node name 无法进入 graph。

### Test 2.1 — Action Schema

对每一个 Action 构造合法 / 非法 payload。

Expected：

```text
合法通过
非法被 schema 拒绝
```

### Test 2.2 — Hard Guard

对应 `03 #3.7`：

```text
no plan → GENERATE
```

必须拒绝。

```text
no verification → ACCEPT
```

必须拒绝。

```text
budget=0 → GENERATE
```

必须拒绝。

### Test 2.3 — Accept Guard

输入：

```text
overall_pass=False
```

即使 Planner LLM 输出：

```text
ACCEPT
```

Expected：

```text
代码 guard 拒绝
```

### Test 2.4 — Automatic Subgraph Stub

Mock `GENERATE` 后自动执行：

```text
mock generation node
→ mock tournament node
→ mock verifier node
→ mock diagnosis node
```

Expected：

```text
不再次调用 Planner 决定是否 VERIFY
```

### Test 2.5 — One-Action Rule

每轮只能 commit：

```text
ONE PlannerDecision
```

禁止一个 LLM response 同时：

```text
retrieve + constraint + generate
```

### Test 2.6 — Mock E2E

输入：

```text
walk forward
```

Mock 流程：

```text
COMPILE_MOTION
→ GENERATE
→ mock pass
→ ACCEPT
```

Expected：

```text
完整 trace 正确
```

## Stage Gate

```text
[ ] LangGraph topology / conditional routing tests 通过
[ ] 7 Action 全部有 schema
[ ] Hard Guards 全通过
[ ] automatic post-generation stub 跑通
[ ] Mock E2E 可 ACCEPT
[ ] State 每步均通过 reducer commit
```

---

# 6. Phase 3 — Motion Compiler

## 目标

完成：

```text
Natural Language
→ Human Motion DSL
→ Timeline
→ Routing Hints
→ GEMTextCondition
```

## 阅读文件

```text
05_motion_compiler.md
→ #5.4–#5.20
→ #5.21–#5.23

00_project_overview.md
→ #8

03_motion_planner.md
→ COMPILE_MOTION
```

## 执行操作

按照 `05 #5.19–#5.21`：

```text
semantic_parser.py
motion_dsl.py
timeline_compiler.py
control_intent.py
caption_optimizer.py
gem_text_compiler.py
validators.py
```

建立：

```text
golden compiler fixtures
```

保存预期 JSON。

## 达成结果

`COMPILE_MOTION` 可以真实替换 Phase 2 的 Mock。

## 测试

执行 `05 #5.22` 全部 Case：

```text
single action
sequential
simultaneous
explicit time
implicit time
repetition
rare style
contact
trajectory
keyframe
long prompt
revise one segment
```

### Test 3.1 — Golden DSL

示例：

```text
wave the right hand three times and sit down
```

必须：

```text
2 segments
right_hand
repetition=3
correct order
```

### Test 3.2 — Timeline Invariant

验证：

```text
start_0 = 0
end_last = total_duration
no illegal overlap
frame boundary continuous
```

### Test 3.3 — Routing Hint Boundary

例如 contact：

Expected：

```text
constraint_candidate
```

但：

```text
没有直接创建 Constraint
```

### Test 3.4 — Caption Semantic Preservation

对 golden fixtures：

```text
DSL required fields
```

必须都能在 structured caption validation 中被覆盖。

不能：

```text
增加不存在的动作
```

### Test 3.5 — `GEMTextCondition` Contract

验证：

```text
captions count
==
window_start count
==
window_end count

0 <= window <= 1
```

然后通过 08 的 adapter schema test。

### Test 3.6 — Cross-Segment Heading Continuity

Golden cases：

```text
walk forward → wave right hand → sit down
```

Expected：

```text
后两段 inherit_previous
```

```text
walk forward → turn right → sit down
```

Expected：

```text
turn 更新 heading anchor
sit 继承新 heading
```

同时测试：

```text
spin / pirouette
→ 不得被错误固定朝向
```

如果 Phase 3 在本规则加入前已经完成，必须补跑这一组 Compiler schema / caption / validator regression；不要求重做 Phase 0–2。

## Stage Gate

```text
[ ] 05 #5.22 全部 deterministic tests 通过
[ ] golden DSL fixtures 通过
[ ] timeline invariants 全部通过
[ ] GEMTextCondition contract 通过
[ ] Cross-Segment Heading Continuity fixtures 通过
[ ] COMPILE_MOTION 已替换 Mock
```

---

# 7. Phase 4 — Minimal Pure-Text GEM Generation

## 目标

尽早建立第一条真实闭环：

```text
User
→ Compiler
→ Frozen GEM
→ Motion Candidate
```

此阶段只做：

```text
full scope
single candidate
text only
```

## 阅读文件

```text
08_gem_generation_tool.md
→ #8.3–#8.19
→ #8.37–#8.44

05_motion_compiler.md
→ #5.11–#5.13

04_unified_state.md
→ Artifact / GenerationState
```

## 执行操作

优先完成 `08 #8.42 Step 1–4`：

```text
GEMModelManager
PureTextGEMAdapter
CandidateStore
SeedManager
```

先不要实现：

```text
K tournament
constraint
segment repair
guided
```

## 达成结果

真实：

```text
MotionSpecification
+
GEMTextCondition
→ GEM
→ MotionCandidate
```

## 测试

### Test 4.1 — Pure Text Single Segment

输入：

```text
walk forward
```

Expected：

```text
不需要 video preprocessing
all visual masks = false
static synthetic camera
pred_x exists
global SMPL exists
```

### Test 4.2 — Multi-Text

输入 3 segment。

Expected：

```text
multi_text_data contains 3 captions
window 不被 adapter 修改
frame count 正确
```

### Test 4.3 — Same Seed Reproducibility

固定：

```text
checkpoint
config
condition
seed
```

运行两次。

Expected：

```text
candidate fingerprint 相同
pred_x 在数值容差内相同
```

### Test 4.4 — Different Seed

不同 seed。

Expected：

```text
输出不完全相同
```

该测试只验证 sampling diversity 存在，不要求具体质量提升。

### Test 4.5 — Technical Output

必须：

```text
no NaN
no Inf
pred_x last_dim = 151
frame count correct
SMPL keys complete
```

### Test 4.6 — Continuity Metadata / Caption Propagation

使用：

```text
walk forward → wave right hand → sit down
```

验证：

```text
Compiler continuity policy 能进入真实 generation request；
caption 中 continuity cue 与 DSL 一致；
Candidate lineage 可追溯到对应 MotionSpecification；
Generator 不自行创造或修改 heading policy。
```

该测试只验证 Phase 4 contract，不要求 Frozen GEM 在所有 seed 上严格满足 heading continuity；真正的 selection / absolute verification 在 Phase 8 / 9 完成。

Phase 3 continuity patch 完成后，Phase 4 只需要补跑该 regression，不需要重构 GEM Generator。

## Stage Gate

```text
[ ] real Frozen GEM pure-text generation 跑通
[ ] single + multi-text 跑通
[ ] same seed reproducible
[ ] CandidateStore 可追溯
[ ] continuity metadata / caption propagation 通过
[ ] Compiler → Generator contract 真实通过
```

---

# 8. Phase 5 — Retrieval

## 目标

实现：

```text
Text / Segment
→ HumanML3D / TMR
→ RetrievedReference
```

并验证 Retrieval 可以被下游消费。

## 阅读文件

```text
06_retrieval_tool.md
→ #6.4–#6.29
→ #6.30–#6.33

05_motion_compiler.md
→ retrieval-grounded caption

07_constraint_compiler.md
→ retrieved reference use

07b_keyframe_tool.md
→ source resolution
```

## 执行操作

实现：

```text
HumanML3D canonical ID mapping
Caption Index
TMR Motion Index
Query Builder
Top-N Search
Filter
Reranker
Reference Handle
```

## 达成结果

真实：

```text
RETRIEVE_REFERENCE
```

替换 Mock。

## 测试

执行 `06 #6.31`。

### Test 5.1 — ID Mapping Integrity

遍历可检索 train corpus。

Expected：

```text
每个 motion reference
→ 能定位 canonical motion_id

需要 SMPL 的 reference
→ 能定位 GEM-compatible record
```

### Test 5.2 — Leakage

构造 test split query。

Expected：

```text
test item 不可能被 Retrieval Index 返回
```

这是 hard test。

### Test 5.3 — Known-Pair Recall Sanity

从 HumanML3D train/val 构造：

```text
caption → its own motion
```

查询。

记录：

```text
Recall@1
Recall@5
Recall@10
Median Rank
```

阶段完成前必须：

```text
指标可稳定复现
```

这里先形成 baseline，不提前规定论文最终阈值。

### Test 5.4 — Semantic Sanity Set

人工准备至少 20 个明显 Query：

```text
limping
sitting
jumping
waving
turning
...
```

检查 Top-5。

Pass 条件：

```text
不存在明显完全相反类别占据全部 Top-K
并记录人工 relevance
```

### Test 5.5 — Retrieval → Compiler Integration

流程：

```text
rare style
→ retrieve caption
→ COMPILE_MOTION revise gem_caption
```

Expected：

```text
DSL 不改变
caption 被 refinement
```

## Stage Gate

```text
[ ] train-only index
[ ] canonical mapping 通过
[ ] caption/motion retrieval 均可工作
[ ] Recall metrics 已建立
[ ] Retrieval → Compiler contract 跑通
```

---

# 9. Phase 6 — Constraint + Keyframe Shared Control Infrastructure

## 目标

一次完成两者共同依赖的：

```text
GEM Encoder
Joint Registry
FK / IK
Feature Mask
HardMotionCondition
VerificationSpec
Conflict Detection
```

再分别完成 Constraint 和 Keyframe。

## 阅读文件

```text
07_constraint_compiler.md
→ #7.5–#7.48
→ #7.49–#7.53

07b_keyframe_tool.md
→ #7B.5–#7B.49
→ #7B.50–#7B.54

08_gem_generation_tool.md
→ hard condition injection

10_multi_verifier.md
→ constraint/keyframe verification
```

## 执行操作

### Shared

实现：

```text
Joint Registry
Coordinate Resolver
Time Resolver
GEM Feature Mapper
GEM Encoder
Feature Mask Builder
Condition Store
```

### Constraint

按 `07 #7.49`：

```text
body_part_pose
joint_target + IK
soft reward
root trajectory
contact
fixed_joint
composer
```

### Keyframe

按 `07B #7B.50`：

```text
request/spec/result
source resolver
candidate pose extraction
whole-body pose encode
verification spec
conflict detection
IK refinement
```

## 达成结果

真实：

```text
BUILD_CONSTRAINT
BUILD_KEYFRAME
```

替换 Mock。

## 测试

### Constraint

执行 `07 #7.51`，必须包括：

```text
joint target
body part pose
sparse mask
trajectory
contact
fixed foot
IK fail
conflicting constraints
encode/decode consistency
```

### Keyframe

执行 `07B #7B.52`，必须包括：

```text
add/replace/remove
retrieved pose
retrieved motion frame
candidate source
no reference
IK success/fail
frame boundary
root orient
body shape not masked
multiple keyframes
verification spec
```

### Test 6.1 — FK/IK Numerical Test

使用 synthetic known pose：

```text
FK(base pose)
→ choose wrist XYZ
→ perturb pose
→ IK to target
→ FK(solved)
```

Pass：

```text
residual <= configured tolerance
```

### Test 6.2 — GEM Encode/Decode Consistency

构造合法 SMPL pose / short sequence：

```text
encode
→ decode
```

检查 rotation / pose difference 在数值容差内。

### Test 6.3 — Hard Mask Correctness

例如：

```text
right elbow at frame 90
```

Expected：

```text
只有对应 frame + joint feature slice mask=1
betas 默认不被 mask
```

### Test 6.4 — Constraint/Keyframe Conflict

构造：

```text
constraint pose X
keyframe pose Y
same frame / same joint
```

Expected：

```text
conflict detected
GENERATION blocked
```

### Test 6.5 — No Geometry

输入：

```text
touch table
```

但无：

```text
scene geometry
reference
```

Expected：

```text
needs_geometry / soft fallback
绝不能凭空生成 table XYZ
```

### Test 6.6 — Orientation-Only Heading Control

构造：

```text
root_trajectory
points = None
orientation_policy = inherit_anchor
```

Expected：

```text
soft root_heading_continuity RewardSpec
position 不被约束
VerificationSpec 创建
explicit turn 更新 anchor
```

默认不得 hard-freeze full root orientation。

## Stage Gate

```text
[ ] 07 #7.51 全部通过
[ ] 07B #7B.52 全部通过
[ ] FK/IK numerical tests 通过
[ ] GEM encode/decode 通过
[ ] hard mask/verification spec 对应正确
[ ] orientation-only soft heading control 通过
[ ] conflict 能阻止 generate
```

---

# 10. Phase 7 — Complete Generation Tool

> Revised Phase 7 gate: complete generation infrastructure, composite semantic
> compilation and condition interfaces. Hard-condition satisfaction, exact
> keyframe forcing and true diffusion-time inpainting are DEFERRED to Guided
> Generation and do not block this milestone. Retain their measured errors.
> The original acceptance targets below describe future satisfaction goals
> where they exceed this revised scope. See module 08's Phase 7 scope note.

## 目标

把 Phase 4 的最小 Generator 升级成正式 08：

```text
K candidates
Hard Condition
Keyframe
Segment Inpainting
Worker / Queue
Cache
Postprocess Policy
```

## 阅读文件

```text
08_gem_generation_tool.md
→ #8.5–#8.46

07_constraint_compiler.md
→ ConstraintBundle

07b_keyframe_tool.md
→ KeyframeSpec / HardMotionCondition

industrial_harness_design.md
→ GPU worker / queue / idempotency
```

## 执行操作

继续 `08 #8.42 Step 5–10`：

```text
K loop
hard condition injection
segment inpainting
postprocess policy
GPU worker
queue
candidate cache
guided feature gate
```

V1 先不实现真实 guided sampling。

## 达成结果

正式：

```text
GenerationRequest
→ K MotionCandidate
→ GenerationResult
```

## 测试

执行 `08 #8.44` 全部。

### Test 7.1 — K Candidates

```text
K=4
```

Expected：

```text
4 independent candidate IDs
4 seed records
same condition fingerprint
```

### Test 7.2 — Hard Condition Preservation

构造已知 hard pose。

生成后直接测：

```text
masked feature error
```

必须处于数值容差内。

### Test 7.3 — Keyframe Preservation

使用 Phase 6 Keyframe。

Expected：

```text
target frame pose 接近 KeyframeSpec
```

验证使用同一 `KeyframeVerificationSpec`。

### Test 7.4 — Segment Inpainting

上一 Candidate：

```text
segment 0 / 1 / 2
```

只重生成 1。

Expected：

```text
segment 0 / 2 preservation error 很小
betas unchanged
segment 1 editable
```

### Test 7.5 — Partial Failure

Mock 其中一个 candidate runtime error。

Expected：

```text
status=partial
成功 candidate 不丢失
```

### Test 7.6 — Cache

同：

```text
condition fingerprint
seed
sampler
```

再次提交。

Expected：

```text
直接返回 existing candidate
不再次执行 GEM
```

通过 worker inference-call counter 验证。

### Test 7.7 — OOM Failure Injection

Mock / 限制显存触发 OOM。

Expected：

```text
candidate failure recorded
worker cleanup
不修改 prompt/condition
```

## Stage Gate

```text
[ ] 08 #8.44 全部通过
[ ] K sampling 稳定
[ ] hard/keyframe condition preserved
[ ] segment outside region preserved
[ ] duplicate request 不重复 GPU work
[ ] Worker health 可观测
```

---

# 11. Phase 8 — Candidate Tournament

## 目标

实现：

```text
GenerationResult
→ Champion
```

只做相对选择，不做最终 pass/fail。

## 阅读文件

```text
09_candidate_tournament.md
→ #9.5–#9.52
→ #9.53–#9.59

08_gem_generation_tool.md
→ Candidate Store

10_multi_verifier.md
→ Tournament / Verifier boundary
```

## 执行操作

按 `09 #9.53`：

```text
schema
candidate store
renderer
quick metrics
probe
pairwise judge
swap
tiebreak
bracket
cache/service
```

## 达成结果

automatic subgraph 中：

```text
K candidates
→ TournamentResult
→ champion_candidate_id
```

## 测试

执行 `09 #9.55`。

### Test 8.1 — Deterministic Mock Judge

使用 fake judge：

```text
A always better than B
```

验证：

```text
K=2
K=3
K=4
bye
bracket
incumbent
```

必须 100% 得到预期 winner。

### Test 8.2 — Position Swap

Mock：

```text
forward=A
swapped=A
```

Expected：

```text
bidirectional_consensus
```

Mock：

```text
forward=A
swapped=B
```

Expected：

```text
进入 tiebreak
不随机决胜
```

### Test 8.3 — Stable Fallback

所有 Judge / metrics 返回 tie。

运行两次。

Expected：

```text
相同 tournament_id / candidate set
→ 相同 deterministic winner
```

### Test 8.3A — Heading Continuity Ranking

准备 pair：

```text
A: 动作正确且保持 inherited heading
B: 动作正确但无指令转身
```

在存在 `HeadingContinuitySpec` 时：

```text
continuity_preservation 应偏好 A
```

再准备 explicit-turn case，确保正确转向不会被惩罚。

### Test 8.4 — Human-Labeled Sanity Set

准备至少：

```text
20 个 obvious motion pairs
```

例如：

```text
正确动作 vs 明显错误动作
自然 vs 明显抖动
constraint satisfied vs severe violation
```

让人工给 preferred candidate。

记录：

```text
Pairwise Human Agreement
Forward/Swap Consistency
```

建议初始工程 sanity gate：

```text
>= 80% human agreement
```

如果低于该值，先修 Judge / Render，不进入下一阶段。

## Stage Gate

```text
[ ] 所有 deterministic tournament tests 通过
[ ] 无随机 tie resolution
[ ] render camera 一致
[ ] cache/idempotency 通过
[ ] heading continuity ranking / explicit-turn exemption 通过
[ ] obvious-pair human agreement 达到 sanity gate
```

---

# 12. Phase 9 — Multi-Verifier

## 目标

实现：

```text
Champion
→ VerificationPlan
→ VerifierFinding[]
→ VerificationReport
```

并真正控制：

```text
overall_pass
```

## 阅读文件

```text
10_multi_verifier.md
→ #10.4–#10.85
→ #10.86–#10.93

07_constraint_compiler.md
→ VerificationSpec

07b_keyframe_tool.md
→ KeyframeVerificationSpec

09_candidate_tournament.md
→ relative vs absolute boundary
```

## 执行操作

按 `10 #10.86`：

```text
schema + plan builder
technical
constraint
TMR
MotionCritic
AToM event
physical
keyframe
preservation
aggregator
cache/service
```

## 达成结果

动态选择：

```text
semantic
event
naturalness
constraint
keyframe
physical
preservation
```

并使用：

```text
Required-Check Gating
```

而不是加权总分。

## 测试

执行 `10 #10.88`。

### Test 9.1 — VerifierPlan Selection

Golden cases：

```text
walk forward
wave 3 times
wave then walk then sit
contact
keyframe
segment retry
```

Expected：

```text
只启用需要的 verifier
```

### Test 9.1A — Heading Continuity Verifier

Fixture：

```text
inherit_previous + small drift
→ PASS

inherit_previous + large sustained uncommanded turn
→ UNCOMMANDED_HEADING_DRIFT
→ overall_pass=false

explicit turn
→ transition window exempt
```

阈值从 config/calibration 读取，不由测试中的 LLM 产生。

### Test 9.2 — Synthetic Constraint Failures

直接构造 Candidate / metric fixture：

```text
error 0.08
threshold 0.05
```

Expected：

```text
fail
overall_pass=false
```

### Test 9.3 — Event Failure Fixtures

准备人工可验证 Motion clips：

```text
missing event
wrong order
2 vs 3 repetitions
```

Expected：

```text
Integrity/Temporal/Frequency
分别输出正确 diagnostic code
```

### Test 9.4 — Physical Failure Injection

在一个正常 Motion 上人工加入：

```text
foot slide
ground penetration
jitter
```

Expected：

```text
对应 metric 明显恶化
```

### Test 9.5 — Required Uncertain

Mock required MLLM verifier：

```text
uncertain
```

Expected：

```text
overall_pass=false
status=incomplete
```

### Test 9.6 — False Accept Calibration Set

建立人工标注：

```text
clear pass
clear fail
```

小集合。

重点记录：

```text
False Accept Rate
False Reject Rate
```

其中 deterministic explicit failures：

```text
constraint
missing event
wrong count
```

在 fixture 集上要求：

```text
False Accept = 0
```

对于 learned / perceptual verifier，通过 calibration set 决定 threshold，不提前硬编码论文最终数值。

## Stage Gate

```text
[ ] 10 #10.88 全部通过
[ ] required gating 正确
[ ] heading continuity verifier / explicit-turn exemption 通过
[ ] deterministic failure fixtures 0 false accept
[ ] threshold profile 已通过 calibration 固定版本
[ ] VerificationReport 可直接被 11 消费
```

---

# 13. Phase 10 — Diagnosis & Targeted Repair

## 目标

实现：

```text
VerificationReport
→ DiagnosisResult
→ RepairProposal[]
```

不直接执行 Repair。

## 阅读文件

```text
11_diagnosis_repair.md
→ #11.5–#11.84
→ #11.85–#11.93

03_motion_planner.md
→ Action / Reason Codes / guards

10_multi_verifier.md
→ diagnostic codes
```

## 执行操作

按 `11 #11.85`：

```text
schemas
failure normalizer
failure cluster
rule repair policy
proposal validator
repair history
specialized proposal builders
selective diagnosis LLM
no-improvement guard
service/store/cache
```

## 达成结果

每一个 Failure：

```text
Observed Failure
→ Root Cause Hypothesis
→ Planner-compatible RepairProposal
```

## 测试

执行 `11 #11.87`。

### Test 10.1 — Rule Fixture Table

对每一个主要 diagnostic code，输入固定 `VerificationFinding`。

检查：

```text
failure family
root cause
repair family
planner action
```

所有 deterministic rule fixture 必须 100% 通过。

### Test 10.2 — Proposal Contract

每一个 Proposal 必须能被：

```text
03 Planner schema
+
对应 Request Builder
```

接受。

### Test 10.3 — No Improvement

模拟：

```text
same failure
same repair family
unchanged
unchanged
```

Expected：

```text
same family blocked
```

### Test 10.4 — Improvement Tracking

例如 contact：

```text
0.28 → 0.12 → 0.04
threshold=0.05
```

Expected：

```text
improved
resolved
```

顺序正确。

### Test 10.5 — Preservation

修复一个 segment 后引入新的 major failure。

Expected：

```text
repair outcome != resolved
记录 regression
```

### Test 10.6 — Ambiguous Diagnosis

给证据不足 Case。

Expected：

```text
UNKNOWN / ambiguous
```

不能强制编造 root cause。

### Test 10.7 — Continuity Repair Routing

验证：

```text
missing continuity policy
→ COMPILE_MOTION

single-seed drift with correct policy
→ REGENERATION

persistent drift with correct policy
→ CONSTRAINT_UPDATE

explicit turn false alarm
→ verifier/config fix, not motion repair
```

不得新增 Planner Action。

## Stage Gate

```text
[ ] 11 #11.87 全部通过
[ ] deterministic routing fixtures 100% 通过
[ ] Proposal 全部 Planner-compatible
[ ] repair history 能更新 outcome
[ ] heading continuity repair routing 通过
[ ] no-improvement guard 生效
```

---

# 14. Phase 11 — Real End-to-End LangGraph Agent Loop

## 目标

第一次运行完整真实：

```text
Planner
→ Compiler/Retrieval/Constraint/Keyframe
→ GEM
→ Tournament
→ Verifier
→ Diagnosis
→ Planner
```

## 阅读文件

```text
00_project_overview.md
→ #5 / #16 / #18–#22

12_system_integration.md
→ #12.4–#12.28

CONSISTENCY_AUDIT.md
```

## 执行操作

替换所有 Mock。

完整开启：

```text
StateReducer
Planner Guards
Request Builders
Automatic Post-Generation Subgraph
Repair History
Budget
```

## 必须建立的 E2E Fixture Suite

至少以下 8 类。

### E2E 1 — Simple

```text
walk forward
```

期望：

```text
COMPILE
→ GENERATE
→ Tournament
→ Verify
→ ACCEPT
```

### E2E 2 — Sequential

```text
walk forward then sit down
```

检查：

```text
multi-segment
temporal verification
```

### E2E 3 — Frequency

```text
wave right hand three times then sit down
```

必须启用：

```text
Frequency Verifier
```

如果第一次 count 错误，必须能：

```text
Diagnosis
→ COMPILE or regenerate
```

而不是直接 ACCEPT。

### E2E 4 — Rare Motion

```text
walk with an unusual asymmetric limping gait
```

检查 retrieval path；在指定测试策略下能触发：

```text
RETRIEVE_REFERENCE
```

### E2E 5 — Constraint

使用可解析 numerical target：

```text
right wrist reaches XYZ at t=3s
```

检查：

```text
BUILD_CONSTRAINT
→ Generation
→ Constraint Verification
```

### E2E 6 — Keyframe

```text
finish in a stable seated whole-body pose
```

检查：

```text
BUILD_KEYFRAME
→ Generation
→ Keyframe Verification
```

### E2E 7 — Segment Repair

人为让：

```text
segment 1 fail
```

检查：

```text
scope=segment
outside segments preserved
```

### E2E 8 — Persistent Failure

Mock 或构造无法满足 Case。

Expected：

```text
budget / no-improvement guard
→ STOP_FAILED
```

不能无限循环。

## 系统测试

### Test 11.0 — Runtime Graph Integrity

真实 graph 必须验证：

```text
thread_id = run_id
Planner route only uses 7-action mapping
post-generation edges are fixed
terminal nodes reach END
node state contains handles / metadata only
```

### Test 11.1 — Contract Chain

真实或 Mock GPU：

```text
05→08
07→08
07B→08
08→09
09→10
10→11
11→03
```

全部执行一次。

### Test 11.2 — Artifact Lineage

随机选择一个 Candidate。

必须能追溯：

```text
Candidate
→ Generation
→ Condition Fingerprint
→ MotionSpec
→ Constraint/Keyframe
→ Seed
```

### Test 11.3 — Crash Resume

在：

```text
Generation commit 后
Tournament 前
```

kill process。

重启。

Expected：

```text
从最后 commit 继续
不重复 GEM generation
```

### Test 11.4 — Illegal Accept

任意一个 Required Check fail。

Expected：

```text
ACCEPT 永远被 guard 阻止
```

## Stage Gate

```text
[ ] LangGraph runtime topology / checkpoint route 正确
[ ] 8 类 E2E scenario 全部能结束
[ ] 无无限循环
[ ] 无非法 ACCEPT
[ ] 所有 artifact 可追溯
[ ] crash resume 不重复昂贵 Tool
[ ] automatic subgraph 顺序固定
```

---

# 15. Phase 12 — Industrial Harness Hardening

## 目标

从：

```text
research script
```

升级到：

```text
稳定、可恢复、可观测的 Agent Harness
```

## 阅读文件

完整阅读：

```text
industrial_harness_design.md
```

重点：

```text
#3–#10 Context / Tool visibility
#11–#21 Schema / Artifact / Retry / Fingerprint
#22–#30 GPU / Queue / Concurrency / Durable Execution
#31–#40 Verifier cost / cache / security / guard
#42–#54 Logging / tracing / eval / failure injection
#66–#68 Definition of Done
```

以及：

```text
04_unified_state.md
12_system_integration.md
```

## 执行操作

实现 / 固化：

```text
LangGraph Persistent Checkpointer
Checkpoint Retention / Recovery
Planner Context Budget
Dynamic Legal Actions
Skill Cards
Artifact Store
Central Fingerprint
Idempotency
GPU Queue
Backpressure
Worker Health
Typed Errors
Retry Policy
Tracing
Metrics
Prompt/Model/Config Version
Cache Invalidation
Security Guards
```

## 测试

### Test 12.1 — Long-Run Context

模拟：

```text
20–30 rounds
```

Expected：

```text
PlannerContext 不线性膨胀
```

记录：

```text
context size vs round
```

### Test 12.2 — Duplicate GPU Request

同 fingerprint 提交 2 次。

Expected：

```text
只发生 1 次真实 inference
```

### Test 12.3 — Backpressure

同时提交：

```text
> worker capacity
```

的请求。

Expected：

```text
任务进入 queue
显存不被无限占用
worker 不崩溃
```

### Test 12.4 — Worker Kill

生成中 kill GPU worker。

Expected：

```text
job 状态明确
可 retry/resume
State 不出现半 commit
```

### Test 12.5 — State Race

并发两次 State update。

Expected：

```text
version conflict 可检测
```

### Test 12.6 — Retrieval Prompt Injection

在测试 corpus 中放：

```text
"Ignore previous instructions and..."
```

Expected：

```text
它只能被当成 data
不能改变 Planner / Compiler system behavior
```

### Test 12.7 — Cache Invalidation

改变：

```text
prompt version
threshold version
model version
```

Expected：

```text
旧 cache 不再被错误复用
```

### Test 12.8 — Trace Completeness

任取一个最终 ACCEPT run。

必须能够从 trace 找到：

```text
Planner decision
Tool request
Tool output IDs
Generation seed
Tournament
Verification
Diagnosis
Final ACCEPT
```

### Test 12.9 — Failure Injection

至少注入：

```text
LLM timeout
invalid structured output
retrieval empty
GPU OOM
judge timeout
verifier unavailable
state conflict
```

每个 Failure 都必须符合 typed error policy。

## Stage Gate

对应 `industrial_harness_design.md #66`：

```text
[ ] persistent LangGraph checkpointer 可跨进程恢复
[ ] graph checkpoint 不包含重型 Tensor
[ ] context bounded
[ ] crash recoverable
[ ] expensive calls idempotent
[ ] GPU worker queue/health ready
[ ] no infinite repair loop
[ ] prompt/model/config versioned
[ ] tracing complete
[ ] regression suite 可运行
```

---

# 16. Phase 13 — Research Benchmark / Ablation / Paper-Ready Evaluation

## 目标

证明：

> **MotionAgent 的提升来自 Agentic Control，而不是重新训练 GEM。**

## 阅读文件

```text
00_project_overview.md
→ 每个模块 “证明做好的指标”

06 #6.32
07 #7.52
07B #7B.53
08 #8.45
09 #9.56–#9.57
10 #10.89–#10.91
11 #11.88–#11.91
```

## 执行操作

建立固定 Benchmark：

```text
Simple Motion
Sequential Events
Frequency
Rare Motion
Joint Constraint
Contact / Trajectory
Keyframe
Segment Repair
Persistent Failure
```

冻结：

```text
prompt set
GEM checkpoint
seeds
threshold profile
judge version
```

## 核心 Baseline / Ablation

至少：

```text
A. Frozen GEM only
B. GEM + Motion Compiler
C. + Retrieval / Constraint / Keyframe
D. + K Candidates
E. + Tournament
F. + Multi-Verifier
G. + Diagnosis / Targeted Repair
H. Full MotionAgent
```

## 核心指标

### Task Success

```text
Success@1
Final Verified Success
```

### Event

```text
Integrity Accuracy
Temporal Accuracy
Frequency Accuracy
```

### Constraint

```text
Joint Position Error
Trajectory RMSE
Contact Error
Keyframe Satisfaction
```

### Quality

```text
TMR
MotionCritic
Physical Metrics
Human Preference
```

### Agent Efficiency

```text
Generation Rounds
Tool Calls
Retrieval Calls
Repair Rounds
Latency
GPU Cost
MLLM Cost
```

### Reliability

```text
False Accept Rate
False Reject Rate
Regression Rate
Budget Exhaustion Rate
```

## 测试方法

### Test 13.1 — Repeated Seeds

每个 benchmark case 使用固定：

```text
N seeds
```

记录均值和方差。

### Test 13.2 — Human Calibration

抽取固定子集做人类标注：

```text
semantic
naturalness
pairwise preference
pass/fail
```

用于：

```text
Tournament Judge calibration
Verifier threshold calibration
```

### Test 13.3 — Ablation Reproducibility

每个 ablation：

```text
相同 prompt
相同 checkpoint
相同 seed set
```

只能改变目标模块。

### Test 13.4 — Full Replay

任一实验结果都必须可以通过：

```text
run_id
config snapshot
artifact IDs
seed
```

重新定位和复现。

## Stage Gate

```text
[ ] benchmark 固定
[ ] baseline + ablation 可重复运行
[ ] 所有结果带版本 / seed / config
[ ] False Accept 已单独分析
[ ] Agent cost / rounds 已统计
[ ] 能回答“哪个模块带来了什么提升”
```

完成后，V1 才算真正：

```text
research complete
```

而不仅是：

```text
system runs
```

---

# 17. Post-V1 Optional Phase — Guided Generation

这不是当前 V1 Gate。

只有前面全部稳定后再做：

```text
Numerical Reward Guidance
DNO-style Noise Optimization
ReAlign-style Semantic Reward
Planner RL
```

进入该阶段前必须先证明：

```text
normal generation
+
targeted repair
```

已经有稳定 baseline。

否则无法判断：

```text
Guided Generation 到底带来什么提升
```

---

# 18. 推荐实际开发顺序总表

| Phase | 核心结果 | 是否真实 GEM |
|---|---|---:|
| 0 | GEM baseline freeze | 是 |
| 1 | State / Artifact / Schema | 否 |
| 2 | Planner / Orchestrator Mock Loop | 否 |
| 3 | Motion Compiler | 否 |
| 4 | Pure-text GEM minimal loop | 是 |
| 5 | Retrieval | 部分 |
| 6 | Constraint + Keyframe | 部分 |
| 7 | Complete Generation | 是 |
| 8 | Tournament | 使用生成结果 |
| 9 | Multi-Verifier | 使用生成结果 |
| 10 | Diagnosis / Repair | 使用报告 |
| 11 | Full E2E | 是 |
| 12 | Industrial Harness | 是 |
| 13 | Research Evaluation | 是 |

---

# 19. 每个阶段完成后的标准记录模板

每完成一个 Phase，建立：

```markdown
# Phase N Completion Report

## Implementation
- commit:
- config version:
- schema version:

## Tests
- unit:
- contract:
- integration:
- GPU:
- E2E:

## Metrics
- ...

## Failed Cases
- ...

## Known Limitations
- ...

## Gate
PASS / FAIL
```

只有：

```text
Gate = PASS
```

才进入下一阶段。

---

# 20. 最终 V1 完成定义

整个项目只有同时达到以下条件，才算 V1 完成：

```text
[ ] Frozen GEM baseline 可复现
[ ] State / Artifact / Event Log 可恢复
[ ] Planner 只能使用合法 Action
[ ] Motion Compiler 输出稳定 DSL / Timeline / GEMTextCondition
[ ] Retrieval train-only，无 leakage
[ ] Constraint / Keyframe 有统一 numerical condition + verification spec
[ ] GEM 支持 K candidate + hard condition + segment inpainting
[ ] Tournament 可复现，无随机 tie resolution
[ ] Multi-Verifier 使用 dynamic plan + required gating
[ ] deterministic failure fixtures 不被错误 ACCEPT
[ ] Diagnosis 输出 Planner-compatible Proposal
[ ] same failure / same ineffective repair 不会无限循环
[ ] crash/retry 不重复昂贵工作
[ ] 完整 trace / version / fingerprint / seed 可追溯
[ ] E2E benchmark 全部可结束
[ ] Baseline / Ablation 可重复运行
[ ] 能量化 Full MotionAgent 相对 Frozen GEM 的收益与成本
```

---

# 21. 第一批建议立即创建的测试文件

开始实施时优先创建这些测试骨架：

```text
tests/
│
├── unit/
│   ├── test_state_reducer.py
│   ├── test_planner_guards.py
│   ├── test_motion_compiler.py
│   ├── test_retrieval_query.py
│   ├── test_constraint_masks.py
│   ├── test_keyframe_masks.py
│   ├── test_tournament_bracket.py
│   ├── test_verifier_plan.py
│   └── test_repair_policy.py
│
├── contracts/
│   ├── test_compiler_to_generator.py
│   ├── test_constraint_to_generator.py
│   ├── test_constraint_to_verifier.py
│   ├── test_keyframe_to_generator.py
│   ├── test_keyframe_to_verifier.py
│   ├── test_generator_to_tournament.py
│   ├── test_tournament_to_verifier.py
│   ├── test_verifier_to_diagnosis.py
│   └── test_diagnosis_to_planner.py
│
├── gpu/
│   ├── test_gem_pure_text.py
│   ├── test_gem_seed_reproducibility.py
│   ├── test_gem_hard_condition.py
│   └── test_gem_segment_inpainting.py
│
├── e2e/
│   ├── test_simple_motion.py
│   ├── test_frequency_repair.py
│   ├── test_rare_motion_retrieval.py
│   ├── test_constraint_loop.py
│   ├── test_keyframe_loop.py
│   ├── test_segment_repair.py
│   └── test_persistent_failure_stop.py
│
└── failure_injection/
    ├── test_gpu_oom.py
    ├── test_llm_timeout.py
    ├── test_judge_timeout.py
    ├── test_verifier_failure.py
    └── test_state_version_conflict.py
```

这组测试骨架建议在 Phase 1–2 就创建目录和空测试文件，后续每个阶段逐步填充。

---

# 22. 最重要的实施原则

整个项目实施过程中始终保持：

```text
先固定 Contract
→ 再实现模块

先写 deterministic test
→ 再接 LLM / GEM

先证明局部正确
→ 再跑 E2E

先用 Mock 验证控制逻辑
→ 再付 GPU / MLLM 成本

任何 ACCEPT
→ 必须有 complete VerificationReport

任何 Repair
→ 必须能追溯到 Failure

任何昂贵调用
→ 必须可追踪、可复现、尽量可缓存
```

最终目标不是：

```text
“所有模块都写出来了”
```

而是：

> **每个模块都有明确可验证的正确性标准，并且这些标准组合后能够证明整个 MotionAgent 闭环可靠工作。**

## Phase 9B Integration Status (2026-10-02)

Phase 9 Core Engineering Gate: PASS (historical status preserved).
Phase 8: BYPASSED_FOR_K1_MODE; current execution remains K=1.

| Gate | Status |
|---|---|
| Phase 9B TMR | PASS: actual GEM candidate FK/Guo conversion and real official text-motion inference executed |
| Phase 9B MotionCritic | IMPLEMENTED_BLOCKED: real CPU inference executed; exact SMPL hand retargeting missing |
| Phase 9B MLLM | PASS: user-authorized real OpenAI candidate-frame API execution and typed structured findings |
| Phase 9 Full Learned-Verifier Gate | PARTIAL |

The user explicitly authorized continuation and real candidate-frame uploads.
Existing credentials have now produced real structured OpenAI responses. The local
SMPL-X model was recovered from the user's Downloads folder, enabling real TMR
and deterministic joint-based physical checks. Exact full-SMPL MotionCritic
retargeting remains blocked; neutral terminal-hand scores are exploratory only.
Detailed evidence and remaining dependencies: `reports/phase_9b_real_verifier_integration.md`
and `outputs/phase9b_real_verifiers/gate.json`. Phase 10 has not begun.
