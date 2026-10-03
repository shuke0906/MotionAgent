# MotionAgent for GEM：System Integration & Execution Guide

> 本文件是跨模块系统集成规范。  
> 它不重新定义 03–11 的模块内部算法，而是规定这些模块**如何组合成一个可运行的 MotionAgent 服务**。  
> 所有 Action、State、Result、Verification 和 Repair 口径以对应模块文档为准。

# 12. 系统集成总原则

> Phase 7 validation stops after candidate generation/persistence. Its
> condition interfaces are validated independently of motion satisfaction.
> Tournament, Verifier and Repair in the full-system topology below belong
> to later phases. `GENERATE` remains one Planner action, and heavy tensors
> remain outside planner state. See the revised Phase 7 scope in module 08.

Temporal compilation compatibility: source-clause provenance and structured
temporal fields live in the MotionSpecification artifact, not a new Planner
action or enlarged global state. The compiler alone derives timeline boundaries
and compound concurrent captions. GenerationRequest preserves the actual
specification and semantic segment bounds. Validation experiments may compare
legacy/split projections externally, but production has one compiler path and
no prompt-specific generation branch. Frozen GEM and later-phase modules stay
unchanged; fixed-environment fresh inference is tested separately from cache.

系统必须保持：

```text
ONE Global Planner
+
LangGraph StateGraph Runtime
+
Bounded Tools
+
Fixed Post-Generation Workflow
+
Versioned State
+
Code-level Guards
```

不采用：

```text
每个模块都是 autonomous agent
```

---

# 12.1 Source of Truth

跨模块发生冲突时，优先级：

```text
03 Motion Planner
04 Unified State
05 Motion Compiler
06 Retrieval
07 Constraint
07B Keyframe
08 Generation
09 Tournament
10 Multi-Verifier
11 Diagnosis
```

本文件只能引用这些接口，不自行创造新的 Action。

---

# 12.2 唯一 Planner Action Space

```text
COMPILE_MOTION
RETRIEVE_REFERENCE
BUILD_CONSTRAINT
BUILD_KEYFRAME
GENERATE
ACCEPT
STOP_FAILED
```

不存在：

```text
PLAN
VERIFY
REPAIR
GENERATE_GUIDED
REGENERATE
```

其中：

```text
GENERATE(strategy="guided")
```

用于 guided generation。

---

# 12.3 LangGraph Runtime 与 Application Orchestrator 的职责

V1 显式采用 **LangGraph StateGraph** 作为顶层执行 runtime。

LangGraph 负责：

```text
graph topology
conditional routing
fixed edges / fixed subgraphs
checkpoint / resume
node lifecycle
thread-scoped execution state
```

`app/orchestrator.py` 不再自己维护一套重复的 `while + if/elif` 状态机，而负责：

```text
create / load MotionAgentRun
build / compile graph
select checkpointer
invoke / ainvoke graph with thread_id=run_id
API lifecycle / cancellation boundary
final result assembly
```

MotionAgent 自己仍负责：

```text
Planner policy
Action Guard
Request Builder
Domain Tool / Workflow Service
StateReducer / canonical State Store
Artifact Store
Fingerprint / idempotency
GPU worker / queue
```

LangGraph 不做 Motion semantic reasoning、Constraint math、Candidate ranking、Verifier scoring 或 Root-cause reasoning。

# 12.4 LangGraph 顶层拓扑

推荐 topology：

```text
START
  ↓
planner
  ├─ COMPILE_MOTION      → compile_motion ─┐
  ├─ RETRIEVE_REFERENCE  → retrieval ──────┤
  ├─ BUILD_CONSTRAINT    → constraint ─────┤
  ├─ BUILD_KEYFRAME      → keyframe ───────┤→ planner
  ├─ GENERATE            → generation → tournament → verifier → diagnosis ─┘
  ├─ ACCEPT              → accept → END
  └─ STOP_FAILED         → stop_failed → END
```

示意代码：

```python
from langgraph.graph import StateGraph, START, END

builder = StateGraph(MotionGraphState)

builder.add_node("planner", planner_node)
builder.add_node("compile_motion", compile_motion_node)
builder.add_node("retrieval", retrieval_node)
builder.add_node("constraint", constraint_node)
builder.add_node("keyframe", keyframe_node)
builder.add_node("generation", generation_node)
builder.add_node("tournament", tournament_node)
builder.add_node("verifier", verifier_node)
builder.add_node("diagnosis", diagnosis_node)
builder.add_node("accept", accept_node)
builder.add_node("stop_failed", stop_failed_node)

builder.add_edge(START, "planner")
builder.add_conditional_edges("planner", route_planner_action, ACTION_TO_NODE)

for node in ["compile_motion", "retrieval", "constraint", "keyframe"]:
    builder.add_edge(node, "planner")

builder.add_edge("generation", "tournament")
builder.add_edge("tournament", "verifier")
builder.add_edge("verifier", "diagnosis")
builder.add_edge("diagnosis", "planner")
builder.add_edge("accept", END)
builder.add_edge("stop_failed", END)

graph = builder.compile(checkpointer=checkpointer)
```

Graph config 统一使用：

```python
{"configurable": {"thread_id": run_id}}
```

`route_planner_action()` 在 Hard Guard 通过后只做固定 Enum → Node 映射，禁止 LLM 返回任意 node 名称。

# 12.5 Action Executor Registry（Domain Service Registry）

```python
ACTION_EXECUTORS = {
    "COMPILE_MOTION": execute_compile_motion,
    "RETRIEVE_REFERENCE": execute_retrieval,
    "BUILD_CONSTRAINT": execute_constraint,
    "BUILD_KEYFRAME": execute_keyframe,
    "GENERATE": execute_generation,
}
```

`ACCEPT / STOP_FAILED` 由对应 terminal graph node 处理，不作为外部 Tool。

LangGraph node 可以复用该 Registry，避免把 domain execution 逻辑写进 graph topology。

---

# 12.6 Request Builder Registry

Planner `payload` 不是底层 Request。

统一：

```python
REQUEST_BUILDERS = {
    "COMPILE_MOTION": build_compiler_request,
    "RETRIEVE_REFERENCE": build_retrieval_request,
    "BUILD_CONSTRAINT": build_constraint_request,
    "BUILD_KEYFRAME": build_keyframe_request,
    "GENERATE": build_generation_request,
}
```

Request Builder 从：

```text
PlannerDecision
+
Current State
```

构造强类型 Request。

---

# 12.7 Action Guard

执行前：

```python
validate_planner_decision(...)
```

至少：

```text
plan missing → GENERATE illegal

verification missing → ACCEPT illegal

overall_pass false → ACCEPT illegal

generation budget 0 → GENERATE illegal

same retrieval signature → RETRIEVE_REFERENCE illegal

blocked repair family → matching proposal illegal

guided unavailable → GENERATE guided illegal
```

---

# 12.8 State Commit Pattern

固定：

```text
Tool Result
↓
Persist Artifact
↓
State Reducer
↓
State Invariant Validation
↓
Atomic Commit
↓
Event Log
```

Tool 不直接修改共享 State。

---

# 12.9 LangGraph Automatic Post-Generation Subgraph

当：

```text
GENERATE
```

成功返回至少一个 Candidate 后，由固定 graph edge 执行：

```text
GenerationResult
↓
TournamentRequestBuilder
↓
Candidate Tournament
↓
TournamentResult / Champion
↓
VerificationRequestBuilder
↓
Multi-Verifier
↓
VerificationReport
↓
DiagnosisRequestBuilder
↓
Diagnosis & Repair Planning
↓
DiagnosisResult
↓
Planner
```

---

# 12.10 Post-Generation Node Contract

`generation / tournament / verifier / diagnosis` 四个 node 不共享大型对象，只通过 IDs / handles 和 canonical state 连接。

```python
async def tournament_node(gs: MotionGraphState):
    state = state_store.load_latest(gs.run_id)
    request = build_tournament_request(state)
    result = await tournament_service.run(request)
    state = persist_and_commit_tournament(state, result)
    return {"state_version": state.state_version}
```

Verifier / Diagnosis 使用同一模式。

固定 edge 保证：

```text
generation succeeded
→ tournament
→ verifier
→ diagnosis
→ planner
```

如果 `GenerationResult.status=failed`，generation node 根据 typed failure policy 写入 canonical state 后路由到 Planner/terminal handling；不能伪造 Candidate 进入 Tournament。

Checkpoint/resume 时，node 必须先根据 fingerprint / committed artifact 判断昂贵操作是否已经成功，避免重复 GEM / render / MLLM 调用。

# 12.11 Generation Failed / Partial

## failed

如果：

```text
GenerationResult.status = failed
```

且无成功 Candidate：

```text
不进入 Tournament
```

写 Tool Failure Observation 回 State，由 Planner 下一轮处理。

## partial

如果：

```text
至少 1 Candidate success
```

进入 Tournament，只使用成功 Candidate。

---

# 12.12 Tool / Workflow 边界

## Tools

```text
Compiler
Retrieval
Constraint
Keyframe
GEM Generation
```

## Fixed Workflows

```text
Tournament
Multi-Verifier
Diagnosis
```

## Agent

```text
Motion Planner
```

---

# 12.13 Shared Infrastructure

以下不属于任何业务模块：

```text
State Store
Artifact Store
Event Store
Fingerprint Utility
Config Loader
Cache
Worker Queue
Tracing
Metrics
```

统一放 Harness 层。

---

# 12.13.1 Cross-Segment Continuity Contract

跨动作连续性使用现有模块传递，不新增 Graph Node：

```text
Motion Compiler
→ HeadingContinuitySpec
→ MotionSpecification artifact
→ Generation / Candidate
→ TournamentRequestBuilder
→ Multi-Verifier
→ Diagnosis
→ Planner
```

如果 persistent heading drift 需要数值控制：

```text
Planner
→ BUILD_CONSTRAINT
→ root_trajectory orientation-only soft control
→ Condition Assembly
→ Generation
```

Source of Truth：

```text
用户显式 orientation
> trajectory explicit policy
> continuity inheritance
> free
```

后级模块不得自行覆盖这个优先级。

Generation Tool 不负责猜测 continuity；它只消费 Compiler caption / active condition。Tournament 和 Verifier 只评估已经存在于 MotionSpecification / VerificationSpec 中的 expectation。

# 12.13.2 Temporal Semantics Contract

重复、连续和交替动作同样使用现有模块传递，不新增 Graph Node：

```text
Motion Compiler Semantic Parser
→ Temporal Resolver
→ MotionSpecification.temporal_constraint / temporal_mode / temporal_relation
→ GenerationRequest.motion_spec
→ Tournament / Multi-Verifier / Diagnosis
→ Planner
```

Planner 不解析 `twice`、`three times`、`repeatedly` 或 `continuously`。Generation
也不保证实际动作满足这些频次；它只接收结构化 specification。是否满足精确频次由后续 verifier / guided generation / repair 工作处理。

# 12.14 Condition Assembly

进入 Generation 前：

```text
GEMTextCondition
+
Constraint HardMotionCondition
+
Keyframe HardMotionCondition
+
RewardSpec[]
↓
GenerationConditionBundle
```

Hard Condition 统一合并：

```text
observed_motion_3d
motion_mask_3d
```

发生冲突：

```text
Generation Preflight = blocked
```

返回 Planner。

---

# 12.15 Verification Spec 的 Single Source of Truth

```text
Constraint
→ Constraint VerificationSpec

Keyframe
→ KeyframeVerificationSpec

MotionSpecification
→ Semantic / Event Requirement
```

Verifier 不重新定义这些 Requirement。

---

# 12.16 Repair 的执行方式

第 11 单元返回：

```text
RepairProposal[]
```

它们只是 Planner Context。

下一轮：

```text
Planner
→ ONE PlannerDecision
```

再由原 Action Tool 真正执行修复。

没有独立 `REPAIR` Executor。

---

# 12.17 Repair History 更新

Planner 采用某 Proposal：

```text
RepairHistoryEntry = pending
```

下一轮 Verification 完成后：

```text
resolved
improved
unchanged
worse
```

然后更新 No-Improvement Guard。

---

# 12.18 Canonical Domain Schema

推荐：

```text
domain/
├── state.py
├── planner.py
├── motion.py
├── retrieval.py
├── constraint.py
├── keyframe.py
├── generation.py
├── tournament.py
├── verification.py
└── diagnosis.py
```

其他模块 import，不复制。

---

# 12.19 Store 划分

## Metadata / State

```text
Postgres / SQLite
```

## Heavy Artifacts

```text
filesystem / object storage
```

## Ephemeral Queue / Cache

```text
in-process
或 Redis
```

V1 不要求全部上分布式组件。

---

# 12.20 Worker 划分

```text
CPU / LLM Orchestrator

GEM GPU Worker

Render Worker
```

GEM Worker 长驻模型。

---

# 12.21 Config 文件

推荐：

```text
configs/
├── planner.yaml
├── retrieval.yaml
├── constraints.yaml
├── keyframes.yaml
├── generation.yaml
├── tournament.yaml
├── verifier_thresholds.yaml
├── repair_policy.yaml
└── runtime.yaml
```

`runtime.yaml`：

```yaml
budgets:
  max_iterations: 8
  max_generations: 4
  max_retrievals: 4
  max_guided_generations: 1

workers:
  generation_queue_limit: 8
  render_queue_limit: 16

context:
  recent_actions: 6
```

实际数值通过实验调整。

---

# 12.22 Trace / Event

每次 Run：

```text
trace_id
```

每个 Planner / Tool / Workflow：

```text
span_id
```

Event 至少记录：

```text
input artifact ids
output artifact ids
state version
status
latency
model / prompt / config version
```

---

# 12.23 系统级 Error Policy

```text
Transient Infrastructure Error
→ module-level retry

Invalid Request
→ no retry, return structured failure

Motion Quality Failure
→ Verify / Diagnose / Planner

Unrecoverable Tool Failure
→ Planner STOP_FAILED candidate
```

---

# 12.24 推荐目录

```text
motion_agent/
├── app/
├── graph/
│   ├── builder.py
│   ├── state.py
│   ├── routing.py
│   ├── checkpoints.py
│   └── nodes/
│       ├── planner.py
│       ├── tools.py
│       └── post_generation.py
├── domain/
├── agent/
├── state/
├── compiler/
├── retrieval/
├── constraints/
├── keyframes/
├── generation/
├── tournament/
├── verification/
├── repair/
├── rendering/
├── stores/
├── workers/
├── observability/
├── evaluation/
├── common/
└── configs/
```

详细见 `00_project_overview.md` 与 `industrial_harness_design.md`。

---

# 12.25 最小闭环实现顺序

```text
1. Canonical Schemas
2. Unified State / Store / Reducer
3. LangGraph GraphState / Checkpointer / Topology Skeleton
4. Motion Planner + Guarded Conditional Routing
5. Motion Compiler
6. Pure-text GEM Generator
7. Candidate Store
8. Tournament basic
9. Verifier basic
10. Diagnosis basic
11. Retrieval
12. Constraint
13. Keyframe
14. Full regression / repair loop
```

---

# 12.26 最小闭环测试

LangGraph runtime 必须额外覆盖：

```text
Graph topology matches documented edges
Every Planner Action maps to exactly one legal route
Illegal Action cannot enter a node
thread_id == run_id isolation
checkpoint resume returns to correct next node
resume after committed generation does not rerun GEM
terminal ACCEPT / STOP_FAILED reaches END
```


```text
User:
walk forward then sit down

Expected:
COMPILE_MOTION
→ GENERATE
→ Tournament
→ Verify
→ Diagnosis(no failure)
→ Planner
→ ACCEPT
```

第二个测试：

```text
wave right hand 3 times then sit down
```

必须出现：

```text
frequency verification
```

第三个：

```text
finish in a stable seated pose
```

必须出现：

```text
BUILD_KEYFRAME
```

---

# 12.27 系统级 Contract Test

必须写一条完整自动测试：

```python
async def test_end_to_end_contracts():
    # compile
    # state commit
    # generate mock candidates
    # tournament
    # verify
    # diagnose
    # planner context rebuild
    ...
```

可以 Mock GEM，不需要每次 CI 真跑 GPU。

---

# 12.28 V1 Definition of Done

整个系统 V1 做好后必须满足：

```text
LangGraph topology / routing / checkpoint tests pass。

所有 Action 都有 Request Builder。

所有 Tool Result 都通过 State Reducer Commit。

没有 Tool 可以绕过 State Store 直接控制另一个 Tool。

GENERATE 后 automatic subgraph 固定执行。

ACCEPT 只能发生在 complete + overall_pass。

所有大型数据都以 Artifact Handle 传递。

所有昂贵调用可追踪、可缓存、可复现。

Crash 后可从最近 commit 继续。

旧版 PLAN / VERIFY / REPAIR / GENERATE_GUIDED 已完全移除。
```
