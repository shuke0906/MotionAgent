# MotionAgent for GEM：模块实现指南

> 本文件对应第 4 单元 Unified State / Harness State。  
> 本模块是 03–11 所有模块之间的**状态契约**。它不负责推理，也不负责生成 Motion；它负责保存“当前真实状态”、引用重型 Artifact、控制版本、预算、执行历史，并为 Planner 构造最小高信号上下文。

# 4. Unified State / Harness State

V1 的 orchestration runtime 使用 LangGraph；本文件定义的 MotionAgentState 仍是 canonical domain state，不由 LangGraph 内部消息历史替代。


本节定义 MotionAgent 的**统一运行状态、Artifact 引用方式、状态更新规则和 Planner Context 构造规范**。

核心原则：

> **Persistent Application State ≠ LLM Context。**

系统必须同时维护：

```text
Current State
+
Immutable / Versioned Artifacts
+
Append-only Event Log
```

而不是把所有模块输出不断 append 到一个大 JSON，再每轮全部发送给 Planner。

---

## 4.1 在整体系统中的定位

```text
User Request
     ↓
MotionAgentRun (run_id = LangGraph thread_id)
     ↓
LangGraph StateGraph / Checkpointer
     │
     ├── Planner Node → Motion Planner
     ├── Tool Nodes → 05/06/07/07b/08
     └── Fixed Post-Generation Nodes → 09/10/11
     │
     ▼
Unified State Store / StateReducer
     │
     ├── Artifact Store
     └── Event Log
     ↓
Versioned Canonical State Update
```

State 是所有跨模块决策的**单一事实源（single source of truth）**。

任何模块都不应通过：

```text
读取另一个模块的临时 Python 全局变量
读取旧聊天记录
解析自然语言日志
```

来推断当前系统状态。

---

# 4.2 三层数据模型

## 4.2.1 Current State

Current State 只保存：

```text
当前有效 Artifact ID
轻量 Summary
控制状态
预算
版本
```

例如：

```text
current_motion_spec_id = spec_003
active_constraint_ids = [constraint_01]
active_keyframe_ids = [keyframe_02]
latest_generation_id = gen_004
champion_candidate_id = cand_004_2
latest_verification_id = verify_005
latest_diagnosis_id = diag_005
```

不保存大型 Tensor。

---

## 4.2.2 Artifact Store

Artifact Store 保存不可直接放进 Planner Context 的对象：

```text
MotionSpecification
GEMTextCondition
RetrievedReference payload
CompiledConstraint
HardMotionCondition Tensor
KeyframeSpec
GenerationRequest
MotionCandidate Tensor / SMPL
TournamentResult
VerificationReport
DiagnosisResult
Render Bundle
```

State 中只保存：

```text
artifact_id / handle
```

---

## 4.2.3 Event Log

Event Log 记录系统发生过什么：

```text
Planner Decision
Tool Start
Tool Success / Failure
State Commit
Generation Start / End
Tournament End
Verification End
Diagnosis End
Repair Outcome Update
```

Event Log 用于：

```text
trace
replay
debug
evaluation
RL trajectory collection
```

但不直接作为 Planner Prompt。

---

# 4.3 顶层 `MotionAgentState`

推荐正式使用 Pydantic：

```python
class MotionAgentState(BaseModel):

    # identity / lifecycle
    run: RunState

    # original task
    task: TaskState

    # shared GEM coordinate / model context
    gem_adapter_context: GEMAdapterContext

    # current semantic plan
    plan: PlanState

    # active control conditions
    conditions: ConditionState

    # generation / selection
    generation: GenerationState

    # verification / diagnosis
    evaluation: EvaluationState

    # repair / history
    repair: RepairState

    # execution control
    control: ControlState

    # concise execution history
    history: HistoryState

    # optimistic concurrency / schema
    state_version: int
    schema_version: str
```

---

# 4.4 `RunState`

```python
class RunState(BaseModel):
    run_id: str

    status: Literal[
        "created",
        "running",
        "accepted",
        "failed",
        "cancelled",
    ]

    created_at: str
    updated_at: str

    trace_id: str

    model_profile: str
    config_profile: str
```

`run_id` 是整个 MotionAgent 工作流的主键。

---

# 4.5 `TaskState`

```python
class TaskState(BaseModel):
    original_request: str

    target_duration_s: float
    fps: int
    total_frames: int

    user_constraints_summary: list[str]

    scene_context_id: str | None

    task_version: int
```

其中：

```text
original_request
```

永远保留原始用户输入，不被 Compiler rewrite 覆盖。

---


# 4.5A `GEMAdapterContext`

Constraint、Keyframe、Generation 必须共享同一坐标 / 模型上下文：

```python
class GEMAdapterContext(BaseModel):
    fps: int
    total_frames: int

    width: int
    height: int

    static_camera: bool

    coordinate_system: str

    model_version: str
    checkpoint_version: str

    camera_handle: str | None = None
```

该对象属于当前 Run 的轻量 Context，可以直接放 State。

禁止 Constraint / Keyframe / Generation 各自构造不同坐标定义。

---

# 4.6 `PlanState`

```python
class PlanState(BaseModel):
    motion_spec_id: str | None
    gem_text_condition_id: str | None

    status: Literal[
        "missing",
        "draft",
        "ready",
        "needs_revision",
    ]

    segment_summaries: list[MotionSegmentSummary]

    routing_hints: list[RoutingHintSummary]

    plan_version: int
```

`MotionSegmentSummary` 只保留 Planner 真正需要的信息：

```python
class MotionSegmentSummary(BaseModel):
    segment_id: int
    start_s: float
    end_s: float

    action: str
    body_parts: list[str]
    style: list[str]
    repetition: int | None

    status: str
```

不要在 State 中重复保存完整 `MotionSegment` 和完整 `MotionSpecification`。

---

# 4.7 `ConditionState`

```python
class ConditionState(BaseModel):
    text: TextConditionSummary

    retrievals: list[RetrievalSummary]

    constraints: list[ConstraintSummary]

    keyframes: list[KeyframeSummary]

    condition_fingerprint: str | None
```

---

## 4.7.1 Text Summary

```python
class TextConditionSummary(BaseModel):
    ready: bool
    gem_text_condition_id: str | None
    segment_count: int
    version: int
```

---

## 4.7.2 Retrieval Summary

```python
class RetrievalSummary(BaseModel):
    ref_id: str
    segment_id: int

    retrieval_type: Literal[
        "caption",
        "motion",
        "pose",
        "trajectory",
    ]

    purpose: Literal[
        "prompt_grounding",
        "motion_prior",
        "keyframe_source",
        "constraint_source",
    ]

    score: float | None

    status: Literal[
        "active",
        "superseded",
        "low_confidence",
        "invalid",
    ]
```

---

## 4.7.3 Constraint Summary

```python
class ConstraintSummary(BaseModel):
    constraint_id: str
    segment_id: int
    constraint_type: str

    requested_strength: str
    effective_mode: str

    status: Literal[
        "active",
        "superseded",
        "removed",
        "conflict",
    ]

    verification_spec_id: str | None
```

---

## 4.7.4 Keyframe Summary

```python
class KeyframeSummary(BaseModel):
    keyframe_id: str
    segment_id: int

    target_time_s: float
    source_type: str

    control_root_translation: bool

    status: Literal[
        "active",
        "superseded",
        "removed",
        "conflict",
    ]

    verification_spec_id: str | None
```

---

# 4.8 `GenerationState`

```python
class GenerationState(BaseModel):
    latest_generation_id: str | None

    strategy: Literal[
        "normal",
        "guided",
    ] | None

    scope: Literal[
        "full",
        "segment",
    ] | None

    target_segments: list[int] | None

    candidate_ids: list[str]

    champion_candidate_id: str | None

    latest_tournament_id: str | None

    previous_champion_candidate_id: str | None

    generation_round: int
```

大型 Candidate 内容通过：

```text
candidate_id
```

从 Candidate Store 获取。

---

# 4.9 `EvaluationState`

```python
class EvaluationState(BaseModel):
    latest_verification_id: str | None
    latest_diagnosis_id: str | None

    verification_summary: VerificationSummary | None
    diagnosis_summary: DiagnosisSummary | None
```

---

## 4.9.1 `VerificationSummary`

必须和第 10 单元保持一致：

```python
class VerificationSummary(BaseModel):
    status: Literal[
        "complete",
        "incomplete",
        "invalid",
        "failed_service",
    ]

    overall_pass: bool

    passed_check_ids: list[str]
    failed_check_ids: list[str]
    warning_check_ids: list[str]
    uncertain_check_ids: list[str]

    critical_failures: list[str]
    affected_segments: list[int]
    diagnostic_codes: list[str]
```

Planner 不读取全部 `VerifierFinding[]`。

---

## 4.9.2 `DiagnosisSummary`

```python
class DiagnosisSummary(BaseModel):
    status: str

    primary_failures: list[str]
    root_causes: list[str]

    target_segments: list[int]

    proposal_summaries: list[RepairProposalSummary]

    terminal_hint: str | None
```

```python
class RepairProposalSummary(BaseModel):
    proposal_id: str
    action: str
    reason_code: str
    target_segments: list[int] | None
    confidence: float
```

完整 `RepairProposal.payload` 仍在 Diagnosis Store。

---

# 4.10 `RepairState`

```python
class RepairState(BaseModel):
    active_proposal_id: str | None

    recent_repairs: list[RepairHistorySummary]

    blocked_repair_families: list[BlockedRepairFamily]
```

```python
class RepairHistorySummary(BaseModel):
    failure_signature: str
    repair_family: str
    planner_action: str

    target_segments: list[int]

    outcome: Literal[
        "pending",
        "resolved",
        "improved",
        "unchanged",
        "worse",
        "failed_execution",
    ]
```

---

# 4.11 `ControlState`

```python
class ControlState(BaseModel):
    iteration: int

    budgets: BudgetState

    capabilities: CapabilitySummary

    accepted: bool

    terminal_reason: str | None
```

---

## 4.11.1 `BudgetState`

必须至少支持：

```python
class BudgetState(BaseModel):
    iterations_left: int

    generations_left: int
    retrieval_calls_left: int
    guided_generations_left: int

    llm_judge_calls_left: int | None

    max_total_candidates_left: int | None
```

Budget 是代码级 Hard Guard，不是 Planner 建议。

---

## 4.11.2 `CapabilitySummary`

```python
class CapabilitySummary(BaseModel):
    retrieval_available: bool
    keyframe_available: bool
    hard_constraint_available: bool
    guided_generation_available: bool

    semantic_mllm_available: bool
    motioncritic_available: bool

    scene_geometry_available: bool
```

Planner 和 Diagnosis 只能提出当前 capability 支持的 Action / Proposal。

---

# 4.12 `HistoryState`

不要保存完整 Prompt Transcript。

```python
class HistoryState(BaseModel):
    recent_actions: list[ActionHistorySummary]

    repeated_action_counts: dict[str, int]

    last_failure_signatures: list[str]
```

```python
class ActionHistorySummary(BaseModel):
    step: int
    action: str

    target_segments: list[int] | None

    result_status: str

    artifact_ids: list[str]
```

建议只保留：

```text
最近 5~10 个 Action Summary
```

旧历史留在 Event Log。

---

# 4.13 Artifact ID 规范

建议统一 ID Prefix：

```text
run_*

spec_*
textcond_*

ref_*
constraint_*
keyframe_*

hardcond_*
reward_*
verspec_*

gen_*
cand_*
tour_*
verify_*
diag_*
repair_*
render_*
```

例如：

```text
spec_0003
constraint_0002
keyframe_0001
gen_0004
cand_0004_02
verify_0005
```

不要使用业务代码自己拼接不一致 ID。

统一：

```python
new_artifact_id(
    artifact_type,
    run_id,
)
```

---

# 4.14 Artifact 必须 Versioned

例如 Motion Plan 不应原地覆盖：

```text
spec_001
→ 修改
→ spec_002
```

Generation 记录：

```text
motion_spec_id = spec_002
condition_fingerprint = ...
```

这样可以准确回答：

```text
candidate_4
到底由哪一版 plan / constraint / keyframe 生成？
```

---

# 4.15 Canonical Schema 放在哪里

随着模块增加，最危险的问题之一是：

```text
03 定义一份 Segment
05 又定义一份 Segment
08 再定义一个 Segment
```

最后字段逐渐漂移。

因此建议增加：

```text
motion_agent/domain/
```

目录：

```text
domain/
├── planner.py
├── motion.py
├── retrieval.py
├── constraint.py
├── keyframe.py
├── generation.py
├── tournament.py
├── verification.py
├── diagnosis.py
└── state.py
```

模块目录中的 `schemas.py` 应：

```text
import / extend canonical schema
```

而不是重新复制。

---

# 4.16 State Mutation 规则

任何 Tool 不应：

```python
state.conditions.constraints.append(...)
```

然后继续执行。

推荐：

```text
Tool
→ Result
→ State Reducer
→ Validate
→ Commit
```

例如：

```python
result = compile_constraint(...)

new_state = state_reducer.apply_constraint_result(
    state,
    result,
)

state_store.commit(
    expected_version=state.state_version,
    new_state=new_state,
)
```

---

# 4.17 为什么需要 State Reducer

State Reducer 统一处理：

```text
active / superseded
version increment
condition fingerprint
history update
budget decrement
artifact references
```

避免每个 Tool 自己更新 State，导致逻辑不一致。

建议：

```text
state/reducer.py
```

---

# 4.18 Optimistic Concurrency

如果未来：

```text
render
metric
verification
```

存在并行 worker，可能多个结果同时返回。

State Store 应支持：

```text
expected_state_version
```

如果：

```text
current_version != expected_version
```

拒绝旧结果直接覆盖新 State。

例如：

```python
class StateConflictError(Exception):
    ...
```

---

# 4.19 State Commit

每次成功 Tool / Workflow 后：

```text
1. Persist Artifact
2. Build State Update
3. Validate State Invariants
4. Atomic Commit State
5. Append Event Log
```

不要：

```text
先改 State
再保存 Artifact
```

否则 Artifact 保存失败时 State 会指向不存在对象。

---

# 4.20 State Invariants

必须实现：

```python
validate_state_invariants(state)
```

至少检查：

```text
champion 必须属于某个成功 generation

latest verification 的 candidate
必须等于当前 champion

accepted == true
→ latest verification overall_pass == true

active constraint ID 不得重复

active keyframe ID 不得重复

current spec 必须与 current text condition version compatible

condition fingerprint 必须覆盖所有 active control conditions

budget 不得为负
```

---

# 4.21 Planner Context Builder

Planner 绝对不直接读取：

```text
MotionAgentState.model_dump()
```

而是：

```python
context = build_planner_context(state)
```

输出必须和第 3 单元：

```python
class PlannerContext:
    task: TaskSummary
    plan: MotionPlanSummary | None
    conditions: ConditionSummary
    latest_generation: GenerationSummary | None
    latest_verification: VerificationSummary | None
    diagnosis: DiagnosisSummary | None
    history: HistorySummary
    budget: BudgetState
    capabilities: CapabilitySummary
```

完全一致。

---

# 4.22 Planner Context Token Budget

Planner Context 应设置硬限制。

建议先按字段控制：

```text
Original request
→ 原文保留

Plan
→ 每个 segment 只保留关键字段

Conditions
→ summary only

Verification
→ failures + pass summary

Diagnosis
→ top 1~3 proposals

History
→ recent N actions
```

如果仍然过长，再使用：

```text
structured compaction
```

而不是把所有字段全文截断。

---

# 4.23 Context Compaction

建议保留：

```text
current truth
unresolved failures
architectural decisions
active controls
recent repair effects
```

删除：

```text
旧 tool raw output
已 superseded artifact details
旧 candidate tensor summary
重复成功日志
```

---

# 4.24 Tool Request Builder

每个 Tool Request 都应从：

```text
PlannerDecision
+
Current State
```

构造。

Planner 不负责填写 Tool 全部底层参数。

例如：

```text
Planner:
BUILD_CONSTRAINT
segment 1
contact
```

由：

```text
ConstraintRequestBuilder
```

补充：

```text
fps
resolved time context
existing constraint IDs
scene context
references
```

---

# 4.25 Automatic Post-Generation State Update

第 3 单元规定：

```text
GENERATE
→ Tournament
→ Verify
→ Diagnosis
→ Planner
```

State 的更新顺序必须是：

```text
GenerationResult commit
↓
TournamentResult commit
↓
champion update
↓
VerificationReport commit
↓
DiagnosisResult commit
↓
PlannerContext build
```

不能把三个 Workflow 临时结果只存在内存里，最后一次性更新。

这样中间故障可以恢复。

---

# 4.26 Repair History 更新时机

Planner 选择某个 RepairProposal 后：

```text
RepairHistoryEntry
status = pending
```

执行 Tool 后还不能立即写：

```text
resolved
```

必须等待下一次：

```text
VerificationReport
```

再由第 11 单元 `RepairEffectTracker` 更新：

```text
resolved / improved / unchanged / worse
```

---

# 4.27 Fingerprint 统一管理

统一放：

```text
common/fingerprints.py
```

至少定义：

```text
condition_fingerprint
retrieval_signature
candidate_fingerprint
criteria_fingerprint
evidence_fingerprint
failure_signature
repair_signature
```

不要各模块自己使用不同 hash canonicalization。

---

# 4.28 State Store 实现

V1 推荐：

```text
PostgreSQL
或开发阶段 SQLite
```

存：

```text
MotionAgentState
Artifact metadata
Event Log
Repair History
```

重型：

```text
.pt
.npy
render video
```

放：

```text
filesystem / object store
```

---

# 4.29 推荐表结构

最低：

```text
runs
run_states
artifacts
events
repair_history
```

例如：

```text
runs
- run_id
- status
- created_at

run_states
- run_id
- version
- state_json
- created_at

artifacts
- artifact_id
- run_id
- type
- version
- uri
- fingerprint
- metadata_json

events
- event_id
- run_id
- state_version
- event_type
- payload_json
- timestamp
```

---

# 4.30 Artifact Store 接口

```python
class ArtifactStore:

    def put(
        self,
        artifact_type: str,
        payload,
        metadata: dict,
    ) -> ArtifactHandle:
        ...

    def get(
        self,
        artifact_id: str,
    ):
        ...

    def exists(
        self,
        artifact_id: str,
    ) -> bool:
        ...
```

---

# 4.31 Event Log 接口

```python
class EventStore:

    def append(
        self,
        event: AgentEvent,
    ) -> None:
        ...
```

```python
class AgentEvent(BaseModel):
    event_id: str
    run_id: str

    event_type: str

    state_version_before: int
    state_version_after: int | None

    input_artifact_ids: list[str]
    output_artifact_ids: list[str]

    status: str

    trace_id: str
    span_id: str

    timestamp: str
```

---

# 4.32 Checkpoint / Resume

V1 正式采用 **LangGraph Checkpointer + MotionAgent canonical state persistence** 的双层恢复方式。

```text
LangGraph Checkpointer
→ 保存 thread-scoped graph execution state
→ 当前/下一 node、轻量 routing state、interrupt/resume metadata

MotionAgent State Store
→ 保存 committed canonical MotionAgentState

Artifact Store
→ 保存 GEM motion / SMPL / render / embeddings 等重型对象

Event Log
→ 保存可审计的业务 transition
```

推荐直接令：

```text
LangGraph thread_id = MotionAgent run_id
```

开发环境可以使用 SQLite checkpointer；需要跨进程持久化时使用 persistent checkpointer（例如 Postgres）。禁止把 `InMemorySaver` 当作重启恢复方案。

Resume 流程：

```text
process restart
→ load LangGraph checkpoint by run_id
→ load latest committed MotionAgentState
→ validate state_version / artifact references
→ reconcile lightweight graph state
→ continue from the next legal graph node
```

重要：LangGraph checkpoint **不能替代幂等性**。对于 GEM generation、render、MLLM judge 等昂贵 node，仍然必须使用 fingerprint / artifact existence / committed result 检查，保证 crash-resume 后不会无意义地重复昂贵调用。

Resume 不依赖 LLM conversation memory，也不把完整 tensor 写入 GraphState。

# 4.33 State 与 Conversation 的边界

如果产品层允许用户多轮聊天：

```text
Conversation
```

只负责：

```text
用户交互
澄清
展示结果
```

MotionAgent 真正的 workflow state 仍由：

```text
MotionAgentRun
```

独立管理。

禁止：

```text
“之前聊天里应该说过”
```

作为执行正确性的依赖。

---

# 4.34 Current State Snapshot 示例

```json
{
  "run": {
    "run_id": "run_001",
    "status": "running",
    "trace_id": "trace_001"
  },
  "task": {
    "original_request": "A person limps forward, supports the body with the right hand on a table, then sits down.",
    "target_duration_s": 6.0,
    "fps": 30,
    "total_frames": 180
  },
  "gem_adapter_context": {
    "fps": 30,
    "total_frames": 180,
    "width": 1280,
    "height": 720,
    "static_camera": true,
    "coordinate_system": "gem_canonical_v1",
    "model_version": "gem_smpl",
    "checkpoint_version": "checkpoint_hash",
    "camera_handle": null
  },
  "plan": {
    "motion_spec_id": "spec_003",
    "gem_text_condition_id": "textcond_003",
    "status": "ready",
    "plan_version": 3
  },
  "conditions": {
    "retrievals": [
      {
        "ref_id": "ref_002",
        "segment_id": 0,
        "retrieval_type": "motion",
        "purpose": "motion_prior",
        "status": "active"
      }
    ],
    "constraints": [
      {
        "constraint_id": "constraint_001",
        "segment_id": 1,
        "constraint_type": "contact",
        "effective_mode": "selection_reward",
        "status": "active"
      }
    ],
    "keyframes": [
      {
        "keyframe_id": "keyframe_001",
        "segment_id": 2,
        "target_time_s": 5.8,
        "source_type": "retrieval",
        "status": "active"
      }
    ],
    "condition_fingerprint": "..."
  },
  "generation": {
    "latest_generation_id": "gen_004",
    "candidate_ids": [
      "cand_004_0",
      "cand_004_1",
      "cand_004_2",
      "cand_004_3"
    ],
    "champion_candidate_id": "cand_004_2",
    "latest_tournament_id": "tour_004",
    "generation_round": 2
  },
  "evaluation": {
    "latest_verification_id": "verify_004",
    "latest_diagnosis_id": "diag_004",
    "verification_summary": {
      "status": "complete",
      "overall_pass": false,
      "failed_check_ids": ["contact_constraint_001"],
      "critical_failures": ["CONSTRAINT_CONTACT_ERROR"],
      "affected_segments": [1],
      "diagnostic_codes": ["CONSTRAINT_CONTACT_ERROR"]
    },
    "diagnosis_summary": {
      "status": "diagnosed",
      "primary_failures": ["CONSTRAINT_CONTACT_ERROR"],
      "root_causes": ["CONSTRAINT_TOO_WEAK"],
      "target_segments": [1],
      "proposal_summaries": [
        {
          "proposal_id": "repair_007",
          "action": "BUILD_CONSTRAINT",
          "reason_code": "CONSTRAINT_FAILURE",
          "target_segments": [1],
          "confidence": 0.91
        }
      ]
    }
  },
  "control": {
    "iteration": 5,
    "budgets": {
      "iterations_left": 5,
      "generations_left": 2,
      "retrieval_calls_left": 2,
      "guided_generations_left": 1
    },
    "accepted": false
  },
  "state_version": 18,
  "schema_version": "v1"
}
```

---

# 4.35 内部代码结构

```text
state/
│
├── schemas.py
├── store.py
├── reducer.py
├── invariants.py
├── context_builder.py
├── summaries.py
├── versioning.py
├── event_store.py
└── migrations.py
```

通用：

```text
common/
├── ids.py
├── fingerprints.py
├── errors.py
└── time.py
```

LangGraph-specific state adapter 放在：

```text
graph/
├── state.py          # MotionGraphState: lightweight execution envelope
└── checkpoints.py    # checkpointer creation / persistence config
```

禁止在 `graph/state.py` 再复制一套与 `state/schemas.py` 漂移的完整业务 Schema。

---

# 4.36 V1 实现顺序

## Step 1

实现 canonical Pydantic State Schema。

## Step 2

实现 Artifact Store + Artifact IDs。

## Step 3

实现 State Reducer。

## Step 4

实现 State Invariants。

## Step 5

实现 PlannerContextBuilder。

## Step 6

实现 Event Log。

## Step 7

实现 State Version / optimistic concurrency。

## Step 8

实现 LangGraph `MotionGraphState` adapter 与 `thread_id = run_id`。

## Step 9

接入 persistent LangGraph Checkpointer，并实现 canonical-state reconciliation。

## Step 10

实现 crash/resume + expensive-node idempotency 测试。

---

# 4.37 单元测试要求

至少覆盖：

```text
Create Run

Plan Commit

Constraint Add / Replace / Remove

Keyframe Add / Replace / Remove

Generation Commit

Champion Update

Verification Commit

Diagnosis Commit

Accepted Guard

Artifact Missing

State Version Conflict

Budget Cannot Go Negative

Condition Fingerprint Changes

Planner Context Does Not Contain Tensor

Resume From Snapshot
```

---

# 4.38 模块完成标准

Unified State 完成后必须保证：

```text
任何一次 Planner 决策
都能从 Current State 重建必要 Context。

任何 Candidate
都能追溯到对应 MotionSpec / Condition / Seed。

任何 Verification
都能追溯到对应 Champion。

任何 Repair
都能追溯到对应 Failure。

State 中不存在大型 Motion Tensor。

LLM Context 不依赖完整 Run History。

进程重启后可以从最后一次 commit 恢复。

所有跨模块 Schema 有唯一 canonical definition。
```

