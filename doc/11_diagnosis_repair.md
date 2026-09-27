# MotionAgent for GEM：模块实现指南

> 本文件对应第 11 单元 Diagnosis & Repair。  
> 本模块必须与 `10_multi_verifier.md` 的 `VerificationReport / VerifierFinding / VerificationFailureSummary`、`03_motion_planner.md` 的 Action Space / Reason Code，以及 `05–08` 各执行模块的输入接口保持一致。
>
> 本模块的核心职责是：
>
> **把“哪里失败了”转换成“下一步有哪些合法、局部、可执行的修复方案”。**
>
> 它不直接执行 Repair，也不拥有全局控制权。

# 11. Diagnosis & Repair Planning

本节定义 Diagnosis & Repair Planning 的**实现规范**。

整体职责：

> **读取 Multi-Verifier 的结构化失败证据，识别失败类型与可能根因，生成一个或多个与 Planner Action Space 完全兼容的 `RepairProposal`，供顶层 Motion Planner 选择下一步动作。**

本模块采用：

```text
Rule-first Diagnosis
+
Selective LLM Reasoning
+
Specialized Repair Planning
+
Strict Structured Output
```

而不是：

```text
一个 LLM
→ 看完整 Motion
→ 自己决定所有修复
→ 自己修改 State
→ 自己重新生成
```

### LangGraph 执行边界

V1 中本模块是 `post_generation` LangGraph subgraph 的固定 `diagnosis` node。它由 graph edge 自动进入，并固定流向 `planner`；这条 edge 不由 Planner/LLM 每轮重新选择。模块内部算法和 typed service interface 保持 framework-agnostic。

---

# 11.1 在整体系统中的定位

完整链路：

```text
Planner
   │
   └── GENERATE
          │
          ▼
   GEM Generation Tool
          │
          ▼
 Candidate Tournament
          │
          ▼
    Multi-Verifier
          │
          ▼
 VerificationReport
          │
          ▼
┌──────────────────────────┐
│ Diagnosis & Repair Plan  │
└──────────────────────────┘
          │
          ├── DiagnosisResult
          └── RepairProposal[]
          │
          ▼
     Update State
          │
          ▼
        Planner
          │
          ├── COMPILE_MOTION
          ├── RETRIEVE_REFERENCE
          ├── BUILD_CONSTRAINT
          ├── BUILD_KEYFRAME
          ├── GENERATE
          ├── ACCEPT
          └── STOP_FAILED
```

注意：

```text
11 不直接调用 05/06/07/08。

11 只生成 Proposal。

03 Planner 决定真正执行哪一个 Action。
```

---

# 11.2 与第 10 单元 Multi-Verifier 的区分

必须固定：

```text
Verifier
= Observation

Diagnosis
= Interpretation

Repair Planning
= Action Proposal

Motion Planner
= Final Decision

Execution Tool
= Actual State Change
```

例如：

```text
Verifier:
right wrist error = 0.28 m
threshold = 0.05 m
```

Verifier 只输出：

```text
CONSTRAINT_CONTACT_ERROR
```

Diagnosis 再判断：

```text
这是局部 Contact Failure
已有 semantic / temporal 都通过
当前没有可靠 hard contact pose
```

Repair Planning 可以提出：

```text
Proposal A:
BUILD_CONSTRAINT
replace contact
soft

Proposal B:
RETRIEVE_REFERENCE
constraint_source
```

Planner 最后选择：

```text
BUILD_CONSTRAINT
```

或者：

```text
RETRIEVE_REFERENCE
```

---

# 11.3 与第 3 单元 Motion Planner 的区分

第 3 单元明确规定：

```text
REPAIR
```

不是 Planner Action。

因此本模块绝对不能输出：

```text
action = REPAIR
```

所有 Proposal 必须映射到已经存在的 Planner Action：

```text
COMPILE_MOTION

RETRIEVE_REFERENCE

BUILD_CONSTRAINT

BUILD_KEYFRAME

GENERATE
```

如果当前没有可恢复路径：

```text
DiagnosisResult
→ no_valid_repair = true
```

由 Planner 决定是否：

```text
STOP_FAILED
```

如果 Verify 已全部通过：

```text
repair_proposals = []
```

由 Planner 决定：

```text
ACCEPT
```

---

# 11.4 参考方法的具体使用方式

本模块主要参考：

```text
GENMAC
VISTA
NEWTON
```

但不直接复制其 Agent 数量。

---

## 11.4.1 GENMAC：Division-of-Labor

GENMAC 的 REDESIGN 阶段采用：

```text
Verification Agent
→ Suggestion Agent
→ Self-Routing
→ Correction Agent
→ Output Structuring Agent
```

其目的不是让更多 Agent 自由讨论，而是：

> **将 Verification、Routing、Correction、Structured Execution 分工。**

MotionAgent 中对应关系：

```text
GENMAC Verification Agent
→ 第 10 单元 Multi-Verifier

GENMAC Suggestion Agent + Self-Routing
→ Diagnosis Router

GENMAC Correction Agent
→ Repair Family Planner

GENMAC Output Structuring Agent
→ Pydantic Schema + Deterministic Validator
```

这样保留 GENMAC 的：

```text
division-of-labor
self-routing
specialist correction
```

但减少：

```text
不必要的 LLM Agent 数量
```

---

## 11.4.2 GENMAC Self-Routing

GENMAC 根据错误类型选择：

```text
Consistency
Temporal Dynamics
Spatial Dynamics
```

等不同 Correction Agent。

MotionAgent 也采用：

> **Failure Type → Repair Family**

但 Repair Family 与现有 Planner Action 对齐：

```text
Semantic / Temporal
→ COMPILE_MOTION

Missing Motion Prior
→ RETRIEVE_REFERENCE

Geometric / Contact / Trajectory
→ BUILD_CONSTRAINT

Whole-Body State
→ BUILD_KEYFRAME

Sampling / Naturalness / Physical Artifact
→ GENERATE
```

因此不会再创造新的执行体系。

---

## 11.4.3 VISTA：Critique → Revision

VISTA 在 Specialized Critics 后，由 reasoning agent 汇总反馈并改写下一轮 Prompt。

本项目不直接：

```text
Verifier
→ rewrite prompt
```

而是：

```text
Verifier
→ Diagnosis
→ Repair Proposal
→ Planner
```

只有 Planner 选择：

```text
COMPILE_MOTION
```

之后，Motion Compiler 才真正修改：

```text
Motion DSL
Timeline
GEM Caption
```

这样不会把所有失败都误认为：

```text
Prompt Failure
```

---

## 11.4.4 NEWTON：Verifier Feedback → Replanning

NEWTON 将：

```text
Generation
```

视为 Planner 的一个 Action，并让 Verifier Feedback 回到 Planner 进行重新规划。

本项目保持同样的控制原则：

```text
Diagnosis
→ 不直接执行下一轮

Diagnosis
→ 把 Feedback 结构化

Planner
→ 重新选择 Tool
```

因此 Planner 始终是唯一 Global Controller。

---

# 11.5 核心原则：Observed Failure ≠ Root Cause

Verifier 输出：

```text
observed failure
```

不一定直接等价于：

```text
应该修改哪个模块
```

例如：

```text
MOTION_UNNATURAL
```

可能来自：

```text
bad random sample

hard constraint 太强

segment inpainting boundary

rare motion prior 不足
```

所以 Diagnosis 必须分开：

```text
Observed Failure
→ Root Cause Hypothesis
→ Repair Family
```

不能做简单：

```text
NATURALNESS_FAILURE
→ 永远 regenerate
```

---

# 11.6 Diagnosis 的完整执行链路

固定流程：

```text
VerificationReport
        │
        ▼
1. Diagnosis Request Builder
        │
        ▼
2. Failure Normalization
        │
        ▼
3. Failure Clustering
        │
        ▼
4. Repair History Lookup
        │
        ▼
5. Rule-Based Root Cause Analysis
        │
        ├── high confidence
        │       ↓
        │   route directly
        │
        └── ambiguous
                ↓
        6. Selective Diagnosis LLM
                │
                ▼
7. Repair Family Routing
                │
                ▼
8. Specialized Repair Planner
                │
                ▼
9. Proposal Validation
                │
                ▼
DiagnosisResult
+
RepairProposal[]
        │
        ▼
MotionAgentState
        │
        ▼
Planner
```

---

# 11.7 输入接口：`DiagnosisRequest`

主接口：

```python
async def diagnose_and_plan_repair(
    request: DiagnosisRequest,
) -> DiagnosisResult:
    ...
```

Schema：

```python
class DiagnosisRequest(BaseModel):

    diagnosis_id: str

    verification_id: str

    champion_candidate_id: str

    motion_spec_id: str

    generation_id: str

    tournament_id: str

    verification_summary: VerificationFailureSummary

    finding_ids: list[str]

    active_constraint_ids: list[str]

    active_keyframe_ids: list[str]

    active_reference_ids: list[str]

    generation_scope: Literal[
        "full",
        "segment",
    ]

    target_segments: list[int] | None

    repair_history: list[RepairHistorySummary]

    budget: BudgetState

    capability_summary: CapabilitySummary
```

---

# 11.8 Diagnosis 不直接重新看完整 Raw History

Diagnosis 默认读取：

```text
VerificationFailureSummary

Relevant VerifierFinding

Current Motion Specification

Current Conditions

Recent Repair History

Generation Metadata
```

不读取：

```text
全部 Agent 对话

全部旧 Candidate Tensor

完整 151-D 序列

所有过去 Retrieval 原始结果
```

需要 Motion Evidence 时，通过：

```text
evidence_handle
```

按需读取。

---

# 11.9 `FailureNormalizer`

第 10 单元可能产生多个：

```text
VerifierFinding
```

先标准化为：

```python
class FailureCase(BaseModel):

    failure_id: str

    diagnostic_code: str

    direction: str

    severity: str

    target_segments: list[int]

    body_parts: list[str]

    source_spec_ids: list[str]

    expected: dict

    observed: dict

    metrics: dict

    confidence: float | None

    evidence_handles: list[str]

    failure_signature: str
```

---

# 11.10 `failure_signature`

用于判断：

> 这是不是上一轮同一个问题？

建议由：

```text
diagnostic_code

target segments

body parts

source constraint / keyframe / event ids
```

共同计算。

例如：

```text
CONSTRAINT_CONTACT_ERROR
+
segment 1
+
right_wrist
+
constraint_03
```

生成稳定：

```text
failure_signature
```

---

# 11.11 Failure Taxonomy

第 10 单元的 `diagnostic_code` 是最细粒度 Observation。

第 11 单元再将其映射到有限的 Failure Family。

V1 建议：

```text
SEMANTIC

EVENT

KNOWLEDGE_PRIOR

CONSTRAINT

KEYFRAME

NATURALNESS

PHYSICAL

PRESERVATION

TECHNICAL

VERIFICATION_INFRA
```

---

# 11.12 Semantic Failure

对应：

```text
SEMANTIC_MAIN_ACTION_MISMATCH

SEMANTIC_BODY_PART_MISMATCH

SEMANTIC_DIRECTION_MISMATCH
```

主要候选 Repair Family：

```text
COMPILE_MOTION

GENERATE

RETRIEVE_REFERENCE
```

具体选哪个取决于：

```text
MotionSpecification 是否正确

GEM Caption 是否正确

Failure 是否重复出现

动作是否 rare
```

---

# 11.13 Event Failure

对应：

```text
EVENT_MISSING

EVENT_ORDER_VIOLATION

EVENT_FREQUENCY_MISMATCH
```

第一候选：

```text
COMPILE_MOTION
```

对应：

```text
semantic

timeline

frequency

gem_caption
```

如果：

```text
Spec / Caption 已经正确
但只出现一次随机失败
```

可以先：

```text
GENERATE
```

而不是立刻重写 Motion Plan。

---

# 11.14 Knowledge / Prior Failure

不是第 10 单元直接产生的单一 Code。

它通常由：

```text
rare style

semantic failure repeated

naturalness failure repeated

已有 Caption 正确
但多轮 GEM 仍无法生成
```

推断。

Repair Family：

```text
RETRIEVE_REFERENCE
```

---

# 11.15 Constraint Failure

对应：

```text
CONSTRAINT_POSITION_ERROR

CONSTRAINT_CONTACT_ERROR

CONSTRAINT_TRAJECTORY_ERROR
```

Repair Family：

```text
BUILD_CONSTRAINT

RETRIEVE_REFERENCE

GENERATE(strategy="guided")
```

取决于：

```text
Constraint 是否已正确编译

hard / soft mode

reference 是否存在

normal generation 是否已经重复失败

guided capability 是否可用
```

---

# 11.16 Keyframe Failure

对应：

```text
KEYFRAME_POSE_MISMATCH
```

Repair Family：

```text
BUILD_KEYFRAME
```

必要时：

```text
RETRIEVE_REFERENCE(
    purpose="keyframe_source"
)
```

---

# 11.17 Naturalness Failure

对应：

```text
MOTION_UNNATURAL

MOTION_DISCONTINUITY
```

可能 Repair：

```text
GENERATE normal

GENERATE segment

RETRIEVE_REFERENCE

BUILD_CONSTRAINT replace hard→soft
```

必须结合：

```text
最近是否增加了 Hard Constraint
mask density
generation scope
repair history
```

判断。

---

# 11.18 Physical Failure

对应：

```text
PHYSICS_FOOT_SKATING

PHYSICS_GROUND_PENETRATION
```

V1 优先：

```text
GENERATE normal
```

如果重复：

```text
RETRIEVE_REFERENCE(
    purpose="motion_prior"
)
```

如果后续已有：

```text
physical RewardSpec
```

可建议：

```text
GENERATE guided
```

但：

> 如果当前 Guided Generator 不支持该 reward，不允许提出 guided Proposal。

---

# 11.19 Preservation Failure

对应：

```text
PRESERVATION_FAILURE
```

只在：

```text
scope="segment"
```

中出现。

可能：

```text
GENERATE scope="segment"
```

再尝试，

或如果局部 inpainting 持续破坏全局一致性：

```text
GENERATE scope="full"
```

Planner 决定采用哪个。

---

# 11.19.1 Heading Continuity Failure

新增标准 Failure Code：

```text
UNCOMMANDED_HEADING_DRIFT
```

触发条件来自 Multi-Verifier，而不是 Diagnosis 自己重新估计 Motion。

Diagnosis 首先区分三类 root cause：

### A. Specification Missing

```text
用户没有要求转向
但 MotionSpecification 没有 continuity policy
```

→ `SPECIFICATION_INCOMPLETE`

建议：

```text
COMPILE_MOTION
focus=["heading_continuity", "gem_caption"]
```

### B. Sampling Variance

```text
continuity policy 正确
caption 也包含 continuity cue
只在某个 seed 出现明显漂移
```

→ `SAMPLING_VARIANCE`

建议：

```text
GENERATE
```

### C. Control Too Weak / Persistent Drift

```text
policy 正确
连续多个 generation 仍发生 drift
```

→ `CONSTRAINT_TOO_WEAK` 或 `GUIDANCE_NEEDED`

建议：

```text
BUILD_CONSTRAINT
使用 root_trajectory orientation-only soft control
```

必要时后续再进入 guided generation。

禁止：

```text
因为 heading drift 就重写所有动作段；
把正确的 explicit turn 当成 drift；
默认 hard-freeze 整段 root orientation。
```

# 11.20 Technical Failure

例如：

```text
Candidate NaN

Invalid Motion Output

Hard Condition 未生效

Adapter Output Invalid
```

这些不是 Semantic Repair。

优先：

```text
GENERATE
```

如果确定：

```text
Tool / Adapter 持续失败
```

则：

```text
no_valid_repair = true
terminal_hint = UNRECOVERABLE_TOOL_FAILURE
```

交给 Planner。

---

# 11.21 Verification Infrastructure Failure

例如：

```text
required verifier service unavailable

required MLLM 连续 timeout

verification status = failed_service
```

此时：

```text
禁止 COMPILE / CONSTRAINT / KEYFRAME
```

因为没有证据说明 Motion 本身错误。

Verifier 内部 Infrastructure Retry 已执行后仍失败：

```text
Diagnosis:
no_valid_repair = true
terminal_hint = UNRECOVERABLE_TOOL_FAILURE
```

---

# 11.22 Failure Clustering

同一轮可能同时失败多个 Check。

例如：

```text
EVENT_MISSING

EVENT_ORDER_VIOLATION

EVENT_FREQUENCY_MISMATCH
```

都发生在：

```text
segment 1
```

如果分成三个独立 Repair：

```text
COMPILE
COMPILE
COMPILE
```

没有意义。

因此要先做：

```text
Failure Clustering
```

---

## 11.22.1 Cluster Key

建议：

```text
Failure Family

Target Segment

Source Spec
```

例如：

```python
cluster_key = (
    failure_family,
    tuple(target_segments),
    primary_source_spec_id,
)
```

---

## 11.22.2 Example

```text
segment 1:
EVENT_ORDER_VIOLATION
EVENT_FREQUENCY_MISMATCH
```

合并为：

```text
EVENT cluster
```

生成一个：

```text
COMPILE_MOTION
focus=[
    "timeline",
    "frequency",
    "gem_caption"
]
```

---

# 11.23 不要把所有 Failure 合成一个 Repair

如果：

```text
segment 0 semantic fail

segment 2 contact fail
```

不能生成：

```text
一个超级 Repair
同时改 Prompt + Constraint
```

应该生成：

```text
Proposal A
COMPILE_MOTION segment 0

Proposal B
BUILD_CONSTRAINT segment 2
```

Planner 每轮仍然只选择：

```text
ONE next action
```

保持 03 的 Bounded-ReAct 设计。

---

# 11.24 Root Cause Analysis

Diagnosis 不追求输出：

```text
“真正的因果真相”
```

而是输出：

```text
best supported repair hypothesis
```

Schema：

```python
class RootCauseHypothesis(BaseModel):

    cause_code: str

    confidence: float

    evidence_failure_ids: list[str]

    supporting_facts: list[str]

    contradicting_facts: list[str]

    affected_segments: list[int]
```

---

# 11.25 V1 Root Cause Codes

建议限制：

```text
SPECIFICATION_INCOMPLETE

CAPTION_MISMATCH

MISSING_MOTION_PRIOR

SAMPLING_VARIANCE

CONSTRAINT_TOO_WEAK

CONSTRAINT_TOO_STRONG

CONSTRAINT_TARGET_INADEQUATE

KEYFRAME_INADEQUATE

GUIDANCE_NEEDED

SEGMENT_BOUNDARY_ARTIFACT

PHYSICAL_SAMPLING_ARTIFACT

CONDITION_APPLICATION_FAILURE

UNKNOWN
```

---

# 11.26 Rule-First Diagnosis

大量 Failure 不需要 LLM。

例如：

```text
EVENT_FREQUENCY_MISMATCH
+
Motion DSL repetition=3
+
GEM caption 没写 three times
```

直接：

```text
cause = CAPTION_MISMATCH

Proposal:
COMPILE_MOTION
focus=["frequency","gem_caption"]
```

---

# 11.27 什么时候需要 Diagnosis LLM

只在以下情况启用：

```text
多个 Failure 之间可能存在共同原因

Semantic Specification 看起来正确但生成持续失败

Naturalness Failure 可能由 Constraint 引起

Physical Failure 与 Constraint / Segment Repair 相互影响

多个 Repair Family 都合法但没有明显规则优势
```

简单：

```text
joint target error
```

不需要 LLM 来“思考人生”。

---

# 11.28 `DiagnosisLLMContext`

只提供压缩信息：

```python
class DiagnosisLLMContext(BaseModel):

    original_request: str

    motion_spec_summary: dict

    failure_clusters: list[dict]

    active_conditions: dict

    latest_generation: dict

    repair_history: list[dict]

    capabilities: dict
```

禁止直接给：

```text
完整 Raw Motion Tensor

完整 Agent History
```

---

# 11.29 Diagnosis LLM Prompt

```text
You are the Motion Failure Diagnoser for MotionAgent.

Your task is to interpret structured verification failures
and identify the most plausible repair-relevant cause.

You do NOT execute repairs.
You do NOT choose the final Planner action.
You do NOT rewrite the user's request.
You do NOT generate motion.

Use only the provided evidence.

Important principles:

1. Preserve components that already passed verification.

2. Distinguish:
   - specification failure;
   - missing motion prior;
   - constraint/control failure;
   - sampling failure;
   - naturalness/physical artifact;
   - segment-preservation failure.

3. Do not assume that every generated-motion failure is a prompt failure.

4. Do not recommend changing a passed segment unless evidence
   shows that it contributes to the failure.

5. Prefer the smallest repair surface consistent with the evidence.

6. If evidence is insufficient, use UNKNOWN rather than inventing a cause.

7. Return repair-relevant causes, not hidden chain-of-thought.

Return only the required RootCauseHypothesis schema.
```

---

# 11.30 Repair Family

V1 Repair Family 与 Planner Action 一一对齐：

```text
MOTION_COMPILE

REFERENCE_RETRIEVAL

CONSTRAINT_UPDATE

KEYFRAME_UPDATE

REGENERATION
```

映射：

```text
MOTION_COMPILE
→ COMPILE_MOTION

REFERENCE_RETRIEVAL
→ RETRIEVE_REFERENCE

CONSTRAINT_UPDATE
→ BUILD_CONSTRAINT

KEYFRAME_UPDATE
→ BUILD_KEYFRAME

REGENERATION
→ GENERATE
```

---

# 11.31 `RepairProposal`

这是第 11 单元最核心的输出对象。

```python
class RepairProposal(BaseModel):

    proposal_id: str

    repair_family: Literal[
        "MOTION_COMPILE",
        "REFERENCE_RETRIEVAL",
        "CONSTRAINT_UPDATE",
        "KEYFRAME_UPDATE",
        "REGENERATION",
    ]

    planner_action: Literal[
        "COMPILE_MOTION",
        "RETRIEVE_REFERENCE",
        "BUILD_CONSTRAINT",
        "BUILD_KEYFRAME",
        "GENERATE",
    ]

    planner_reason_code: str

    target_segments: list[int] | None

    payload: dict

    evidence_failure_ids: list[str]

    root_cause_codes: list[str]

    preserves: list[str]

    preconditions: list[str]

    expected_effect: str

    confidence: float

    repair_signature: str
```

---

# 11.32 `RepairProposal` 不是 PlannerDecision

区别：

```text
RepairProposal
= Diagnosis 给 Planner 的候选方案

PlannerDecision
= Planner 最终选择的 ONE Action
```

Planner 可以：

```text
采用 Proposal

忽略 Proposal

根据 Budget 选择另一个 Proposal

STOP_FAILED
```

因此 11 不会架空 03 Planner。

---

# 11.33 Planner Reason Code 必须兼容 03

RepairProposal 不允许创造 Planner 未定义的 Reason Code。

必须从：

```text
SEMANTIC_FAILURE

TEMPORAL_FAILURE

RARE_MOTION

MISSING_REFERENCE

CONSTRAINT_FAILURE

NATURALNESS_FAILURE

PERSISTENT_CONSTRAINT_FAILURE

PERSISTENT_FAILURE

WHOLE_BODY_STATE_REQUIRED

UNRECOVERABLE_TOOL_FAILURE
```

等第 3 单元已有 Reason Code 中选择。

---

# 11.34 Semantic Repair Planner

输入：

```text
Semantic / Event Failure Cluster

Root Cause

Current Motion Segment

Current GEM Caption

Relevant Retrieved Captions
```

输出：

```text
COMPILE_MOTION Proposal
```

---

## 11.34.1 Main Action / Body Part / Direction

例如：

```text
SEMANTIC_BODY_PART_MISMATCH
```

Proposal：

```json
{
  "repair_family": "MOTION_COMPILE",
  "planner_action": "COMPILE_MOTION",
  "planner_reason_code": "SEMANTIC_FAILURE",
  "target_segments": [1],
  "payload": {
    "mode": "revise",
    "focus": [
      "semantic",
      "body_part",
      "gem_caption"
    ]
  }
}
```

---

## 11.34.2 Event Order

```json
{
  "planner_action": "COMPILE_MOTION",
  "planner_reason_code": "TEMPORAL_FAILURE",
  "target_segments": [1, 2],
  "payload": {
    "mode": "revise",
    "focus": [
      "timeline",
      "gem_caption"
    ]
  }
}
```

---

## 11.34.3 Frequency

```json
{
  "planner_action": "COMPILE_MOTION",
  "planner_reason_code": "TEMPORAL_FAILURE",
  "target_segments": [1],
  "payload": {
    "mode": "revise",
    "focus": [
      "frequency",
      "gem_caption"
    ]
  }
}
```

---

# 11.35 Retrieval Repair Planner

只在有证据支持：

```text
missing prior
rare motion
reference needed
```

时生成。

---

## 11.35.1 Prompt Grounding

例如：

```text
rare style
+
Caption 不稳定
```

```json
{
  "planner_action": "RETRIEVE_REFERENCE",
  "planner_reason_code": "RARE_MOTION",
  "target_segments": [0],
  "payload": {
    "query": "asymmetric limping gait",
    "retrieval_type": "caption",
    "purpose": "prompt_grounding",
    "top_k": 5
  }
}
```

Planner 之后可能：

```text
COMPILE_MOTION revise gem_caption
```

---

## 11.35.2 Motion Prior

```json
{
  "planner_action": "RETRIEVE_REFERENCE",
  "planner_reason_code": "MISSING_REFERENCE",
  "target_segments": [0],
  "payload": {
    "query": "unstable staggering forward walk",
    "retrieval_type": "motion",
    "purpose": "motion_prior",
    "top_k": 5
  }
}
```

---

## 11.35.3 Constraint Source

```json
{
  "planner_action": "RETRIEVE_REFERENCE",
  "planner_reason_code": "MISSING_REFERENCE",
  "target_segments": [1],
  "payload": {
    "query": "right hand reaching and supporting on a surface",
    "retrieval_type": "motion",
    "purpose": "constraint_source",
    "top_k": 5
  }
}
```

---

# 11.36 Constraint Repair Planner

输入：

```text
Failed Constraint ID

ConstraintFinding

Current CompiledConstraint

Constraint Mode

Relevant Reference

Previous Repair Outcome
```

输出：

```text
BUILD_CONSTRAINT Proposal
```

---

## 11.36.1 Replace Weak Constraint

```json
{
  "planner_action": "BUILD_CONSTRAINT",
  "planner_reason_code": "CONSTRAINT_FAILURE",
  "target_segments": [1],
  "payload": {
    "mode": "replace",
    "constraint_id": "constraint_03",
    "constraint_type": "contact",
    "body_part": "right_wrist",
    "time_range": [2.8, 3.3],
    "target": "table_surface",
    "strength": "hard"
  }
}
```

只有：

```text
target geometry 已解析

hard condition 可构造
```

时才允许提出。

---

## 11.36.2 Constraint Too Strong

例如：

```text
新增 Hard Constraint 后
Naturalness 明显下降
+
mask density 高
```

Proposal：

```json
{
  "planner_action": "BUILD_CONSTRAINT",
  "planner_reason_code": "CONSTRAINT_FAILURE",
  "target_segments": [1],
  "payload": {
    "mode": "replace",
    "constraint_id": "constraint_03",
    "constraint_type": "contact",
    "body_part": "right_wrist",
    "time_range": [2.8, 3.3],
    "target": "table_surface",
    "strength": "soft"
  }
}
```

---

# 11.37 Keyframe Repair Planner

如果：

```text
KEYFRAME_POSE_MISMATCH
```

且当前 Keyframe 本身：

```text
不够明确
reference 不合适
```

Proposal：

```json
{
  "planner_action": "BUILD_KEYFRAME",
  "planner_reason_code": "WHOLE_BODY_STATE_REQUIRED",
  "target_segments": [2],
  "payload": {
    "mode": "add",
    "time": 5.8,
    "description": "a stable seated whole-body pose",
    "source_preference": "retrieval"
  }
}
```

如果是已有 Keyframe 的更新，

`07b_keyframe_tool.md` 的正式 Schema 已支持：

```text
mode="replace"
```

因此已有 Keyframe 的修复必须携带：

```text
keyframe_id
mode="replace"
```

并通过 Keyframe Proposal Validator 检查。

---

# 11.38 Generation Repair Planner

用于：

```text
输入条件看起来已经正确
但 Sample 本身失败
```

---

## 11.38.1 Normal Regeneration

第一次 Sampling Failure：

```json
{
  "planner_action": "GENERATE",
  "planner_reason_code": "NATURALNESS_FAILURE",
  "target_segments": [1],
  "payload": {
    "strategy": "normal",
    "scope": "segment",
    "num_candidates": 4,
    "reward_targets": []
  }
}
```

---

## 11.38.2 Full Regeneration

如果：

```text
Failure 跨多个 Segment

Motion Plan 发生较大变化

Segment Preservation 连续失败
```

可以建议：

```json
{
  "planner_action": "GENERATE",
  "planner_reason_code": "PERSISTENT_FAILURE",
  "target_segments": null,
  "payload": {
    "strategy": "normal",
    "scope": "full",
    "num_candidates": 4,
    "reward_targets": []
  }
}
```

---

## 11.38.3 Guided Regeneration

只有满足：

```text
compatible RewardSpec exists

Guided Generator available

normal generation has already failed
```

才允许：

```json
{
  "planner_action": "GENERATE",
  "planner_reason_code": "PERSISTENT_CONSTRAINT_FAILURE",
  "target_segments": [1],
  "payload": {
    "strategy": "guided",
    "scope": "segment",
    "num_candidates": 4,
    "reward_targets": [
      "contact"
    ]
  }
}
```

如果 Guided Capability 不存在：

```text
禁止生成该 Proposal
```

---

# 11.39 Semantic Failure 不一定立刻 COMPILE

这是很重要的 Rule。

如果：

```text
Motion DSL 正确

GEM Caption 正确

只生成了一轮

只有 Champion semantic fail
```

根因可能只是：

```text
sampling variance
```

第一候选可以：

```text
GENERATE normal
```

而不是立即修改 Prompt。

---

## 11.39.1 什么时候更像 Compiler Failure

例如：

```text
Motion DSL 本身遗漏 body part

Caption 与 DSL 不一致

Event Count 未进入 Caption

Timeline 编译错误
```

则：

```text
COMPILE_MOTION
```

是明确 Repair。

---

## 11.39.2 什么时候更像 Missing Prior

例如：

```text
DSL 正确

Caption 正确

连续多轮相同 rare style 失败

TMR / Retrieval 表明该 style 罕见
```

则：

```text
RETRIEVE_REFERENCE
```

优先级提高。

---

# 11.40 Naturalness Failure 不一定直接 Regenerate

例如：

```text
上一轮刚把 contact
从 soft 改成 hard

之后：
contact pass
naturalness fail
```

这说明可能：

```text
CONSTRAINT_TOO_STRONG
```

而不是：

```text
bad seed
```

此时 Proposal 应包括：

```text
BUILD_CONSTRAINT
replace hard → soft
```

而不仅是：

```text
GENERATE another seed
```

---

# 11.41 多 Failure 的 Root Cause 合并

例如：

```text
MOTION_UNNATURAL

PHYSICS_FOOT_SKATING

PRESERVATION_FAILURE
```

都在：

```text
segment 1
```

且来自一次：

```text
segment regeneration
```

可以形成：

```text
root cause:
SEGMENT_BOUNDARY_ARTIFACT
```

Proposal：

```text
GENERATE scope="segment"
```

或重复失败后：

```text
GENERATE scope="full"
```

而不是生成三个互相竞争的独立 Proposal。

---

# 11.42 Repair Proposal Ranking 的边界

Diagnosis 可以给 Proposal：

```text
confidence

evidence support
```

但：

> 最终哪一个 Proposal 被执行，仍由 Planner 决定。

不要在 Diagnosis 内：

```text
执行 top-1
```

---

# 11.43 Proposal Confidence

Confidence 代表：

```text
“这个 Repair 与当前 Evidence 的匹配程度”
```

不是：

```text
“执行后一定成功的概率”
```

第一版只作为 Planner Context / Debug 信息。

---

# 11.44 `DiagnosisResult`

最终输出：

```python
class DiagnosisResult(BaseModel):

    diagnosis_id: str

    verification_id: str

    status: Literal[
        "no_failure",
        "diagnosed",
        "ambiguous",
        "no_valid_repair",
        "service_error",
    ]

    failure_cases: list[FailureCase]

    failure_clusters: list[FailureCluster]

    root_causes: list[RootCauseHypothesis]

    repair_proposals: list[RepairProposal]

    unresolved_failures: list[str]

    terminal_hint: str | None

    diagnosis_summary: str

    evidence_fingerprint: str
```

---

# 11.45 Verification Pass

如果：

```text
VerificationReport.overall_pass = true
```

直接：

```python
DiagnosisResult(
    status="no_failure",
    repair_proposals=[],
    terminal_hint=None,
)
```

Planner 读取：

```text
overall_pass=true
```

后选择：

```text
ACCEPT
```

Diagnosis 不需要制造：

```text
ACCEPT Proposal
```

---

# 11.46 `FailureCluster`

```python
class FailureCluster(BaseModel):

    cluster_id: str

    family: str

    failure_ids: list[str]

    target_segments: list[int]

    body_parts: list[str]

    source_spec_ids: list[str]

    severity: str

    persistent: bool
```

---

# 11.47 Persistent Failure

不是：

```text
同一个 diagnostic_code 出现过一次
```

就算 persistent。

建议条件：

```text
same failure_signature
+
至少一次针对该问题的 Repair
+
新的 Verification 后仍失败
```

然后：

```text
persistent = true
```

---

# 11.48 Repair History

为了实现第 3 单元的：

```text
同一 Failure 连续两轮使用同一种 Repair Family
且无改善
→ 必须换 Repair Family
```

第 11 单元必须维护：

```text
RepairHistory
```

---

# 11.49 `RepairHistoryEntry`

```python
class RepairHistoryEntry(BaseModel):

    repair_id: str

    failure_signature: str

    proposal_id: str

    planner_action: str

    repair_family: str

    target_segments: list[int]

    repair_signature: str

    before_verification_id: str

    after_verification_id: str | None

    outcome: Literal[
        "pending",
        "resolved",
        "improved",
        "unchanged",
        "worse",
        "failed_execution",
    ]

    before_metrics: dict

    after_metrics: dict

    created_at: str
```

---

# 11.50 `repair_signature`

由：

```text
repair family

action

target segments

relevant payload fields
```

计算。

例如：

```text
BUILD_CONSTRAINT
+
constraint_03
+
hard
+
segment_1
```

形成稳定 Signature。

---

# 11.51 Repair Effect Tracker

Repair 执行后，不立即知道有没有成功。

必须等下一轮：

```text
VerificationReport
```

回来后再更新。

接口：

```python
def update_repair_effect(
    history_entry: RepairHistoryEntry,
    new_verification: VerificationReport,
) -> RepairHistoryEntry:
    ...
```

---

# 11.52 Repair Outcome

对于 Numeric Failure：

```text
before error = 0.28

after error = 0.12

threshold = 0.05
```

虽然仍 Fail：

```text
outcome = improved
```

不能记成：

```text
unchanged
```

---

## 11.52.1 `resolved`

```text
same failure signature
→ pass
```

---

## 11.52.2 `improved`

```text
metric 朝目标明显改善
但仍未 pass
```

---

## 11.52.3 `unchanged`

```text
几乎没有变化
```

---

## 11.52.4 `worse`

```text
metric 更差
或引入新的 major/critical failure
```

---

# 11.53 No-Improvement Guard

如果：

```text
same failure_signature

same repair_family

两次 outcome:
unchanged / worse
```

则：

```text
block_same_family = true
```

Planner Guard：

```text
下一轮不能继续相同 Repair Family
```

除非：

```text
State 有实质变化
```

---

# 11.54 Preserve Passed Components

每个 RepairProposal 必须包含：

```text
preserves
```

例如：

```json
{
  "preserves": [
    "segment_0",
    "segment_2",
    "constraint_01",
    "retrieval_ref_02"
  ]
}
```

用于提醒 Planner / Executor：

> 已经通过的部分默认不动。

---

# 11.55 Minimal Repair Surface

默认原则：

```text
Local Failure
→ Local Repair
```

例如：

```text
segment 1 frequency fail
```

Proposal：

```text
COMPILE_MOTION
target_segments=[1]
```

不是：

```text
重写整个 Motion Plan
```

---

# 11.56 什么时候允许 Full Repair

只有：

```text
Failure 跨多个 Segment

Global semantics wrong

Motion Plan 本身结构错误

Segment regeneration 多次无法保持 continuity
```

才允许：

```text
target_segments = null

scope = full
```

---

# 11.57 Repair Proposal Validator

所有 Proposal 输出后必须经过：

```python
validate_repair_proposal(
    proposal,
    state,
)
```

---

## 11.57.1 通用检查

```text
planner_action 属于 03 Action Space

reason_code 属于 03 Reason Code

target segment 合法

payload schema 合法

budget 足够

capability available

不违反 duplicate guard

不违反 no-improvement guard
```

---

# 11.58 `COMPILE_MOTION` Proposal Validator

检查：

```text
mode ∈ {initial, revise}

repair 时必须 mode = revise

focus 合法

target segment 合法
```

---

# 11.59 `RETRIEVE_REFERENCE` Proposal Validator

检查：

```text
query 非空

retrieval_type 合法

purpose 合法

不是 duplicate retrieval

retrieval budget > 0
```

---

# 11.60 `BUILD_CONSTRAINT` Proposal Validator

检查：

```text
constraint type 合法

target 已存在 / 可解析

replace 时 constraint_id 存在

strength 合法

没有明显 conflict
```

如果缺少 geometry：

```text
不能提 hard constraint
```

---

# 11.61 `BUILD_KEYFRAME` Proposal Validator

检查：

```text
time 合法

target segment 合法

source preference 可用
```

---

# 11.62 `GENERATE` Proposal Validator

检查：

```text
generation budget > 0

segment scope 时 previous candidate 存在

guided 时 capability available

guided 时 RewardSpec compatible

num_candidates > 0
```

---

# 11.63 Repair Family Router

V1 优先使用：

```text
Rule Table
```

而不是让 LLM 自由选 Tool。

---

# 11.64 Rule Table

建议第一版固定：

| Diagnostic / Cause | Primary Repair Family | Alternative |
|---|---|---|
| `SEMANTIC_MAIN_ACTION_MISMATCH` + spec/caption wrong | `MOTION_COMPILE` | — |
| `SEMANTIC_BODY_PART_MISMATCH` + spec/caption wrong | `MOTION_COMPILE` | — |
| semantic fail + spec/caption correct + first failure | `REGENERATION` | — |
| repeated rare-style semantic fail | `REFERENCE_RETRIEVAL` | `MOTION_COMPILE` |
| `EVENT_MISSING` | `MOTION_COMPILE` | `REGENERATION` |
| `EVENT_ORDER_VIOLATION` | `MOTION_COMPILE` | — |
| `EVENT_FREQUENCY_MISMATCH` | `MOTION_COMPILE` | `REGENERATION` |
| `CONSTRAINT_POSITION_ERROR` | `CONSTRAINT_UPDATE` | `REGENERATION` |
| `CONSTRAINT_CONTACT_ERROR` | `CONSTRAINT_UPDATE` | `REFERENCE_RETRIEVAL` |
| `CONSTRAINT_TRAJECTORY_ERROR` | `CONSTRAINT_UPDATE` | `REFERENCE_RETRIEVAL` |
| repeated soft constraint fail + guidance available | `REGENERATION guided` | `CONSTRAINT_UPDATE` |
| `KEYFRAME_POSE_MISMATCH` | `KEYFRAME_UPDATE` | `REFERENCE_RETRIEVAL` |
| `MOTION_UNNATURAL` first occurrence | `REGENERATION` | — |
| naturalness fail after new hard constraint | `CONSTRAINT_UPDATE` | `REGENERATION` |
| repeated naturalness fail on rare action | `REFERENCE_RETRIEVAL` | `REGENERATION` |
| `PHYSICS_FOOT_SKATING` first occurrence | `REGENERATION` | — |
| repeated locomotion physical fail | `REFERENCE_RETRIEVAL` | `REGENERATION` |
| `PRESERVATION_FAILURE` | `REGENERATION segment` | `REGENERATION full` |
| technical invalid candidate | `REGENERATION` | — |
| verifier service unrecoverable | none | terminal hint |

该表可以放入：

```text
configs/repair_policy.yaml
```

避免散落在代码里。

---

## 11.64.1 Heading Continuity Rule

固定 routing：

| Evidence | Primary Repair | Secondary Repair |
|---|---|---|
| continuity policy 缺失/错误 | `COMPILE_MOTION` | `REGENERATION` |
| policy 正确但单次漂移 | `REGENERATION` | `CONSTRAINT_UPDATE` |
| policy 正确且重复漂移 | `CONSTRAINT_UPDATE` | `GUIDED_REGENERATION` |
| 用户显式 turn 被误报 | 修 Verifier / exemption | 不修改 Motion |

Planner 仍然每轮只选择一个现有 Action。

# 11.65 多 Proposal 输出

Diagnosis 可以返回：

```text
1~3 个
```

合法 Proposal。

不要一次返回十几个候选。

例如：

```text
contact fail
```

可以返回：

```text
Proposal 1:
BUILD_CONSTRAINT replace

Proposal 2:
RETRIEVE_REFERENCE constraint_source
```

Planner 再结合：

```text
Budget

History

Capabilities
```

选下一步。

---

# 11.66 Proposal 不应该修改 State

Diagnosis 结束时只写：

```python
state.diagnosis = diagnosis_summary

state.repair_proposals = proposal_summaries
```

禁止：

```python
state.motion_plan = ...
state.constraints = ...
state.keyframes = ...
```

真正修改发生在：

```text
Planner Decision
→ 对应 Tool Execution
```

---

# 11.67 Planner Context 写回

Planner 只需要看到压缩 Summary：

```json
{
  "diagnosis": {
    "status": "diagnosed",
    "primary_failures": [
      "CONSTRAINT_CONTACT_ERROR"
    ],
    "root_causes": [
      "CONSTRAINT_TOO_WEAK"
    ],
    "target_segments": [1],
    "repair_proposals": [
      {
        "proposal_id": "rp_01",
        "action": "BUILD_CONSTRAINT",
        "reason_code": "CONSTRAINT_FAILURE",
        "confidence": 0.91
      },
      {
        "proposal_id": "rp_02",
        "action": "RETRIEVE_REFERENCE",
        "reason_code": "MISSING_REFERENCE",
        "confidence": 0.63
      }
    ]
  }
}
```

完整 Payload 存：

```text
Diagnosis Store
```

Planner 选择 Proposal 后再加载。

---

# 11.68 Diagnosis Service

建议服务结构：

```text
Orchestrator
      │
      ▼
Diagnosis Service
      │
      ├── Verification Store
      ├── Motion Spec Store
      ├── Constraint Store
      ├── Keyframe Store
      ├── Repair History Store
      │
      ├── Failure Normalizer
      ├── Failure Clusterer
      ├── Rule Diagnosis Engine
      ├── Optional LLM Diagnoser
      ├── Repair Router
      ├── Repair Proposal Builders
      └── Proposal Validator
```

---

# 11.69 对外接口

```python
class DiagnosisService:

    async def diagnose(
        self,
        request: DiagnosisRequest,
    ) -> DiagnosisResult:
        ...
```

---

# 11.70 内部代码结构

建议：

```text
repair/
│
├── schemas.py
├── service.py
│
├── failure_normalizer.py
├── failure_cluster.py
├── root_cause.py
├── policy.py
├── validator.py
│
├── history/
│   ├── store.py
│   ├── effect_tracker.py
│   └── signatures.py
│
├── planners/
│   ├── semantic.py
│   ├── retrieval.py
│   ├── constraint.py
│   ├── keyframe.py
│   └── generation.py
│
├── llm/
│   ├── diagnoser.py
│   └── prompt.py
│
└── configs/
    └── repair_policy.yaml
```

---

# 11.71 主流程伪代码

```python
async def diagnose_and_plan_repair(
    request: DiagnosisRequest,
) -> DiagnosisResult:

    verification = verification_store.load(
        request.verification_id
    )

    if verification.overall_pass:

        return DiagnosisResult(
            diagnosis_id=request.diagnosis_id,
            verification_id=request.verification_id,
            status="no_failure",
            failure_cases=[],
            failure_clusters=[],
            root_causes=[],
            repair_proposals=[],
            unresolved_failures=[],
            terminal_hint=None,
            diagnosis_summary="No repair required.",
            evidence_fingerprint=
                build_evidence_fingerprint(
                    verification
                ),
        )

    if verification.status == "failed_service":

        return no_valid_repair_result(
            request=request,
            terminal_hint=
                "UNRECOVERABLE_TOOL_FAILURE",
        )

    findings = verification_store.load_findings(
        request.finding_ids
    )

    failures = normalize_failures(
        findings
    )

    clusters = cluster_failures(
        failures
    )

    history = repair_history_store.lookup(
        failure_signatures=[
            x.failure_signature
            for x in failures
        ]
    )

    root_causes = []

    for cluster in clusters:

        rule_result = rule_diagnose(
            cluster=cluster,
            request=request,
            history=history,
        )

        if rule_result.confident:

            root_causes.extend(
                rule_result.root_causes
            )

        else:

            llm_result = await diagnosis_llm.diagnose(
                build_llm_context(
                    cluster=cluster,
                    request=request,
                    history=history,
                )
            )

            root_causes.extend(
                llm_result.root_causes
            )

    proposals = build_repair_proposals(
        clusters=clusters,
        root_causes=root_causes,
        request=request,
        history=history,
    )

    valid_proposals = []

    for proposal in proposals:

        validation = validate_repair_proposal(
            proposal=proposal,
            request=request,
            history=history,
        )

        if validation.valid:

            valid_proposals.append(
                proposal
            )

    if not valid_proposals:

        return DiagnosisResult(
            ...,
            status="no_valid_repair",
            terminal_hint=
                infer_terminal_hint(
                    request,
                    failures,
                ),
        )

    return DiagnosisResult(
        diagnosis_id=request.diagnosis_id,
        verification_id=request.verification_id,
        status=(
            "diagnosed"
            if all_root_causes_confident(
                root_causes
            )
            else "ambiguous"
        ),
        failure_cases=failures,
        failure_clusters=clusters,
        root_causes=root_causes,
        repair_proposals=limit_proposals(
            valid_proposals,
            max_count=3,
        ),
        unresolved_failures=find_unresolved(
            failures,
            valid_proposals,
        ),
        terminal_hint=None,
        diagnosis_summary=build_summary(
            failures,
            root_causes,
        ),
        evidence_fingerprint=
            build_evidence_fingerprint(
                verification
            ),
    )
```

---

# 11.72 Failure Normalization 伪代码

```python
def normalize_failures(
    findings,
):

    failures = []

    for finding in findings:

        if finding.status not in [
            "fail",
            "uncertain",
            "error",
        ]:
            continue

        failures.append(
            FailureCase(
                failure_id=new_id(),
                diagnostic_code=
                    finding.diagnostic_code,
                direction=
                    finding.direction,
                severity=
                    finding.severity,
                target_segments=
                    finding.target_segments,
                body_parts=
                    finding.body_parts,
                source_spec_ids=
                    finding.source_spec_ids,
                expected=
                    finding.expected,
                observed=
                    finding.observed,
                metrics=
                    finding.metrics,
                confidence=
                    finding.confidence,
                evidence_handles=
                    finding.evidence_handles,
                failure_signature=
                    build_failure_signature(
                        finding
                    ),
            )
        )

    return failures
```

---

# 11.73 Semantic Diagnosis 伪代码

```python
def diagnose_semantic_failure(
    cluster,
    motion_spec,
    gem_text_condition,
    history,
):

    if specification_missing_requirement(
        cluster,
        motion_spec,
    ):

        return root_cause(
            "SPECIFICATION_INCOMPLETE"
        )

    if caption_missing_requirement(
        cluster,
        gem_text_condition,
    ):

        return root_cause(
            "CAPTION_MISMATCH"
        )

    if is_persistent(cluster, history):

        if is_rare_motion(
            motion_spec,
            cluster.target_segments,
        ):
            return root_cause(
                "MISSING_MOTION_PRIOR"
            )

    return root_cause(
        "SAMPLING_VARIANCE"
    )
```

---

# 11.74 Constraint Diagnosis 伪代码

```python
def diagnose_constraint_failure(
    cluster,
    constraint,
    history,
    generation_meta,
):

    if constraint is None:
        return root_cause(
            "CONSTRAINT_TARGET_INADEQUATE"
        )

    if (
        constraint.effective_mode
        == "hard_condition"
        and generation_meta.hard_mask_preservation_error
        > allowed_preservation_error
    ):
        return root_cause(
            "CONDITION_APPLICATION_FAILURE"
        )

    if (
        has_recent_naturalness_failure(
            cluster,
            history,
        )
        and generation_meta.mask_density
        > configured_warning_level
    ):
        return root_cause(
            "CONSTRAINT_TOO_STRONG"
        )

    if (
        constraint.effective_mode
        in [
            "selection_reward",
            "guided_reward",
        ]
        and is_persistent(
            cluster,
            history,
        )
    ):
        return root_cause(
            "GUIDANCE_NEEDED"
        )

    return root_cause(
        "CONSTRAINT_TOO_WEAK"
    )
```

---

# 11.75 Repair Proposal Builder 伪代码

```python
def build_repair_proposals(
    clusters,
    root_causes,
    request,
    history,
):

    proposals = []

    for cluster in clusters:

        causes = causes_for_cluster(
            root_causes,
            cluster.cluster_id,
        )

        families = route_repair_families(
            cluster=cluster,
            causes=causes,
            history=history,
            capabilities=
                request.capability_summary,
        )

        for family in families:

            planner = REPAIR_PLANNER_REGISTRY[
                family
            ]

            proposal = planner.build(
                cluster=cluster,
                causes=causes,
                request=request,
            )

            proposals.append(
                proposal
            )

    return deduplicate_proposals(
        proposals
    )
```

---

# 11.76 Repair Planner Registry

```python
REPAIR_PLANNER_REGISTRY = {

    "MOTION_COMPILE":
        SemanticRepairPlanner(),

    "REFERENCE_RETRIEVAL":
        RetrievalRepairPlanner(),

    "CONSTRAINT_UPDATE":
        ConstraintRepairPlanner(),

    "KEYFRAME_UPDATE":
        KeyframeRepairPlanner(),

    "REGENERATION":
        GenerationRepairPlanner(),
}
```

---

# 11.77 LLM 不是每个 Repair Family 都必须使用

V1 建议：

```text
Semantic Repair
→ LLM helpful

Retrieval Query
→ DSL/template + optional LLM

Constraint Repair
→ deterministic

Keyframe Repair
→ deterministic/template

Generation Repair
→ deterministic
```

这样避免：

```text
五个 LLM Agent
```

带来的：

```text
成本
不稳定
难调试
```

---

# 11.78 Infrastructure Retry 与 Agent Repair 再次区分

第 11 单元只处理：

```text
Motion / Control Repair
```

以下情况：

```text
API timeout

schema parse failure

GPU transient error
```

由对应服务内部：

```text
Infrastructure Retry
```

处理。

不记录为：

```text
semantic repair
```

---

# 11.79 Repair Budget

Repair Proposal Validator 必须读取：

```text
state.budget
```

例如：

```text
generation budget = 0
```

则禁止生成：

```text
GENERATE Proposal
```

如果：

```text
retrieval budget = 0
```

则禁止：

```text
RETRIEVE_REFERENCE Proposal
```

---

# 11.80 Terminal Hint

Diagnosis 不能执行：

```text
STOP_FAILED
```

但可以提供：

```python
terminal_hint: Literal[
    "UNRECOVERABLE_TOOL_FAILURE",
    "BUDGET_EXHAUSTED",
] | None
```

Planner 结合 Hard Guard 决定：

```text
STOP_FAILED
```

---

# 11.81 Diagnosis Cache

Cache Key：

```text
verification_id

motion_spec_version

active condition fingerprint

repair history version

capability version

diagnosis policy version
```

如果任何一项改变：

```text
cache invalid
```

---

# 11.82 Idempotency

同一个：

```text
DiagnosisRequest
```

必须得到：

```text
同样的 Failure Normalization
同样的 Rule Routing
```

如果使用 LLM：

```text
temperature = 0
strict structured output
prompt version recorded
```

保证尽量稳定。

---

# 11.83 Diagnosis Store

保存：

```text
DiagnosisResult

RootCauseHypothesis

RepairProposal

Proposal Validation Result
```

用于：

```text
Debug

Ablation

Planner Training

Repair Success Analysis
```

---

# 11.84 Planner RL 数据接口

每次完整闭环：

```text
State_t

Planner Action_t

Tool Observation_t

Verification_t

Diagnosis_t

Repair Outcome_{t+1}
```

可以记录：

```python
class AgentTransition(BaseModel):

    state_id: str

    planner_action: dict

    observation_ids: list[str]

    verification_id: str | None

    diagnosis_id: str | None

    reward: float | None
```

但 V1：

```text
只记录
不训练
```

---

# 11.85 V1 实现顺序

## Step 1：Schema

先实现：

```text
FailureCase

FailureCluster

RootCauseHypothesis

RepairProposal

DiagnosisResult

RepairHistoryEntry
```

---

## Step 2：Failure Normalizer

完成：

```text
VerifierFinding
→ FailureCase
```

---

## Step 3：Rule Repair Policy

先覆盖：

```text
Semantic

Event

Constraint

Keyframe

Naturalness

Preservation
```

---

## Step 4：Proposal Validator

保证所有 Proposal：

```text
能直接映射到 03 Planner Action
```

---

## Step 5：Repair History

实现：

```text
failure_signature

repair_signature

effect tracking
```

---

## Step 6：Specialized Proposal Builders

实现：

```text
Semantic

Retrieval

Constraint

Keyframe

Generation
```

---

## Step 7：Selective Diagnosis LLM

只处理：

```text
ambiguous cases
```

---

## Step 8：No-Improvement Guard

实现：

```text
same family
+
same failure
+
no improvement
→ block
```

---

## Step 9：Service / Store / Cache

最后工程化：

```text
Diagnosis Service

Diagnosis Store

Cache

Prompt Version
```

---

# 11.86 V1 最低实现范围

V1 必须完成：

```text
DiagnosisRequest

Failure Normalizer

Failure Clustering

Failure Taxonomy

Rule-First Root Cause

Selective Diagnosis LLM

Repair Family Router

Semantic Repair Proposal

Retrieval Repair Proposal

Constraint Repair Proposal

Keyframe Repair Proposal

Generation Repair Proposal

Proposal Validator

Repair History

Repair Effect Tracker

No-Improvement Guard

DiagnosisResult

Planner Handoff
```

V1 不需要：

```text
训练 Diagnosis Model

训练 Repair Router

复杂 Multi-Agent Debate

五个独立大型模型

自动执行多步 Repair Chain

Planner RL

GEM Finetuning
```

---

# 11.87 单元测试要求

至少覆盖：

```text
Verification Pass

Single Semantic Failure

Body-Part Failure

Direction Failure

Event Missing

Temporal Order Failure

Frequency Failure

Rare Style Repeated Failure

Constraint Position Failure

Contact Failure

Trajectory Failure

Keyframe Failure

Naturalness First Failure

Naturalness After Hard Constraint

Foot Skating

Segment Preservation Failure

Multiple Failures Same Cluster

Multiple Failures Different Segments

Persistent Failure

No-Improvement Guard

Budget Exhausted

Guided Capability Missing

Verifier Service Failure

Technical Candidate Failure

Duplicate Repair Proposal

Proposal Schema Validation
```

---

## Case 1：Verification Pass

```text
overall_pass = true

Expected:
status = no_failure

repair_proposals = []

Planner may ACCEPT
```

---

## Case 2：Frequency Failure

```text
Expected = 3

Observed = 2

Motion DSL has repetition=3

Caption omitted "three times"

Expected Diagnosis:
CAPTION_MISMATCH

Proposal:
COMPILE_MOTION
focus=[
    "frequency",
    "gem_caption"
]
```

---

## Case 3：Semantic Failure but Spec Correct

```text
Motion DSL correct

Caption correct

First generation round

Champion performs wrong action

Expected:
SAMPLING_VARIANCE

Proposal:
GENERATE normal
```

不要立即重写全部 Prompt。

---

## Case 4：Repeated Rare Style Failure

```text
staggering style

Spec correct

Caption correct

2 failed rounds

No reference exists

Expected:
MISSING_MOTION_PRIOR

Proposal:
RETRIEVE_REFERENCE
purpose=motion_prior
```

---

## Case 5：Contact Failure

```text
constraint_03:
contact

error = 0.20m
threshold = 0.05m

Expected:
CONSTRAINT failure cluster

Proposal:
BUILD_CONSTRAINT replace
or
RETRIEVE_REFERENCE constraint_source
```

---

## Case 6：Naturalness After Hard Constraint

```text
Before:
naturalness pass

Repair:
contact soft → hard

After:
contact pass
naturalness fail
mask density high

Expected:
CONSTRAINT_TOO_STRONG

Proposal:
BUILD_CONSTRAINT
replace hard → soft

Not:
blind regenerate forever
```

---

## Case 7：No Improvement

```text
same contact failure

Repair family:
CONSTRAINT_UPDATE

Round 1:
unchanged

Round 2:
unchanged

Expected:
block same repair family

Planner must choose another family
or STOP_FAILED
```

---

## Case 8：Verifier Service Failure

```text
verification.status = failed_service

Expected:
no semantic / constraint proposal

status = no_valid_repair

terminal_hint =
UNRECOVERABLE_TOOL_FAILURE
```

---

# 11.88 Diagnosis / Repair 评估指标

Diagnosis Accuracy：

```text
Failure Family Accuracy

Root Cause Agreement

Target Segment Accuracy

Target Body-Part Accuracy
```

Routing：

```text
Repair Family Routing Accuracy

Planner-compatible Proposal Validity

Invalid Proposal Rate

Duplicate Proposal Rate
```

Repair Effect：

```text
Repair Success Rate

Resolved Failure Rate

Improvement Rate

No-Improvement Rate

Regression Rate
```

Agent Efficiency：

```text
Average Repair Rounds

Average Generation Rounds

Tool Calls per Successful Task

Repeated Repair Family Rate

Budget Exhaustion Rate
```

---

# 11.89 Targeted Repair Success

重点指标：

```text
Targeted Repair Success Rate
```

定义：

> Repair 后目标 Failure 得到解决，同时已通过的无关部分没有出现新的 Major / Critical Failure。

因为：

```text
只修好一个问题
但破坏另外两个已通过问题
```

不能算成功 Repair。

---

# 11.90 Regression Rate

记录：

```text
Repair 之前 pass

Repair 之后 fail
```

的 Check 数量。

例如：

```text
contact 修好了

semantic / naturalness 被破坏
```

则算：

```text
Repair Regression
```

---

# 11.91 Ablation

建议比较：

```text
No Diagnosis:
Verifier → always regenerate

Rule-only Repair

LLM-only Diagnosis

Rule-first + Selective LLM

Single Generic Repair Agent

Specialized Repair Family

Without Repair History

With No-Improvement Guard
```

重点验证：

```text
是否真正减少无效 generation rounds
```

---

# 11.92 参考工作与本模块对应关系

| 工作 | 借鉴内容 | 本模块位置 |
|---|---|---|
| GENMAC | Verification → Suggestion → Correction → Structured Output | 总体流程 |
| GENMAC | Self-Routing | Failure → Repair Family |
| GENMAC | Specialist Correction Agents | Specialized Repair Planners |
| VISTA | Specialized Critique → reasoning → next-iteration revision | Diagnosis synthesis |
| NEWTON | Verifier feedback returned to Planner | Planner Handoff |
| PhysAgent | Stage-specific verification + targeted executable repair | Repair targeting reference |

---

# 11.93 当前模块完成标准

Diagnosis & Repair Planning 完成后，必须稳定实现：

```text
VerificationReport
       ↓
FailureNormalizer
       ↓
FailureCase[]
       ↓
FailureCluster[]
       ↓
Root Cause Analysis
       ↓
Repair Family Routing
       ↓
Specialized Repair Planner
       ↓
RepairProposal[]
       ↓
Proposal Validation
       ↓
DiagnosisResult
       ↓
MotionAgentState
       ↓
Planner
```

并保证：

```text
Diagnosis 不重新做 Verification。

Diagnosis 不重新选择 Candidate。

Diagnosis 不直接修改 Motion Plan。

Diagnosis 不直接添加 Constraint。

Diagnosis 不直接生成 Keyframe。

Diagnosis 不直接调用 GEM。

Diagnosis 不直接 Retry。

RepairProposal 必须映射到 03 已有 Planner Action。

Planner 每轮仍只选择 ONE next action。

已经通过的 Motion / Constraint / Segment 默认保留。

局部 Failure 优先局部 Repair。

连续无改善时必须切换 Repair Family。

没有可靠证据时允许 UNKNOWN，
禁止强行编造 Root Cause。

只有 Planner 才拥有最终全局控制权。
```

# 11.94 Heading Continuity Completion Addendum

Diagnosis & Repair 还必须保证：

```text
UNCOMMANDED_HEADING_DRIFT 能区分 specification missing / sampling variance / weak control；
优先局部修复；
单次 drift 不直接升级为 hard constraint；
持续 drift 才建议 soft root-heading control；
不新增 Planner Action。
```
