# MotionAgent for GEM：Project Overview & Implementation Blueprint

> **用途**  
> 本文件是整个 MotionAgent 项目的总入口。实现任何模块之前，先阅读本文件，再进入对应模块文档。  
> 本文件以 `03_motion_planner.md`、`04_unified_state.md`、`05–11` 及 `07b_keyframe_tool.md` 的最新接口为 source of truth。

# 0. 项目目标

> Phase 7 milestone revision: complete Agentic Generation Infrastructure.
> Condition construction, adapter forwarding and artifact persistence are
> required; arbitrary hard-constraint satisfaction, exact keyframe forcing
> and diffusion-time inpainting are deferred to future Guided Generation.
> The full-project architecture below remains the long-term design.

MotionAgent 的目标是：

> **在不重新训练 GEM 的前提下，在 Frozen GEM 外建立一个可规划、可检索、可约束、可验证、可定向修复的 Agentic Control Layer，使复杂 Human Motion 指令更稳定地得到满足。**

核心输出：

```text
Structured 3D Human Motion
→ SMPL / GEM Motion Representation
```

非 V1 核心：

```text
RGB video generation
GEM finetuning
full physics simulation
```

---

# 1. V1 研究边界

V1 固定：

```text
Frozen GEM
Frozen T5 text encoder

+ MotionAgent Harness
+ LangGraph orchestration runtime
+ Motion Planner
+ Motion Compiler
+ Retrieval
+ Constraint
+ Keyframe
+ K-candidate Generation
+ Tournament
+ Multi-Verifier
+ Diagnosis / Targeted Repair
```

V1 不做：

```text
GEM weight update
GEM RL post-training
new keyframe diffusion network
new constraint encoder training
```

因此如果 V1 的复杂指令成功率提高，可以较清楚地归因于：

```text
Agentic control / test-time reasoning / structured conditioning
```

而不是 foundation model 重新训练。

---

# 2. 系统总架构

建议从四个逻辑 Plane 理解系统。

```text
┌────────────────────────────────────────────────────┐
│                   Control Plane                    │
│                                                    │
│ Motion Planner                                     │
│ State / Guard / Budget / Capability                │
└─────────────────────────┬──────────────────────────┘
                          │
                          ▼
┌────────────────────────────────────────────────────┐
│                  Condition Plane                   │
│                                                    │
│ Motion Compiler                                    │
│ Retrieval Tool                                     │
│ Constraint Compiler                                │
│ Keyframe Tool                                      │
└─────────────────────────┬──────────────────────────┘
                          │
                          ▼
┌────────────────────────────────────────────────────┐
│                  Generation Plane                  │
│                                                    │
│ GEM Adapter → Frozen GEM → K Motion Candidates     │
└─────────────────────────┬──────────────────────────┘
                          │
                          ▼
┌────────────────────────────────────────────────────┐
│             Evaluation / Improvement Plane         │
│                                                    │
│ Candidate Tournament                              │
│ → Multi-Verifier                                  │
│ → Diagnosis & Repair Planning                     │
│ → Motion Planner                                  │
└────────────────────────────────────────────────────┘

      ─────────── Harness / Infrastructure ───────────
 LangGraph StateGraph / Routing / Checkpointer
 State Store / Artifact Store / Event Log / Cache
 GPU Workers / Render Workers / Tracing / Config / Eval
```

---

# 3. Agent / Workflow / Tool 的正式划分

必须统一口径。

| 模块 | 类型 | 是否有全局控制权 |
|---|---|---|
| Motion Planner | **Agent** | 是，唯一 Global Controller |
| Motion Compiler | LLM-assisted Tool / bounded workflow | 否 |
| Retrieval | Tool | 否 |
| Constraint Compiler | Numerical Tool | 否 |
| Keyframe Tool | Numerical / retrieval-assisted Tool | 否 |
| GEM Generator | GPU Tool / Workflow | 否 |
| Candidate Tournament | Fixed Workflow + Judge Service | 否 |
| Multi-Verifier | Fixed Workflow + specialized evaluators | 否 |
| Diagnosis & Repair | Rule-first Workflow + selective LLM | 否 |
| State / Guard / Budget | Harness | 否 |

不要把所有模块都叫 Agent。

V1 真正的 Autonomous Agent 只有：

```text
Motion Planner
```

其余模块的职责是：

```text
给定明确输入
→ 执行有限职责
→ 返回严格结构化结果
```

## LangGraph 的正式定位

V1 显式采用 **LangGraph StateGraph** 作为 Control / Orchestration Runtime。

```text
LangGraph
= graph execution + conditional routing + checkpoint / resume

Motion Planner
= 唯一拥有全局语义决策权的 Agent

Tool / Workflow
= 被 graph node wrapper 调用的有限职责 domain service
```

必须区分：

```text
LangGraph Node ≠ Autonomous Agent
```

LangGraph 不重新定义 Planner policy、MotionAgentState、Tool Contract、Artifact Store、Guard、Verifier threshold 或 GPU worker；这些仍然是 MotionAgent 自己的 domain / harness 设计。这样可以让 framework-specific orchestration 与 motion-specific research logic 解耦。

---

# 4. Planner Action Space

第 3 单元定义唯一 Action Space：

```text
COMPILE_MOTION
RETRIEVE_REFERENCE
BUILD_CONSTRAINT
BUILD_KEYFRAME
GENERATE
ACCEPT
STOP_FAILED
```

明确不存在：

```text
VERIFY
REPAIR
REGENERATE
GENERATE_GUIDED
TOURNAMENT
```

原因：

```text
Tournament / Verify / Diagnosis
属于 GENERATE 后的 automatic subgraph。

Repair
通过下一轮选择已有 Action 实现。

Guided
是 GENERATE.strategy。
```

---

# 5. 最完整 End-to-End 数据流

```text
User Request
      │
      ▼
Create MotionAgentRun
      │
      ▼
Unified State
      │
      ▼
PlannerContextBuilder
      │
      ▼
Motion Planner
      │
      ├── COMPILE_MOTION
      ├── RETRIEVE_REFERENCE
      ├── BUILD_CONSTRAINT
      ├── BUILD_KEYFRAME
      ├── GENERATE
      ├── ACCEPT
      └── STOP_FAILED
      │
      ▼
Selected Tool
      │
      ▼
Structured Result
      │
      ▼
State Reducer / State Commit
      │
      └───────────────────────────→ Planner
```

当 Action = `GENERATE`：

```text
PlannerDecision
      ↓
GenerationRequestBuilder
      ↓
Generation Preflight
      ↓
Frozen GEM
      ↓
K Candidates
      ↓
GenerationResult commit
      ↓
Candidate Tournament
      ↓
Champion commit
      ↓
VerifierPlanBuilder
      ↓
Multi-Verifier
      ↓
VerificationReport commit
      ↓
Diagnosis & Repair Planning
      ↓
DiagnosisResult commit
      ↓
PlannerContextBuilder
      ↓
Motion Planner
```

这一段不再次询问 Planner：

```text
要不要 Tournament？
要不要 Verify？
要不要 Diagnose？
```

它们是固定系统流程。

---

# 6. 03 Motion Planner

## 功能

回答：

> **当前状态下，下一步应该做什么？**

每一轮严格输出：

```python
PlannerDecision
```

并且：

```text
ONE turn
→ ONE action
```

---

## 输入

只读压缩：

```python
PlannerContext(
    task,
    plan,
    conditions,
    latest_generation,
    latest_verification,
    diagnosis,
    history,
    budget,
    capabilities,
)
```

Planner 不读取：

```text
SMPL Tensor
151-D Motion
完整 VerificationReport
完整 Retrieval DB Result
完整 Action Transcript
```

---

## 输出

```python
PlannerDecision(
    action,
    reason_code,
    target_segments,
    reason_summary,
    payload,
)
```

---

## 证明做好的指标

```text
Legal Action Rate

Routing Accuracy

Illegal ACCEPT Rate = 0

Repeated-action Loop Rate

Budget Violation Rate = 0

Repair Routing Efficiency
```

---

# 7. 04 Unified State / Harness State

## 功能

回答：

> **当前系统真实状态是什么？各模块如何共享状态而不膨胀 LLM Context？**

采用：

```text
Current State
+
Artifact Store
+
Event Log
```

State 只保存：

```text
IDs
Summaries
Budget
Capabilities
Version
```

重型数据通过 Handle 外存。

---

## 关键能力

```text
State Reducer
State Invariants
Artifact IDs
Versioning
Optimistic Concurrency
PlannerContextBuilder
LangGraph Checkpointer / Resume
Canonical State Reconciliation
Fingerprints
```

---

## 证明做好的指标

```text
State Replay Success

Missing Artifact Reference Rate = 0

Invalid State Commit Rate = 0

Planner Context Tensor Leakage = 0

Resume Success Rate

Schema Contract Test Pass Rate
```

---

# 8. 05 Motion Compiler

## 功能

回答：

> **用户到底要求什么 Motion？怎样把它转换成 GEM 更容易消费的条件？**

```text
Natural Language
→ Semantic Parser
→ Temporal Resolver
→ Human Motion DSL
→ Timeline
→ Control Intent Hints
→ GEM-aware Caption
→ GEMTextCondition
```

---

## 输出

```text
MotionSpecification
GEMTextCondition
RoutingHints
```

RoutingHints：

```text
retrieval_candidate
constraint_candidate
keyframe_candidate
```

只提示 Planner，不自动执行。

---

## GEM 接口

```text
Motion Segment
→ Caption
→ normalized window_start / window_end
→ multi_text_data
→ frozen T5
→ GEM cross-attention
```

---

## 参考工作

```text
LAMP
→ Language → structured motion program

RAPO
→ generator-aware prompt optimization

HumanML3D
→ motion-language corpus / caption style
```

---

## 证明做好的指标

```text
DSL Semantic Accuracy

Timeline Accuracy

Body-Part / Direction / Structured Temporal Recall

Caption Semantic Preservation

Caption Token Limit Compliance

GEM multi_text_data Correctness
```

---

## Cross-Segment Continuity：默认运动连续性

Motion Compiler 除了显式动作语义，还必须生成**跨动作段连续性期望**。V1 首先实现最重要的一项：

```text
Heading / Facing Continuity
```

典型例子：

```text
walk forward
→ wave the right hand
→ sit down
```

如果用户没有显式要求：

```text
turn / rotate / face another direction / spin
```

则后续 Segment 默认继承前一动作已经建立的朝向，而不是允许 GEM 任意改变 root heading。

正式数据流：

```text
User Request
→ Motion Compiler
→ MotionSpecification.continuity_policy
→ GEM caption / generation condition
→ Candidate Tournament continuity evidence
→ Multi-Verifier heading-continuity check
→ Diagnosis
→ Planner repair
```

这不是新的 Planner Action，也不是新的 Agent。它是贯穿现有模块的结构化 expectation。

V1 原则：

```text
Compiler
→ 推断“应该保持什么”

Constraint Compiler
→ 在持续失败时提供 soft root-heading control

Tournament
→ 相对偏好更少无指令转向的 Candidate

Verifier
→ 绝对检查 uncommanded heading drift

Diagnosis / Planner
→ 决定 recompile / constraint / regenerate
```

默认不要对整段 `global_orient` 做 hard freeze；自然的小幅身体转动必须允许。只有用户显式指定固定朝向，或存在可靠 reference/root-orientation target 时，才允许升级为更强控制。

# 9. 06 Retrieval Tool

## 功能

回答：

> **当前 Segment 是否需要外部 Motion Knowledge？检索到的 Reference 是什么？**

支持：

```text
caption
motion
pose
trajectory
```

用途：

```text
prompt_grounding
motion_prior
keyframe_source
constraint_source
```

---

## V1 数据

```text
HumanML3D
+
TMR
```

---

## Pipeline

```text
MotionSegment
→ Retrieval Query
→ TMR Encoding
→ Top-N
→ Filter
→ Structured Rerank
→ Top-K RetrievedReference
```

---

## 关键边界

Retrieval：

```text
只找 Reference
```

不：

```text
修改 Motion Plan
生成 Constraint
生成 Keyframe
直接修改 GEM Input
```

---

## 参考工作

```text
TMR
RAPO
Retrieval-Guided DNO
```

---

## 证明做好的指标

```text
Recall@K

Top-K Semantic Relevance

Train/Test Leakage Rate = 0

TMR ↔ GEM Motion ID Mapping Coverage

Low-confidence Calibration

Downstream Utilization / Improvement Rate
```

---

# 10. 07 Constraint Compiler

## 功能

回答：

> **一个可测量的 Motion Requirement 应怎样变成 GEM 可执行数值条件？**

V1：

```text
joint_target
body_part_pose
root_trajectory
contact
fixed_joint
```

---

## 输出

```python
CompiledConstraint(
    hard_condition_handle,
    reward_spec,
    verification_spec,
)
```

同一个 Constraint Spec 被：

```text
Generation
Guidance
Verification
```

共同使用。

---

## Hard Path

```text
Target
→ FK / IK / Reference
→ SMPL
→ GEM EnDecoder
→ 151-D
→ Feature Mask
→ observed_motion_3d + motion_mask_3d
```

---

## Soft Path

```text
Target
→ RewardSpec
```

V1：

```text
Candidate selection / verifier
```

Phase II：

```text
Guided generation
```

---

## 参考工作

```text
Retrieval-Guided DNO
OmniControl / spatial control literature
NEWTON deterministic scientific tools
```

---

## 证明做好的指标

```text
Compile Success Rate

IK Residual

Hard Mask Accuracy

Constraint Conflict Detection

Generation / Verification Spec Consistency

Joint / Contact / Trajectory Error
```

---

# 11. 07B Keyframe Tool

## 功能

回答：

> **某个时间点必须出现的完整 Whole-Body State，怎样变成 GEM Keyframe Condition？**

Keyframe ≠ Constraint：

```text
Constraint
→ local / geometric relationship

Keyframe
→ complete whole-body state at time t
```

---

## Source

V1 优先：

```text
explicit pose
retrieved pose
retrieved motion frame
existing candidate frame
IK-refined reliable base pose
```

不允许：

```text
LLM text
→ invent SMPL numbers
```

---

## Pipeline

```text
BUILD_KEYFRAME
→ KeyframeRequest
→ Source Resolver
→ Whole-Body Pose
→ Optional IK
→ GEM Encode
→ HardMotionCondition
→ KeyframeVerificationSpec
```

---

## 参考工作

```text
GMD
→ sparse keyframe / spatial guidance

KeyMotion
→ keyframe-first + motion infilling

Sparse Keyframe Motion Generation literature
```

---

## 证明做好的指标

```text
Keyframe Satisfaction Rate

Joint Rotation Error

Joint Position Error

Temporal Match Offset

Naturalness Regression

Conflict Detection Rate
```

---

# 12. 08 GEM Generation Tool

## 功能

回答：

> **给定已经编译好的 Conditions，如何稳定调用 Frozen GEM 生成可复现的 Candidate？**

---

## 输入

```python
GenerationRequest(
    strategy,
    scope,
    target_segments,
    text_condition,
    condition_bundle,
    num_candidates,
    seeds,
    ...
)
```

---

## Normal Generation

```text
PureTextGEMAdapter
→ Text + Hard Conditions
→ Frozen GEM
→ K Seeds
→ K Candidates
```

---

## Segment Repair

不是：

```text
生成短片
→ 拼接
```

而是：

```text
previous full motion
→ preserve mask outside failed segment
→ free target segment
→ full-sequence inpainting
```

---

## V1 关键约束

```text
Frozen GEM

B=1 wrapper
→ K candidate sequential generation

persistent GPU worker

same seed reproducibility
```

---

## 证明做好的指标

```text
Same-seed Reproducibility

Different-seed Diversity

Hard Mask Preservation Error

Outside-Segment Preservation Error

Candidate Failure Rate

GPU Latency / VRAM / OOM Rate
```

---

# 13. 09 Candidate Tournament

## 功能

回答：

> **当前 K 个合法 Candidate 中，哪个最值得进入绝对验证？**

这是：

```text
relative selection
```

不是 absolute acceptance。

---

## Pipeline

```text
Candidates
→ Lightweight Render / Metrics
→ Candidate Probe
→ Binary Bracket
→ A/B
→ B/A
→ Tie Resolution
→ Champion
```

---

## Hybrid Evidence

```text
MLLM
TMR
MotionCritic
Constraint quick metrics
Event evidence
```

---

## 参考工作

```text
VISTA
→ pairwise tournament / probing / swap

AToM
→ event evidence

MotionCritic
→ naturalness

TMR
→ semantic
```

---

## 证明做好的指标

```text
Forward / Swapped Consistency

Human Pairwise Agreement

Bracket Stability

Champion Verifier Pass Rate

Champion > Random / metric-only baseline
```

---

# 14. 10 Multi-Verifier

## 功能

回答：

> **Champion 是否真正达到当前任务的 Required Conditions？**

---

## VerifierPlanBuilder

不是每轮全部跑。

根据 Task 自动选择：

```text
technical
semantic
naturalness

event_integrity
event_temporal
event_frequency

constraint
keyframe
physical
preservation
```

---

## Event

参考 AToM：

```text
Integrity
Temporal
Frequency
```

独立验证。

---

## Naturalness

```text
MotionCritic
+
kinematic sanity
```

---

## Constraint / Keyframe

直接读取：

```text
VerificationSpec
KeyframeVerificationSpec
```

不重新让 LLM 解释自然语言。

---

## Final Pass

不采用总分互相抵消。

采用：

```text
Required-Check Gating
```

即：

```text
complete
AND
all required checks pass
→ overall_pass = true
```

---

## 参考工作

```text
VISTA specialized critics
AToM
MotionCritic
TMR
PP-Motion
```

---

## 证明做好的指标

最重要：

```text
False Accept Rate
```

以及：

```text
Verifier / Human Agreement

Failure Localization Accuracy

Event Accuracy

Constraint Pass/Fail Accuracy

Incomplete Verification Rate
```

---

# 15. 11 Diagnosis & Repair Planning

## 功能

回答：

> **已经发现的失败最可能是什么原因？下一步有哪些最小、合法、可执行的 Repair Proposal？**

---

## Pipeline

```text
VerificationReport
→ Failure Normalization
→ Failure Cluster
→ Root Cause
→ Repair Family
→ RepairProposal[]
→ Planner
```

---

## Repair Family

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

## 核心边界

```text
RepairProposal
≠
PlannerDecision
```

Diagnosis 只建议；Planner 最终决定。

---

## 参考工作

```text
GENMAC
→ division of labor / self-routing / correction specialization

VISTA
→ critique → revision

NEWTON
→ verifier feedback → replanning
```

---

## 证明做好的指标

```text
Failure Family Accuracy

Repair Family Routing Accuracy

Targeted Repair Success Rate

Repair Regression Rate

Average Repair Rounds

Repeated No-Improvement Rate
```

---

# 16. Automatic Post-Generation Subgraph

该 Subgraph 在 V1 中作为 **LangGraph 固定子图** 实现，而不是由 Planner 临时决定路径：

```text
generation
→ tournament
→ verifier
→ diagnosis
→ planner
```

其中每个 node 只做 adapter / orchestration：

```text
Graph Node
→ build typed request
→ call domain service
→ persist heavy artifact
→ StateReducer / atomic commit
→ return lightweight GraphState delta
```

Planner 不参与该 Subgraph 中间 routing，也不存在 `TOURNAMENT / VERIFY / DIAGNOSE` Planner Action。

示意：

```python
builder.add_edge("generation", "tournament")
builder.add_edge("tournament", "verifier")
builder.add_edge("verifier", "diagnosis")
builder.add_edge("diagnosis", "planner")
```

这样 Graph topology 与系统语义保持一致，同时 Tournament / Verifier / Diagnosis 的算法实现继续保留在各自 domain service 中。

# 17. 最终推荐代码架构

```text
motion_agent/
│
├── app/
│   ├── orchestrator.py
│   ├── lifecycle.py
│   └── api.py
│
├── graph/
│   ├── builder.py
│   ├── state.py
│   ├── routing.py
│   ├── checkpoints.py
│   └── nodes/
│       ├── planner.py
│       ├── tool_executor.py
│       └── post_generation.py
│
├── domain/
│   ├── planner.py
│   ├── motion.py
│   ├── retrieval.py
│   ├── constraint.py
│   ├── keyframe.py
│   ├── generation.py
│   ├── tournament.py
│   ├── verification.py
│   ├── diagnosis.py
│   └── state.py
│
├── agent/
│   ├── planner.py
│   ├── context_builder.py
│   ├── guards.py
│   ├── budgets.py
│   └── skills/
│
├── state/
│   ├── store.py
│   ├── reducer.py
│   ├── invariants.py
│   ├── event_store.py
│   └── versioning.py
│
├── compiler/
├── retrieval/
├── constraints/
├── keyframes/
├── generation/
├── tournament/
├── verification/
├── repair/
├── rendering/
│
├── stores/
│   ├── artifact_store.py
│   ├── metadata_store.py
│   └── cache.py
│
├── workers/
│   ├── gpu_queue.py
│   ├── gem_worker.py
│   └── render_worker.py
│
├── observability/
│   ├── tracing.py
│   ├── metrics.py
│   └── logging.py
│
├── evaluation/
│   ├── module_evals/
│   ├── integration/
│   ├── regression/
│   └── ablation/
│
├── common/
│   ├── ids.py
│   ├── fingerprints.py
│   ├── errors.py
│   └── time.py
│
└── configs/
    ├── planner.yaml
    ├── retrieval.yaml
    ├── constraints.yaml
    ├── keyframes.yaml
    ├── generation.yaml
    ├── tournament.yaml
    ├── verifier_thresholds.yaml
    └── repair_policy.yaml
```

目录边界：

```text
graph/
→ LangGraph-specific topology / node adapter / routing / checkpoint integration

agent/ + domain/ + state/
→ MotionAgent 自己的 policy、schema、guard、reducer 与业务语义

workers/
→ 独立 GPU / render execution，不放进 LLM graph state
```

`app/orchestrator.py` 负责创建/加载 Run、compile/invoke LangGraph，并提供 API lifecycle；它不再维护一套与 LangGraph 重复的手写 `while + if/elif` 状态机。

---

# 18. 模块接口 Contract Matrix

| Producer | Output | Consumer |
|---|---|---|
| Motion Compiler | `MotionSpecification` | Planner / Tournament / Verifier |
| Motion Compiler | `GEMTextCondition` | Generator |
| Motion Compiler | `RoutingHints` | Planner |
| Retrieval | `RetrievedReference[]` | Compiler / Constraint / Keyframe / Guided Generator |
| Constraint | `CompiledConstraint` | Condition Assembler / Verifier |
| Constraint | `VerificationSpec` | Verifier |
| Keyframe | `KeyframeSpec` | Condition Assembler / Verifier |
| Keyframe | `KeyframeVerificationSpec` | Verifier |
| Generator | `GenerationResult` | Tournament |
| Candidate Store | `MotionCandidate` | Tournament / Verifier / Segment Inpainting |
| Tournament | `TournamentResult` | Verifier / State |
| Verifier | `VerificationReport` | Diagnosis / Planner Summary |
| Diagnosis | `DiagnosisResult` | Planner |
| Planner | `PlannerDecision` | LangGraph Router / Tool Request Builder |

必须对这些连接写：

```text
contract tests
```

---

# 19. Request Builder Pattern

Planner 不应该生成底层 Tool 完整参数。

统一：

```text
PlannerDecision
+
Current State
→ ModuleRequestBuilder
→ Strongly Typed Tool Request
```

例如：

```text
PlannerDecision BUILD_KEYFRAME
↓
KeyframeRequestBuilder
↓
KeyframeRequest
```

这能缩短 Planner Prompt，也能减少参数错误。

---

# 20. State Update Pattern

任何 Tool：

```text
Tool Result
→ Persist Artifact
→ State Reducer
→ Invariant Validation
→ Atomic Commit
→ Event Log
```

不要让 Tool 自己直接：

```text
mutate shared state
```

---

# 21. Fingerprint / Idempotency

统一维护：

```text
retrieval_signature
condition_fingerprint
candidate_fingerprint
criteria_fingerprint
evidence_fingerprint
failure_signature
repair_signature
```

用途：

```text
cache
idempotency
duplicate guard
reproducibility
repair tracking
```

---

# 22. Error Classification

全系统统一三类：

## Infrastructure Error

```text
timeout
GPU transient failure
network
schema transport error
```

模块内部有限 retry。

## Motion / Agent Failure

```text
semantic
constraint
naturalness
physical
```

必须：

```text
Verify → Diagnose → Planner
```

## Unrecoverable Error

```text
invalid checkpoint
missing required artifact
service unavailable after retries
```

返回 Planner：

```text
UNRECOVERABLE_TOOL_FAILURE
```

---

# 23. 实现阶段建议

## Phase A：可跑通闭环

```text
04 State
03 Planner
05 Compiler
08 Pure Text GEM
09 Tournament basic
10 basic verifier
11 basic repair
```

## Phase B：可控生成

```text
06 Retrieval
07 Constraint
07B Keyframe
```

## Phase C：稳定工程化

```text
Artifact Store
Event Log
GPU Worker
Cache
Trace
Regression Tests
```

## Phase D：增强

```text
Guided Generation
DNO / ReAlign-style guidance
Planner RL
```

---

# 24. 每个模块的 Definition of Done

| 模块 | Done 条件 |
|---|---|
| 03 Planner | 只输出合法 Action；ACCEPT guard 100% 生效；无明显 action loop |
| 04 State | Run 可重放；Artifact 可追溯；PlannerContext 不含 Tensor；可 resume |
| 05 Compiler | DSL/timeline/caption 正确；multi-text 映射正确 |
| 06 Retrieval | Recall@K 有效；无 test leakage；Reference 能被下游消费 |
| 07 Constraint | 数值目标正确；hard mask 正确；VerificationSpec 一致 |
| 07B Keyframe | key pose 可执行；hard anchor 正确；Verifier 可复用 Spec |
| 08 Generator | seed 可复现；K candidate 正确；segment preservation 达标 |
| 09 Tournament | swap 稳定；human agreement 合理；Champion 优于随机选择 |
| 10 Verifier | False Accept 低；failure localization 准确 |
| 11 Diagnosis | targeted repair 提高成功率；重复无效修复受到阻止 |

---

# 25. Integration Definition of Done

完整系统必须通过下面的集成 Case。

## Case A：Simple Text

```text
walk forward
```

期望：

```text
Compile
→ Generate
→ Tournament
→ Verify
→ Accept
```

---

## Case B：Sequential Events

```text
walk, wave right hand three times, sit down
```

必须触发：

```text
Motion DSL
multi_text
AToM-style frequency / temporal verifier
```

---

## Case C：Rare Motion

```text
limping asymmetric gait
```

允许：

```text
Retrieval
→ Compiler revise caption / motion prior
```

---

## Case D：Contact

```text
right hand contacts target surface
```

必须：

```text
ConstraintSpec
→ Generation / Verification same target
```

---

## Case E：Whole-Body State

```text
finish in a stable seated pose
```

必须：

```text
BUILD_KEYFRAME
→ KeyframeSpec
→ Hard condition
→ Keyframe verifier
```

---

## Case F：Targeted Repair

初轮：

```text
semantic pass
contact fail
```

后续不应：

```text
重写全部动作
```

而应：

```text
Constraint / Retrieval / Guided Generate
```

针对 contact 修复。

---

# 26. End-to-End Evaluation

最终至少报告：

```text
Task Success Rate

Success@1
Success within N Generation Rounds

Semantic Pass Rate
Event Integrity / Temporal / Frequency
Constraint Satisfaction
Keyframe Satisfaction
Naturalness
Physical Artifact Rate

Average Generation Rounds
Average Tool Calls
GPU Cost / Task
LLM Cost / Task
End-to-End Latency
```

---

# 27. 最重要 Agent 指标

除了最终 Motion 质量，还要看：

```text
Planner Routing Accuracy

Unnecessary Tool Call Rate

Duplicate Retrieval Rate

Repeated Repair Family Rate

Verifier False Accept Rate

Targeted Repair Success Rate

Repair Regression Rate

Budget Exhaustion Rate
```

---

# 28. 建议 Ablation

```text
Frozen GEM

+ Motion Compiler

+ Retrieval

+ Constraint / Keyframe

+ K Candidates

+ Tournament

+ Multi-Verifier

+ Diagnosis / Targeted Repair

Full MotionAgent
```

额外：

```text
Blind Regeneration
vs
Targeted Repair
```

这是最重要的 Agent Ablation 之一。

---

# 29. 主要参考工作与对应位置

| 参考工作 | MotionAgent 中的使用位置 |
|---|---|
| LangGraph | stateful orchestration runtime / conditional routing / checkpoint-resume |
| GEM / GENMO | Frozen base motion generator |
| NEWTON | Planner → Tool → Verifier → Replan 的系统思想 |
| LAMP | Natural language → structured motion program |
| RAPO | generator-aware caption refinement / retrieval grounding |
| TMR | text-motion retrieval / semantic evidence |
| Retrieval-Guided DNO | motion prior + constrained generation / future guided sampling |
| GMD | sparse spatial / keyframe guidance 思想 |
| KeyMotion | keyframe-first / infill 思想 |
| VISTA | candidate tournament + specialized critique + iterative improvement |
| AToM | Integrity / Temporal / Frequency event verification |
| MotionCritic | motion naturalness evidence |
| PP-Motion | perceptual quality 与 physical fidelity 分开 |
| GENMAC | verifier feedback → routed correction / repair specialization |

---

# 30. 开发时应该优先阅读哪些文档

实现顺序：

```text
00_project_overview.md
↓
04_unified_state.md
↓
03_motion_planner.md
↓
05_motion_compiler.md
↓
08_gem_generation_tool.md
↓
09_candidate_tournament.md
↓
10_multi_verifier.md
↓
11_diagnosis_repair.md
↓
06_retrieval_tool.md
↓
07_constraint_compiler.md
↓
07b_keyframe_tool.md
↓
12_system_integration.md
↓
industrial_harness_design.md
```

如果开始写代码：

> **先把 Schema、State、Request Builder、Guard 和最小闭环实现出来，再逐渐增加复杂 evaluator / retrieval / guidance。**
