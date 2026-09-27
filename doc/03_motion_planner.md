# MotionAgent for GEM：模块实现指南

> 本文件从总实现指南中拆分，对应第 3 单元。内容保持模块边界独立，其输入输出接口需与相邻模块的 Schema 保持一致。

# 3. 顶层 Agent：Motion Planner

本节定义 Motion Planner 的**实现规范**。后续实现时，Planner 的输入、输出、Action Space、状态约束与调用规则均以本节为准。

Motion Planner 采用：

> **Bounded-ReAct Controller：基于当前 State，每轮只选择一个结构化 Next Action。**

Planner 不一次性生成完整执行流程，也不直接执行底层 Motion 操作。每次 Tool / Subgraph 执行结束后，结果写回 State，再由 Planner 决定下一步。

V1 的执行 runtime 显式采用 **LangGraph StateGraph**。Planner 仍然是唯一的 Autonomous Agent；LangGraph 只负责把已经验证过的 `PlannerDecision` 路由到对应 node，并负责固定 subgraph、checkpoint / resume 与 graph lifecycle。

```text
Planner policy ≠ LangGraph routing

Planner 决定“下一步做什么”
LangGraph 决定“按照已定义 topology 执行到哪个 node”
```

---

## 3.1 Planner 的职责边界

Planner 只负责：

```text
读取当前 MotionAgentState
→ 判断当前缺少什么
→ 选择 ONE next action
→ 给出该 action 所需的结构化参数
```

Planner 不负责：

```text
直接生成 SMPL
直接运行 GEM
直接计算 IK / FK
直接搜索 Motion Database
直接计算 Constraint Error
直接执行 Tournament
直接执行 Verify
直接输出最终 Repair 结果
```

其中：

```text
GEM × K
→ Candidate Tournament
→ Multi-Verifier
→ Structured Diagnosis
```

属于 `GENERATE` 之后由系统自动执行的固定 Subgraph。

Verifier / Diagnosis 的结果写回 State 后，再由 Planner 决定下一轮应该：

```text
修改 Motion Plan
检索新的 Reference
增加 / 修改 Constraint
增加 / 修改 Keyframe
重新 Generate
或者 ACCEPT
```

因此，`VERIFY`、`REPAIR`、`REGENERATE` 不作为 Planner 的独立 Action。

---

## 3.2 Planner Action Space

第一版 Planner 只允许输出以下 7 种 Action：

```text
COMPILE_MOTION
RETRIEVE_REFERENCE
BUILD_CONSTRAINT
BUILD_KEYFRAME
GENERATE
ACCEPT
STOP_FAILED
```

每轮只允许选择一个 Action。

### 3.2.1 `COMPILE_MOTION`

用于：

```text
第一次将 User Request 转换成 Motion Plan
修改错误的动作语义
修改 Timeline
修改 Frequency / Event 描述
修改 GEM Caption
```

典型输入场景：

```text
plan == null
semantic failure
temporal failure
frequency failure
prompt / caption 不适合 GEM
```

输出格式：

```json
{
  "action": "COMPILE_MOTION",
  "reason_code": "INITIAL_COMPILE",
  "target_segments": null,
  "reason_summary": "The user request has not yet been converted into a structured motion plan.",
  "payload": {
    "mode": "initial",
    "focus": [
      "semantic",
      "timeline",
      "gem_caption"
    ]
  }
}
```

局部修改示例：

```json
{
  "action": "COMPILE_MOTION",
  "reason_code": "TEMPORAL_FAILURE",
  "target_segments": [1],
  "reason_summary": "The event order is incorrect while the remaining segments already pass verification.",
  "payload": {
    "mode": "revise",
    "focus": [
      "timeline",
      "frequency"
    ]
  }
}
```

允许的 `mode`：

```text
initial
revise
```

允许的 `focus`：

```text
semantic
timeline
frequency
body_part
style
gem_caption
```

---

### 3.2.2 `RETRIEVE_REFERENCE`

用于需要外部 Motion Knowledge 的情况：

```text
rare motion
rare motion style
模型多次无法理解的动作
需要 reference pose
需要 reference trajectory
需要 reference motion prior
```

输出格式：

```json
{
  "action": "RETRIEVE_REFERENCE",
  "reason_code": "RARE_MOTION",
  "target_segments": [0],
  "reason_summary": "The requested limping style is uncommon and a reference motion may provide a stronger prior.",
  "payload": {
    "query": "injured asymmetric limping gait",
    "retrieval_type": "motion",
    "purpose": "motion_prior",
    "top_k": 5
  }
}
```

允许的 `retrieval_type`：

```text
caption
motion
pose
trajectory
```

允许的 `purpose`：

```text
prompt_grounding
motion_prior
keyframe_source
constraint_source
```

约束：

```text
同一 State 下，不重复执行相同 query + target_segment 的 Retrieval。
如果已有有效 Reference，默认保留，不因重新 Generate 而删除。
```

---

### 3.2.3 `BUILD_CONSTRAINT`

只用于**可度量的运动 / 几何要求**。

第一版支持：

```text
joint_target
body_part_pose
root_trajectory
contact
fixed_joint
```

不应把抽象 Style 强行转换成数值 Constraint。

例如：

```text
"confident"
"happy"
"drunk-like"
```

默认仍属于 Text / Retrieval 问题，除非已有可用的明确运动参考或数值定义。

输出格式：

```json
{
  "action": "BUILD_CONSTRAINT",
  "reason_code": "CONTACT_REQUIREMENT",
  "target_segments": [1],
  "reason_summary": "The request contains an explicit right-wrist contact requirement that should be enforced numerically.",
  "payload": {
    "mode": "add",
    "constraint_type": "contact",
    "body_part": "right_wrist",
    "time_range": [2.8, 3.3],
    "target": "table_surface",
    "strength": "hard"
  }
}
```

允许的 `mode`：

```text
add
replace
remove
```

允许的 `strength`：

```text
hard
soft
```

后续 Constraint Compiler 负责把该结构转换成：

```text
IK / Pose
Root Trajectory
observed_motion_3d
motion_mask_3d
或 Guided Reward
```

Planner 不直接生成这些 Tensor。

---

### 3.2.4 `BUILD_KEYFRAME`

用于某个时间点必须出现较完整的 Whole-Body State。

典型情况：

```text
第 3 秒必须处于深蹲状态
最后必须稳定坐下
动作边界需要明确完整姿态
```

不应用于简单局部 Joint Target；局部关节约束优先使用 `BUILD_CONSTRAINT`。

输出格式：

```json
{
  "action": "BUILD_KEYFRAME",
  "reason_code": "WHOLE_BODY_STATE_REQUIRED",
  "target_segments": [2],
  "reason_summary": "The final seated state requires a stable whole-body pose rather than a single local joint constraint.",
  "payload": {
    "mode": "add",
    "time": 5.8,
    "description": "a stable seated whole-body pose",
    "source_preference": "retrieval"
  }
}
```

允许的 `source_preference`：

```text
retrieval
ik
generation
auto
```

---

### 3.2.5 `GENERATE`

所有 Motion Generation 均统一通过 `GENERATE`。

不再单独定义：

```text
GENERATE_GUIDED
REGENERATE
```

使用 `strategy` 和 `scope` 参数区分。

普通生成：

```json
{
  "action": "GENERATE",
  "reason_code": "CONDITIONS_READY",
  "target_segments": null,
  "reason_summary": "The current motion plan and conditions are sufficient for a first generation attempt.",
  "payload": {
    "strategy": "normal",
    "scope": "full",
    "num_candidates": 4,
    "reward_targets": []
  }
}
```

Guided Generation：

```json
{
  "action": "GENERATE",
  "reason_code": "PERSISTENT_CONSTRAINT_FAILURE",
  "target_segments": [1],
  "reason_summary": "The same measurable contact objective has failed under normal sampling and should use guided generation.",
  "payload": {
    "strategy": "guided",
    "scope": "segment",
    "num_candidates": 4,
    "reward_targets": [
      "contact",
      "semantic"
    ]
  }
}
```

允许的 `strategy`：

```text
normal
guided
```

允许的 `scope`：

```text
full
segment
```

第一版执行原则：

```text
首次生成优先使用 normal。
只有存在明确 Soft Constraint / Reward，或普通生成已持续失败时，再使用 guided。
```

`guided` 后续可接：

```text
ReAlign-style guidance
DNO-style optimization
其他 reward-guided sampling
```

Planner 只选择策略，不负责具体 diffusion optimization。

---

### 3.2.6 `ACCEPT`

只有当前结果满足系统验收条件时才允许输出。

格式：

```json
{
  "action": "ACCEPT",
  "reason_code": "ALL_REQUIRED_CHECKS_PASSED",
  "target_segments": null,
  "reason_summary": "All required semantic, temporal, motion-quality and constraint checks have passed.",
  "payload": {}
}
```

代码层必须额外检查：

```text
latest_verification != null
overall_pass == true
critical_failures == []
```

不满足上述条件时，`ACCEPT` 为非法 Action。

---

### 3.2.7 `STOP_FAILED`

用于：

```text
Generation Budget 耗尽
Iteration Budget 耗尽
关键 Tool 持续失败且无有效替代路径
当前任务存在不可恢复状态
```

格式：

```json
{
  "action": "STOP_FAILED",
  "reason_code": "BUDGET_EXHAUSTED",
  "target_segments": null,
  "reason_summary": "The remaining generation budget is zero and the current critical failure is unresolved.",
  "payload": {}
}
```

---

## 3.3 Planner 输入：`PlannerContext`

Planner 不直接读取完整 Raw History、SMPL Tensor、151-D Motion Tensor 或完整 Tool Log。

每一轮只读取压缩后的结构化 `PlannerContext`。

建议结构：

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

### `task`

```json
{
  "original_request": "A person limps forward, supports the body with the right hand on a table, then sits down.",
  "target_duration": 6.0,
  "fps": 30
}
```

### `plan`

```json
{
  "status": "ready",
  "segments": [
    {
      "id": 0,
      "time": [0.0, 3.0],
      "action": "walk",
      "style": "limping"
    },
    {
      "id": 1,
      "time": [3.0, 4.0],
      "action": "reach_and_contact"
    },
    {
      "id": 2,
      "time": [4.0, 6.0],
      "action": "sit"
    }
  ],
  "unresolved": []
}
```

### `conditions`

Planner 只看 Condition Summary，不看底层 Tensor：

```json
{
  "text": {
    "ready": true
  },
  "retrieval": [
    {
      "id": "ret_01",
      "segment": 0,
      "type": "motion",
      "purpose": "motion_prior",
      "status": "active"
    }
  ],
  "constraints": [
    {
      "id": "c_01",
      "segment": 1,
      "type": "contact",
      "body_part": "right_wrist",
      "status": "active"
    }
  ],
  "keyframes": []
}
```

### `latest_generation`

```json
{
  "generation_id": "gen_03",
  "strategy": "normal",
  "scope": "full",
  "candidate_count": 4,
  "champion_id": "candidate_2"
}
```

### `latest_verification`

Planner 只读取第 10 单元 `VerificationReport` 的压缩 Summary：

```json
{
  "status": "complete",
  "overall_pass": false,
  "passed_check_ids": [
    "semantic_core",
    "naturalness_core"
  ],
  "failed_check_ids": [
    "contact_constraint_01"
  ],
  "warning_check_ids": [],
  "uncertain_check_ids": [],
  "critical_failures": [
    "CONSTRAINT_CONTACT_ERROR"
  ],
  "affected_segments": [1],
  "diagnostic_codes": [
    "CONSTRAINT_CONTACT_ERROR"
  ]
}
```

### `diagnosis`

Planner 读取第 11 单元 `DiagnosisResult` 的压缩 Summary，而不是完整 Root-Cause Trace。第 11 单元的正式候选修复对象是 `RepairProposal[]`；PlannerContextBuilder 只把这些 Proposal 压缩成可比较的 proposal summary，真正选择后再按 `proposal_id` 加载完整 payload：

```json
{
  "status": "diagnosed",
  "primary_failures": [
    "CONSTRAINT_CONTACT_ERROR"
  ],
  "root_causes": [
    "CONSTRAINT_TOO_WEAK"
  ],
  "target_segments": [1],
  "proposal_summaries": [
    {
      "proposal_id": "repair_07",
      "action": "BUILD_CONSTRAINT",
      "reason_code": "CONSTRAINT_FAILURE",
      "target_segments": [1],
      "confidence": 0.91
    },
    {
      "proposal_id": "repair_08",
      "action": "RETRIEVE_REFERENCE",
      "reason_code": "MISSING_REFERENCE",
      "target_segments": [1],
      "confidence": 0.63
    }
  ],
  "terminal_hint": null
}
```

### `history`

只保留执行轨迹摘要：

```json
{
  "recent_actions": [
    {
      "step": 1,
      "action": "COMPILE_MOTION",
      "result": "success"
    },
    {
      "step": 2,
      "action": "RETRIEVE_REFERENCE",
      "target_segments": [0],
      "result": "success"
    },
    {
      "step": 3,
      "action": "GENERATE",
      "result": "constraint_failure"
    }
  ],
  "repeated_actions": {
    "RETRIEVE_REFERENCE:segment_0": 1,
    "GENERATE": 1
  }
}
```

### `budget`

```json
{
  "iterations_left": 8,
  "generations_left": 3,
  "retrieval_calls_left": 3,
  "guided_generations_left": 1
}
```

---

## 3.4 Planner 输出：`PlannerDecision`

Planner 只能输出一个符合严格 Schema 的对象。

```python
class PlannerDecision(BaseModel):
    action: Literal[
        "COMPILE_MOTION",
        "RETRIEVE_REFERENCE",
        "BUILD_CONSTRAINT",
        "BUILD_KEYFRAME",
        "GENERATE",
        "ACCEPT",
        "STOP_FAILED",
    ]

    reason_code: str

    target_segments: list[int] | None

    reason_summary: str

    payload: dict
```

要求：

```text
每轮只输出一个 Action。
不输出额外自然语言。
不输出 Chain-of-Thought。
reason_summary 只写一句简短、可记录的决策理由。
payload 必须符合对应 Action Schema。
```

---

## 3.5 Reason Code

第一版 Reason Code 固定为：

```text
INITIAL_COMPILE

SEMANTIC_UNCLEAR
TEMPORAL_UNCLEAR
RARE_MOTION
MISSING_REFERENCE

GEOMETRIC_REQUIREMENT
CONTACT_REQUIREMENT
TRAJECTORY_REQUIREMENT
WHOLE_BODY_STATE_REQUIRED

CONDITIONS_READY

SEMANTIC_FAILURE
TEMPORAL_FAILURE
CONSTRAINT_FAILURE
NATURALNESS_FAILURE
PERSISTENT_CONSTRAINT_FAILURE
PERSISTENT_FAILURE

ALL_REQUIRED_CHECKS_PASSED

BUDGET_EXHAUSTED
UNRECOVERABLE_TOOL_FAILURE
```

Reason Code 用于：

```text
调试 Planner
统计 Routing Accuracy
分析 Tool 使用分布
构造后续 Planner RL trajectory
```

禁止 Planner 自由创造新的 Reason Code。

---

## 3.6 Planner System Prompt

第一版 System Prompt 固定采用如下骨架：

```text
You are MotionPlanner, the top-level controller of MotionAgent,
an agentic human-motion generation system built on GEM.

Your job is NOT to generate motion yourself.
Your job is to inspect the current MotionAgentState and select
exactly ONE next action that most efficiently moves the task
toward a verified final motion.

You may choose only from:

COMPILE_MOTION
RETRIEVE_REFERENCE
BUILD_CONSTRAINT
BUILD_KEYFRAME
GENERATE
ACCEPT
STOP_FAILED

GENERAL OBJECTIVE

Produce a motion that satisfies the user's original instruction
while minimizing unnecessary tool calls, generation rounds,
and destructive changes to conditions that already work.

DECISION PRINCIPLES

1. Preserve the user's original intent.

2. Preserve conditions that already passed verification.
   Do not rebuild the whole specification after a local failure.

3. Use COMPILE_MOTION for:
   - initial semantic and temporal decomposition;
   - repairing incorrect action semantics;
   - repairing timeline, frequency, or GEM-caption problems.

4. Use RETRIEVE_REFERENCE only when external motion knowledge is useful:
   - rare or unfamiliar actions/styles;
   - a reference pose or trajectory is needed;
   - repeated semantic failures suggest GEM lacks a useful motion prior.
   Do not repeat the same retrieval without new evidence.

5. Use BUILD_CONSTRAINT for measurable requirements:
   - joint target;
   - root trajectory;
   - contact;
   - fixed body part;
   - local body-part pose.
   Do not convert purely semantic or stylistic requirements into
   geometric constraints unnecessarily.

6. Use BUILD_KEYFRAME when an important whole-body state must occur
   at a specific time or motion boundary.
   Do not use a whole-body keyframe for a simple local joint constraint.

7. Use GENERATE only when the current Motion Specification is sufficiently ready.

8. Prefer strategy="normal" for the first generation when possible.

9. Use strategy="guided" only when:
   - a compatible soft constraint or reward is important; or
   - normal generation has already failed on a persistent measurable objective.

10. After GENERATE, candidate sampling, tournament selection,
    verification, and diagnosis are executed automatically by the system.
    Their results will appear in the next MotionAgentState.

11. When verification fails, modify only the failed component whenever possible:
      semantic / caption failure -> COMPILE_MOTION
      temporal / frequency failure -> COMPILE_MOTION
      rare-style / missing-prior failure -> RETRIEVE_REFERENCE
      joint / contact / trajectory failure -> BUILD_CONSTRAINT
      whole-body-state failure -> BUILD_KEYFRAME
      persistent compatible failure -> consider guided GENERATE

12. ACCEPT only when the state reports overall_pass=true
    and no critical failure remains.

13. Never repeat the same action with essentially identical arguments
    if the previous attempt did not change the relevant state.

14. Preserve successful retrieved references, constraints and keyframes
    unless new evidence shows they are incorrect.

15. If the available budget is exhausted or no valid recovery action exists,
    choose STOP_FAILED.

OUTPUT RULE

Return exactly one PlannerDecision object.
Do not output prose outside the structured object.
Do not output hidden reasoning or chain-of-thought.
Use one fixed reason_code and one short reason_summary only.
```

---

### 3.6.1 Cross-Segment Continuity Routing

Planner 不负责逐帧决定 root orientation，也不新增 Action。

对于：

```text
UNCOMMANDED_HEADING_DRIFT
```

使用现有 Action Space 路由：

```text
如果 MotionSpecification 缺少 / 错误表达 continuity policy
→ COMPILE_MOTION

如果 continuity policy 已正确存在，但 Candidate 仍持续出现可测量 heading drift
→ BUILD_CONSTRAINT
  （使用 root_trajectory 的 orientation-only / fixed-heading soft control）

如果只是单次 sampling variance，且 specification / control 已正确
→ GENERATE
```

Planner 必须保留已经通过验证的动作段和条件，只修改导致 heading drift 的局部组件。

不要把所有自然身体旋转都判为失败。只有在：

```text
没有显式 turn / rotate / spin / facing change
+
MotionSpecification 要求 inherit_previous / explicit heading
+
Verifier 报告超出配置阈值的 heading drift
```

时，才进入 continuity repair。

该逻辑复用现有 Reason Code：

```text
SEMANTIC_FAILURE / CONSTRAINT_FAILURE / NATURALNESS_FAILURE
```

V1 不为此新增 Planner Action 或强制新增 Reason Code，避免破坏 Phase 2 已冻结的 Planner contract。

## 3.7 Hard State Guards

Planner Prompt 只负责行为策略。

所有关键流程约束必须同时在代码层执行，不依赖 LLM 自觉遵守。

建议实现：

```python
def validate_action(state, decision):

    if state.motion_plan is None:
        assert decision.action == "COMPILE_MOTION"

    if decision.action == "GENERATE":
        assert state.motion_plan is not None
        assert state.motion_plan.status == "ready"
        assert state.conditions.text.ready
        assert state.budget.generations_left > 0

    if decision.action == "ACCEPT":
        assert state.latest_verification is not None
        assert state.latest_verification.overall_pass is True
        assert len(state.critical_failures) == 0

    if decision.action == "RETRIEVE_REFERENCE":
        assert not duplicated_retrieval(state, decision)

    if decision.action == "GENERATE":
        if decision.payload["strategy"] == "guided":
            assert guided_generation_is_available(state)
            assert guided_generation_is_justified(state)

    if decision.action == "STOP_FAILED":
        assert budget_exhausted(state) or unrecoverable_failure(state)
```

必须写死的流程规则：

```text
No Motion Plan
→ 禁止 GENERATE

Fresh Generation
→ 自动执行 Tournament + Verify + Diagnosis

No Verification
→ 禁止 ACCEPT

Verification Pass
→ 才允许 ACCEPT

Generation Budget == 0
→ 禁止 GENERATE

同一 Retrieval Query + 同一 Segment + State 未变化
→ 禁止重复 Retrieval

同一 Failure 连续两轮使用同一种 Repair Family 且无改善
→ 下一轮必须更换 Repair Family 或 STOP_FAILED

已经通过验证的 Condition
→ 默认保留，不允许无理由删除
```

---

## 3.8 Automatic Post-Generation Subgraph

Planner 输出：

```text
GENERATE
```

之后 LangGraph 中固定的 `post_generation` subgraph 自动执行：

```text
GenerationRequest
      ↓
GEM × K Candidates
      ↓
VISTA-style Tournament
      ↓
Champion Motion
      ↓
Multi-Verifier
      ↓
GENMAC-style Structured Diagnosis
      ↓
Update MotionAgentState
      ↓
Return to Planner
```

Planner 不单独调用：

```text
Tournament
Verifier
Diagnosis
```

这三个模块属于系统环境产生 Observation 的过程。

---

## 3.9 Tool Skill Cards

不要把所有 Tool 的详细使用规则全部塞进 Planner System Prompt。

建议维护：

```text
planner_skills/
│
├── compile_motion.md
├── retrieval.md
├── constraint.md
├── keyframe.md
└── generation.md
```

Planner 默认只看到：

```text
Tool Name
Short Description
Input Schema
```

需要某个 Tool 时，再读取其 Skill Card。

例如 `constraint.md`：

```text
WHEN TO USE

Use when the request contains a measurable spatial,
kinematic, trajectory, contact, or body-part requirement.

DO NOT USE

Do not encode abstract style such as:
"confident", "happy", "drunk-like"
as a geometric constraint unless an explicit measurable
formulation or reference motion exists.

CHOOSING HARD VS SOFT

Known state or exact target
→ hard constraint

Many possible poses can satisfy the requirement
→ soft reward / guided generation
```

Skill Card 后续分别在各模块实现阶段补充。

`[TODO-PLANNER-SKILLS-01]`

---

## 3.10 Planner 调用接口

Planner 的 domain 接口保持简单，并与 LangGraph 解耦：

```python
def planner_step(state: MotionAgentState) -> PlannerDecision:

    context = build_planner_context(state)

    decision = planner_llm.generate(
        system_prompt=MOTION_PLANNER_PROMPT,
        input=context,
        output_schema=PlannerDecision,
    )

    validate_action(
        state=state,
        decision=decision,
    )

    return decision
```

LangGraph node 只负责 runtime adapter：

```python
async def planner_node(graph_state: MotionGraphState):
    state = state_store.load_latest(graph_state.run_id)

    legal_actions = get_legal_actions(state)
    decision = await planner.step(
        build_planner_context(state, legal_actions)
    )

    validate_planner_decision(
        decision=decision,
        state=state,
        legal_actions=legal_actions,
    )

    return {"planner_decision": decision}
```

之后由 deterministic conditional routing：

```text
planner
  ├─ COMPILE_MOTION      → compile_motion → planner
  ├─ RETRIEVE_REFERENCE  → retrieval      → planner
  ├─ BUILD_CONSTRAINT    → constraint     → planner
  ├─ BUILD_KEYFRAME      → keyframe       → planner
  ├─ GENERATE            → generation → tournament → verifier → diagnosis → planner
  ├─ ACCEPT              → accept → END
  └─ STOP_FAILED         → stop_failed → END
```

`route_planner_action()` 只能映射固定 Action Enum；不能让 LLM 动态产生 node 名称。Hard Guard 在进入 conditional edge 前执行。

## 3.11 Planner 实现原则

实现时必须遵守：

```text
一个 Planner 拥有全局决策权。

每轮只选一个 Next Action。

固定流程不做 Agent 决策。

LangGraph Node 不等于 Agent；除 Planner 外的 node 默认只是 Tool / Workflow adapter。

Tool 不拥有全局控制权。

Verifier 产生 Observation，不直接控制整个系统。

Repair 通过下一轮 Action 完成，而不是使用一个泛化 REPAIR Action。

优先局部修改，不在局部失败后重建整个 Motion Specification。

保留已经验证有效的 Condition。

所有 Action 均有 Budget 与 Hard Guard。

Planner 的决策轨迹必须被完整记录，
为后续 Planner RL 保留：
state → action → observation → reward
数据。
```

`[TODO-PLANNER-01]`

后续在实际实现阶段继续补充：

```text
Pydantic Schema
Tool Skill Card
LLM Provider
Prompt Versioning
Action Validation
Budget Config
Trajectory Logging
Planner RL Dataset
```

---
