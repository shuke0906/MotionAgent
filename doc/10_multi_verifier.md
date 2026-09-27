# MotionAgent for GEM：模块实现指南

> 本文件对应第 10 单元 Multi-Verifier。  
> 本模块必须与 `03_motion_planner.md` 的 Automatic Post-Generation Subgraph、`09_candidate_tournament.md` 的 Champion 输出、`07_constraint_compiler.md` 的 `VerificationSpec`、`07b_keyframe_tool.md` 的 `KeyframeVerificationSpec`、以及后续 `11_diagnosis_repair.md` 的 Diagnosis 输入保持接口一致。
>
> 本模块只负责：
>
> **对 Tournament 选出的 Champion 做绝对验证，并输出结构化 Failure Evidence。**
>
> 它不负责选择 Candidate，也不负责决定 Repair Action。

# 10. Multi-Verifier

本节定义 Multi-Verifier 的**实现规范**。

Multi-Verifier 的职责是：

> **根据当前 Motion Specification、Constraint / Keyframe Requirements 和 Generation Context，自动选择需要执行的验证方向，对 Champion Motion 做绝对 pass/fail 验收，并输出可定位到 Segment / Event / Body Part / Constraint 的结构化 Verification Report。**

Multi-Verifier 是：

```text
Automatic Post-Generation Subgraph
+
Task-Aware Verification Planner
+
Deterministic Metric Tools
+
Learned Motion Critics
+
Selective MLLM Critics
```

它不是 Planner Action，不拥有全局控制权。

### LangGraph 执行边界

V1 中本模块是 `post_generation` LangGraph subgraph 的固定 `verifier` node。它由 graph edge 自动进入，并固定流向 `diagnosis`；这条 edge 不由 Planner/LLM 每轮重新选择。模块内部算法和 typed service interface 保持 framework-agnostic。

---

# 10.1 在整体 Planner 中的定位

根据第 3 单元：

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
       Champion
          │
          ▼
 ┌──────────────────┐
 │  Multi-Verifier  │
 └──────────────────┘
          │
          ▼
 VerificationReport
          │
          ▼
 Structured Diagnosis
          │
          ▼
        Planner
```

因此 Multi-Verifier：

```text
不出现在 Planner Action Space
```

也不返回：

```text
COMPILE_MOTION
BUILD_CONSTRAINT
GENERATE
ACCEPT
```

这种动作。

它只回答：

```text
Champion 是否通过？

如果没有通过：
哪里失败？
实际观察到了什么？
要求是什么？
误差是多少？
证据是什么？
```

之后由：

```text
Diagnosis
```

将 Verification Failure 转成 Planner 可用的 Repair Evidence。

---

# 10.2 与第 9 单元 Tournament 的严格区分

第 9 单元的正式上游产物是：

```text
TournamentResult
```

`VerificationRequestBuilder` 从其中读取：

```text
champion_candidate_id
tournament_id
selection evidence summary（仅作辅助）
```

再结合 Current State 中的 MotionSpecification、Constraint VerificationSpec、KeyframeVerificationSpec 和 Generation Context 构造 `VerificationRequest`。Multi-Verifier 不直接把整个 Tournament transcript 当作判断依据。


必须固定：

```text
Tournament
= Relative Selection

Verifier
= Absolute Acceptance
```

Tournament 只回答：

> 当前 Candidate Set 中谁更好？

Verifier 回答：

> Champion 是否达到任务要求？

例如：

```text
Candidate A:
contact error = 0.20 m

Candidate B:
contact error = 0.15 m
```

Tournament：

```text
B wins
```

如果 Requirement：

```text
contact error <= 0.05 m
```

Verifier：

```text
B = FAIL
```

因此：

> **Champion ≠ Accepted Motion**

Planner 只有在：

```text
VerificationReport.overall_pass == true
```

时，才允许下一轮输出：

```text
ACCEPT
```

---

# 10.3 参考工作的具体使用方式

本模块主要参考：

```text
VISTA
AToM
MotionCritic
TMR
PP-Motion
```

但不原样照搬。

---

## 10.3.1 VISTA：Specialized Critics

VISTA 在 Pairwise Selection 之后，不再用一个通用 Judge 做所有检查，而是使用多个 Specialized Critics 分别检查：

```text
Visual
Audio
Context
```

说明：

> 多维生成任务不适合使用单一总分评价器。

MotionAgent 将这一思想映射为：

```text
Semantic
Event
Naturalness
Constraint
Keyframe
Physical
Preservation
```

但这些 Verifier 不一定全部执行。

由：

```text
VerifierPlanBuilder
```

根据当前任务自动选择。

---

## 10.3.2 AToM：Event-Level Verification

AToM 将复杂 Text-to-Motion Alignment 明确拆成：

```text
Integrity
Temporal Relationship
Frequency
```

并使用：

```text
Rendered Motion
+
Task-specific Instructions
+
Task-specific Scoring Rules
```

让视觉语言模型判断。

本项目直接采用这种拆分。

但不把 AToM 的训练 / RL 部分放入 Verifier。

这里只使用其：

> **Event-Level Evaluation Paradigm**

---

## 10.3.3 MotionCritic：Human-Perceptual Naturalness

MotionCritic 专门学习：

```text
naturalness
smoothness
plausibility
artifact-free motion
```

与人类偏好对齐。

因此 V1：

```text
Naturalness Verifier
→ MotionCritic
```

作为主要 learned motion-quality signal。

---

## 10.3.4 TMR：Semantic Alignment

TMR 将：

```text
Text
Motion
```

编码到共同 latent space。

本项目使用：

```text
TMR similarity
```

作为：

```text
Semantic Alignment
```

的 cheap structured signal。

但 TMR 不单独负责：

```text
complex event order
repetition count
exact numerical constraint
```

这些分别交给其他 Verifier。

---

## 10.3.5 PP-Motion / Physics Evaluation

PP-Motion 指出：

```text
human-perceived naturalness
```

与：

```text
physical feasibility
```

不是同一个问题。

因此 V1 不用 MotionCritic 代替全部 Physical Verification。

V1 额外实现一组容易落地的 deterministic physical metrics：

```text
foot skating
ground penetration
joint / root smoothness
contact stability
```

更复杂的：

```text
physics simulator
dynamic feasibility
balance simulation
PP-Motion learned metric
```

放入 Phase II。

这样 V1 不需要增加 MuJoCo / IsaacGym 等重型依赖。

---

# 10.4 核心设计：Verifier 不是“全部全跑”

本模块不能写成：

```text
Semantic Critic
Event Critic
MotionCritic
Constraint Checker
Physical Metric
全部无条件执行
```

因为不同 User Task 需要的验收方向不同。

例如：

```text
walk forward
```

不需要：

```text
Frequency Verifier
```

而：

```text
wave the right hand three times
```

必须需要：

```text
Frequency Verifier
```

因此必须增加：

> **VerifierPlanBuilder**

---

# 10.5 `VerifierPlanBuilder`

职责：

> 根据已经结构化的任务要求，生成本次 Champion 必须执行的 Verification Check List。

它不是 LLM Agent。

优先使用：

```text
MotionSpecification
Constraint VerificationSpec
Keyframe Requirement
Generation Metadata
```

做 deterministic rule-based selection。

接口：

```python
def build_verification_plan(
    request: VerificationRequest,
) -> VerificationPlan:
    ...
```

---

# 10.6 Verifier 方向

第一版支持：

```text
technical

semantic

event_integrity
event_temporal
event_frequency

naturalness

constraint

keyframe

physical

preservation
```

其中：

```text
technical
```

是系统前置检查，不属于 Motion Quality 评分维度。

---

# 10.7 哪些 Verifier 必须运行

## Always

任何有效 Champion 都必须运行：

```text
technical

semantic

naturalness
```

---

## Conditional

根据任务按需增加：

```text
event_integrity
event_temporal
event_frequency
constraint
keyframe
physical
preservation
```

---

# 10.8 Verifier Direction Selection Rules

推荐固定成以下规则。

| Task Feature | Verifier |
|---|---|
| 任意 Motion | `technical` |
| 任意文本生成任务 | `semantic` |
| 任意 Motion | `naturalness` |
| 2 个及以上 required event | `event_integrity` |
| 存在显式/结构化 event order | `event_temporal` |
| `repetition != None` | `event_frequency` |
| Active `VerificationSpec[]` | `constraint` |
| Active Keyframe | `keyframe` |
| locomotion / foot-contact / explicit physical requirement | `physical` required |
| 普通非 locomotion Motion | `physical` optional / warning profile |
| `generation_scope == "segment"` | `preservation` |

---

## 10.8.1 Heading Continuity Verifier 选择规则

如果 `MotionSpecification` 中任一 Segment 存在：

```text
heading_continuity.mode = inherit_previous
heading_continuity.mode = explicit
heading_continuity.mode = follow_trajectory
```

则启用：

```text
heading_continuity
```

该检查适用于：

```text
full generation
segment regeneration
```

不依赖 `generation_scope == "segment"`。

因此 Verifier 方向集合扩展为：

```text
technical
semantic
naturalness
event_*
constraint
keyframe
physical
preservation
heading_continuity
```

但只有当前 MotionSpecification 有对应 expectation 时才运行。

# 10.9 Event Verifier 的选择

Motion Compiler 中已有：

```text
MotionSegment
repetition
transition
timeline
secondary_actions
```

因此：

### Integrity

如果：

```text
required_event_count >= 2
```

启用。

---

### Temporal

如果：

```text
segment_count >= 2
```

且任务存在：

```text
then
before
after
followed by
first / second / finally
```

等明确顺序，

或 Motion DSL 已明确：

```text
ordered event list
```

则启用。

---

### Frequency

只要：

```text
segment.repetition != None
```

就启用。

---

# 10.10 Constraint Verifier 的选择

如果：

```text
state.active_constraints
```

中存在：

```text
VerificationSpec
```

则每一个 Spec 都自动变成一个：

```text
constraint verification check
```

例如：

```text
joint target

root trajectory

contact

fixed joint

body-part pose
```

无需 MLLM 再判断：

```text
“这是不是 Constraint？”
```

---

# 10.11 Keyframe Verifier 的选择

如果存在：

```text
active_keyframes
```

每个 Keyframe 生成一个：

```text
keyframe check
```

检查：

```text
目标时间
目标 whole-body state
```

是否真正出现。

---

# 10.12 Preservation Verifier 的选择

只在：

```text
GenerationRequest.scope == "segment"
```

时运行。

比较：

```text
new Champion
vs
previous Champion
```

检查：

```text
非目标 Segment 是否被保留
Segment Boundary 是否连续
Body Shape 是否保持一致
```

---

## 10.12.1 Cross-Segment Heading Continuity Check

`heading_continuity` 是 deterministic-first verifier。

输入：

```text
MotionSpecification.heading_continuity
Champion.smpl_global
segment frame ranges
Adapter / coordinate context
```

核心步骤：

```text
root orientation
→ canonical body forward vector
→ ground-plane projection
→ compare with anchor heading
```

对于：

```text
inherit_previous
```

anchor 使用前一 Segment 结束附近稳定窗口的 heading，而不是固定假设 world +Z。

输出建议：

```python
class HeadingContinuityEvidence(BaseModel):
    segment_id: int
    anchor_segment_id: int | None
    mean_drift_deg: float
    p95_drift_deg: float
    max_drift_deg: float
    uncommanded_turn_duration_ratio: float
    explicit_turn_exempted: bool
```

Failure Code：

```text
UNCOMMANDED_HEADING_DRIFT
```

阈值规则：

```text
不要由 LLM 设定；
不要直接照搬论文中的任意角度；
使用 config + 小型人工 calibration set 确定。
```

如果用户显式要求 turn / rotate / spin，则对应 transition window 必须 exempt，避免把正确转向误判成失败。

如果该检查是 required 且超阈值：

```text
overall_pass = false
```

# 10.13 Physical Verifier 的选择

如果动作包含：

```text
walk
run
jump
step
stand
sit-to-stand
locomotion
foot contact
fixed foot
ground interaction
```

启用：

```text
physical required profile
```

其他 Motion：

```text
physical warning profile
```

第一版不要求对所有 Motion 执行：

```text
full rigid-body physics simulation
```

---

# 10.14 `VerificationRequest`

主接口：

```python
async def verify_motion(
    request: VerificationRequest,
) -> VerificationReport:
    ...
```

Schema：

```python
class VerificationRequest(BaseModel):

    verification_id: str

    champion_candidate_id: str

    tournament_id: str

    original_request: str

    motion_spec_id: str

    constraint_verification_specs: list[str]

    active_keyframe_ids: list[str]

    generation_id: str

    generation_scope: Literal[
        "full",
        "segment",
    ]

    target_segments: list[int] | None

    previous_candidate_id: str | None

    verifier_policy: VerifierPolicy

    threshold_profile: str

    timeout_s: float | None
```

---

# 10.15 `VerifierPolicy`

```python
class VerifierPolicy(BaseModel):

    enable_mllm_semantic: bool = True

    enable_atom_event_verifier: bool = True

    enable_motioncritic: bool = True

    enable_tmr: bool = True

    enable_physical_metrics: bool = True

    enable_optional_checks: bool = True

    collect_all_failures: bool = True
```

默认：

```text
collect_all_failures = true
```

原因：

> Diagnosis 需要完整 Failure Picture。

只有：

```text
technical invalid
```

这种无法继续分析的情况才 Early Stop。

---

# 10.16 `VerificationPlan`

```python
class VerificationPlan(BaseModel):

    verification_id: str

    champion_candidate_id: str

    checks: list[VerifierCheck]

    required_check_ids: list[str]

    optional_check_ids: list[str]

    render_requirements: list[RenderRequirement]

    evidence_fingerprint: str
```

---

# 10.17 `VerifierCheck`

```python
class VerifierCheck(BaseModel):

    check_id: str

    direction: Literal[
        "technical",
        "semantic",
        "event_integrity",
        "event_temporal",
        "event_frequency",
        "naturalness",
        "constraint",
        "keyframe",
        "physical",
        "preservation",
    ]

    evaluator: str

    required: bool

    critical: bool

    target_segments: list[int]

    body_parts: list[str]

    source_spec_ids: list[str]

    threshold_config: dict

    render_needed: bool

    metadata: dict
```

---

# 10.18 Verification 执行顺序

推荐：

```text
Stage 0
Technical Validation

Stage 1
Deterministic Numerical Checks

Stage 2
Learned Motion Metrics

Stage 3
MLLM Semantic / Event Checks

Stage 4
Aggregation
```

---

## 为什么这样排序

```text
Deterministic
→ 便宜、稳定、可解释

Learned Critic
→ 中等成本

MLLM
→ 成本最高
```

同时可以最大化：

```text
cache reuse
parallel execution
```

---

# 10.19 Stage 0：Technical Validator

先确认 Candidate 本身有效。

检查：

```text
candidate handle exists

frame count correct

SMPL fields present

no NaN

no Inf

motion_repr shape valid

fps consistent

duration consistent
```

输出：

```python
TechnicalCheckResult
```

如果：

```text
technical = fail
```

直接：

```text
verification_status = invalid
overall_pass = false
```

无需继续调用昂贵 MLLM。

---

# 10.20 Semantic Verifier

Semantic Verifier 负责：

> Motion 整体是否表达了用户要求的动作语义？

它不负责：

```text
精确 Event Count
精确 Temporal Order
数值 Constraint
```

---

# 10.21 Semantic Verification 的两层设计

推荐：

```text
TMR
+
Selective MLLM Semantic Critic
```

---

## 10.21.1 TMR

输入：

```text
Motion Specification canonical semantic description

Champion Motion
```

输出：

```python
tmr_similarity: float
```

TMR 提供 cheap semantic evidence。

---

## 10.21.2 为什么 TMR 不能单独验收

TMR 是 cross-modal retrieval model。

它对：

```text
global semantic similarity
```

有效，

但不保证能稳定判断：

```text
right vs left

3 times vs 2 times

event A before B

contact error 5 cm
```

所以只作为：

```text
Semantic Core Signal
```

---

# 10.22 MLLM Semantic Critic

以下情况启用：

```text
complex composite action

body-part-specific request

direction requirement

rare style

simultaneous action

interaction semantics

TMR score near decision boundary
```

简单单动作任务可以根据配置只运行：

```text
TMR + Naturalness
```

以降低成本。

---

# 10.23 Semantic Critic 输入

使用：

```text
Original Request

Motion Specification

Champion Render

TMR Evidence
```

不要只给：

```text
GEM Caption
```

因为 Motion Specification 保存了：

```text
body part
direction
style
secondary action
```

等更完整信息。

---

# 10.24 Semantic Critic 输出

```python
class SemanticFinding(BaseModel):

    main_action_match: bool

    body_part_match: bool | None

    direction_match: bool | None

    style_match: bool | None

    interaction_match: bool | None

    missing_semantics: list[str]

    hallucinated_semantics: list[str]

    confidence: float

    status: Literal[
        "pass",
        "fail",
        "uncertain",
    ]
```

---

# 10.25 Semantic Critic Prompt

```text
You are the Semantic Motion Verifier for MotionAgent.

Your task is to determine whether the generated human motion
matches the semantic requirements of the structured Motion Specification.

Evaluate only semantic motion content.

Check when applicable:
- main action;
- body part;
- direction;
- motion style;
- simultaneous secondary action;
- interaction semantics.

Do not evaluate:
- exact repetition count;
- exact temporal ordering;
- numerical constraints;
- rendering aesthetics;
- clothing;
- camera;
- background.

Those are handled by other verifiers.

Use the rendered motion only as evidence for the structured specification.

Return only SemanticFinding.
```

---

# 10.26 Event Verifier

Event Verifier 完全按：

```text
Integrity
Temporal
Frequency
```

三个子方向拆开。

不使用一个：

```text
generic event score
```

---

# 10.27 AToM-style Visual Formatting

AToM 的核心经验之一是：

> 不同 Event Task 要使用 Task-specific visual formatting 和 instructions。

本项目采用：

```text
Rendered Motion
→ Storyboard / Clip
→ Event-specific Prompt
```

---

## 10.27.1 Storyboard Builder

建议：

```text
verification/rendering/event_storyboard.py
```

接口：

```python
def build_event_storyboard(
    candidate_id,
    event_spec,
    profile,
) -> RenderBundleHandle:
    ...
```

---

## 10.27.2 Frame Sampling

不能简单：

```text
全局均匀抽 8 帧
```

因为可能错过很短的动作。

建议：

```text
Segment boundary frames
+
Segment center frames
+
Uniform frames
```

组成：

```text
8 ~ 16 frame storyboard
```

根据任务长度配置。

---

## 10.27.3 Frequency Task

Frequency 更依赖时间密度。

优先：

```text
target segment clip
```

而不是整段视频低密度采样。

第一版可以：

```text
short rendered clip
或
12~16 frame dense storyboard
```

---

# 10.28 Event Integrity

问题：

> 要求的 Motion Events 是否全部出现？

例如：

```text
walk
bend down
pick up
```

期望：

```text
3 required events
```

输出：

```python
class IntegrityFinding(BaseModel):

    expected_events: list[str]

    observed_events: list[str]

    missing_events: list[str]

    extra_events: list[str]

    atom_score: int | None

    confidence: float

    status: Literal[
        "pass",
        "fail",
        "uncertain",
    ]
```

---

## 10.28.1 Pass Rule

默认：

```text
missing_events == []
```

才通过。

如果 User Prompt 允许：

```text
optional event
```

由 Motion Specification 明确标记，

不能由 Verifier 自己猜。

---

# 10.29 Event Temporal

问题：

> Event 顺序是否符合 Motion Specification？

例如：

```text
wave
→ walk
→ turn left
```

输出：

```python
class TemporalFinding(BaseModel):

    expected_order: list[str]

    observed_order: list[str]

    order_violations: list[dict]

    confidence: float

    status: Literal[
        "pass",
        "fail",
        "uncertain",
    ]
```

---

## 10.29.1 Pass Rule

```text
所有 required precedence relation 成立
```

才通过。

不要求每个动作：

```text
严格开始 / 结束 frame
```

完全等于 Plan，

除非 Motion Specification 有：

```text
explicit timing requirement
```

---

# 10.30 Event Frequency

问题：

> 目标动作实际发生了几次？

例如：

```text
wave right hand 3 times
```

输出：

```python
class FrequencyFinding(BaseModel):

    event: str

    expected_count: int

    observed_count: int | None

    count_difference: int | None

    confidence: float

    status: Literal[
        "pass",
        "fail",
        "uncertain",
    ]
```

---

## 10.30.1 Pass Rule

默认：

```text
observed_count == expected_count
```

如果用户表达：

```text
at least 3 times
about 3 times
```

应由 Motion Compiler 将：

```text
count relation / tolerance
```

写入 Motion Specification，

Verifier 直接使用该规则。

---

# 10.31 AToM Score 的使用

如果需要与 AToM 风格保持兼容，可以额外保存：

```text
1~5 annotation score
```

但：

> **MotionAgent 的 pass/fail 不应依赖一个模糊总分。**

优先使用：

```text
missing event

order violation

observed count
```

这种结构化 Evidence。

---

# 10.32 Event Critic Prompt

不同任务使用不同 Prompt。

例如 Integrity：

```text
You are the Event Integrity Verifier.

Determine whether every required motion event appears in the motion.

Do not evaluate naturalness.
Do not evaluate exact event order unless needed to identify an event.
Do not evaluate frequency unless the event definition itself requires it.

For each required event:
- mark observed / missing / uncertain;
- identify approximate temporal location.

Return only IntegrityFinding.
```

Temporal / Frequency 使用独立 Prompt。

不使用一个包含全部任务的超长 Prompt。

---

# 10.33 Naturalness Verifier

Naturalness 主要回答：

> Motion 看起来是否自然、平滑、合理，没有明显人体 Motion Artifact？

V1：

```text
MotionCritic
+
Kinematic Sanity Metrics
```

---

# 10.34 MotionCritic Adapter

MotionCritic 官方输入基于：

```text
SMPL 24 joints
+
root translation
```

而 GEM 的主 Motion Representation 使用：

```text
root
+
21 body joints
```

因此必须实现：

```text
MotionCriticAdapter
```

而不是直接把 GEM 151-D Tensor 输入 MotionCritic。

---

## 10.34.1 Adapter 流程

```text
GEM Candidate
      ↓
pred_body_params_global
      ↓
body_pose
global_orient
transl
      ↓
Convert to Standard SMPL Motion
      ↓
append neutral / identity rotations
for unsupported terminal hand joints if required
      ↓
Axis-Angle
      ↓
MotionCritic expected format
```

具体 Joint Mapping 必须用单元测试验证。

---

# 10.35 MotionCritic 输出

```python
class NaturalnessFinding(BaseModel):

    critic_score: float | None

    smoothness_metrics: dict

    artifact_flags: list[str]

    confidence: float

    status: Literal[
        "pass",
        "fail",
        "warning",
        "error",
    ]
```

---

# 10.36 MotionCritic Threshold

不要写：

```text
score > 4.0
→ pass
```

这种未经校准的固定规则。

Threshold 必须通过：

```text
GEM baseline generated motions
+
HumanML3D / valid natural motions
+
人工标注小验证集
```

做 calibration。

保存：

```text
motioncritic_threshold_v1
```

到：

```text
configs/verifier_thresholds.yaml
```

---

# 10.37 Kinematic Sanity

即使有 MotionCritic，也保留 cheap deterministic 指标。

至少：

```text
joint velocity outlier

joint acceleration outlier

joint jerk outlier

root velocity discontinuity
```

这些指标主要用于：

```text
failure diagnosis
```

不一定全部作为 hard fail。

---

# 10.38 Constraint Verifier

Constraint Verifier 是最确定性的 Verifier。

它必须直接读取第 7 单元生成的：

```text
VerificationSpec
```

禁止：

```text
重新让 LLM 理解 User Constraint
```

---

# 10.39 Constraint Evaluation

映射：

```text
joint_target
→ joint_position_error

root_trajectory
→ trajectory_rmse

contact
→ contact_distance + contact_velocity

fixed_joint
→ joint_drift / joint_velocity

body_part_pose
→ rotation / pose error
```

---

# 10.40 Constraint Finding

```python
class ConstraintFinding(BaseModel):

    constraint_id: str

    metric_type: str

    target_segments: list[int]

    body_parts: list[str]

    observed_value: float

    target_value: float | None

    threshold: float

    units: str

    error: float

    status: Literal[
        "pass",
        "fail",
    ]
```

---

# 10.41 Constraint Pass Rule

完全使用：

```text
VerificationSpec.pass_threshold
```

不允许 Verifier 自己重新定义。

例如：

```text
right wrist position error
= 0.08 m

threshold
= 0.05 m

→ fail
```

---

# 10.42 Keyframe Verifier

Keyframe 是独立 Control Requirement。

输入：

```text
KeyframeSpec
+
Candidate Motion
```

检查：

```text
正确 Frame 附近
是否出现目标 whole-body pose
```

---

## 10.42.1 Keyframe Metric

优先：

```text
Joint Rotation Error
+
Joint Position Error
```

不要用：

```text
pixel similarity
```

作为主指标。

---

## 10.42.2 Temporal Tolerance

Keyframe 可以允许：

```text
±N frames
```

的 temporal tolerance。

该值来自：

```text
KeyframeSpec
```

不由 Verifier 自行决定。

在允许窗口中：

```text
取最接近 target pose 的 frame
```

作为实际 Keyframe Match。

---

# 10.43 Physical Verifier

V1 只实现：

```text
容易计算
无需 Physics Simulator
可解释
```

的 3D Motion Metric。

---

# 10.44 Ground Estimation

Pure Text Motion 没有真实 Scene Ground Mesh。

V1 使用：

```text
foot joint height statistics
```

估计：

```text
ground_y
```

例如：

```text
低分位 foot height
```

作为 Ground Reference。

所有阈值通过：

```text
validation data
```

校准。

---

# 10.45 Foot Skating

使用：

```text
left_foot
right_foot
```

Joint Position。

先检测 Contact Frame：

```text
foot height close to ground
+
vertical velocity small
```

再计算：

```text
horizontal foot speed
```

例如：

\[
v_{xy}(t)
=
\frac{
\|p_{xy}(t+1)-p_{xy}(t)\|
}{
\Delta t
}
\]

记录：

```text
mean contact skating speed

max contact skating speed

skating frame ratio
```

---

# 10.46 Ground Penetration

定义：

\[
E_{penetration}
=
\max(0, y_{ground} - y_{foot})
\]

记录：

```text
mean penetration

max penetration

penetration frame ratio
```

---

# 10.47 Smoothness / Jerk

对 Joint / Root Position：

```text
velocity
acceleration
jerk
```

计算：

```text
RMS
max
outlier ratio
```

主要用于：

```text
naturalness / physical diagnostic
```

---

# 10.48 Contact Stability

如果：

```text
active contact constraint
```

可进一步计算：

```text
contact distance
+
contact-point velocity
```

这部分与 Constraint Verifier 共用底层 Metric。

不要重复实现两套。

---

# 10.49 V1 Physical Scope

V1 不做：

```text
MuJoCo simulation

IsaacGym simulation

full-body dynamic force feasibility

balance controller

contact force estimation
```

如果后续需要更强 Physical Fidelity：

```text
PP-Motion
PhyMotion
Physics Simulator
```

可以作为 Phase II verifier plugin。

---

# 10.50 Preservation Verifier

只用于：

```text
scope="segment"
```

检查 Repair 是否破坏已通过部分。

---

## 10.50.1 Outside-Segment Preservation

比较：

```text
Previous Candidate
vs
New Champion
```

在：

```text
target_segment 之外
```

的：

```text
joint rotations

root translation

body shape
```

---

## 10.50.2 Metrics

至少：

```text
outside_pose_error

outside_root_error

body_shape_error

left_boundary_pose_jump

right_boundary_pose_jump

boundary_velocity_jump
```

---

# 10.51 Preservation Finding

```python
class PreservationFinding(BaseModel):

    previous_candidate_id: str

    current_candidate_id: str

    outside_pose_error: float

    outside_root_error: float

    boundary_pose_error: float

    boundary_velocity_error: float

    status: Literal[
        "pass",
        "fail",
        "warning",
    ]
```

---

# 10.52 Render Strategy

不是所有 Verifier 都需要 Render。

---

## No Render

```text
TMR
MotionCritic
Constraint Metrics
Keyframe Metrics
Physical Metrics
Preservation Metrics
```

直接从：

```text
SMPL / Motion Representation
```

计算。

---

## Render Required

```text
MLLM Semantic Critic
AToM-style Integrity
AToM-style Temporal
AToM-style Frequency
```

---

# 10.53 Render Cache 与第 9 单元复用

如果 Tournament 已经生成兼容的：

```text
RenderBundle
```

Verifier 优先复用。

Cache Key：

```text
candidate_id
+
render_profile_hash
```

如果 Event Verifier 需要：

```text
更高时间密度 storyboard
```

再单独创建：

```text
event-specific render
```

---

# 10.54 Verifier Worker Registry

不要把所有 evaluator 写在一个文件里。

建议：

```python
VERIFIER_REGISTRY = {

    "technical":
        TechnicalVerifier(),

    "semantic_tmr":
        TMRVerifier(),

    "semantic_mllm":
        SemanticMLLMVerifier(),

    "event_integrity":
        IntegrityVerifier(),

    "event_temporal":
        TemporalVerifier(),

    "event_frequency":
        FrequencyVerifier(),

    "motioncritic":
        MotionCriticVerifier(),

    "constraint":
        ConstraintVerifier(),

    "keyframe":
        KeyframeVerifier(),

    "physical":
        PhysicalVerifier(),

    "preservation":
        PreservationVerifier(),
}
```

---

# 10.55 并行执行

Stage 1 以后，大部分 Check 可以并行。

例如：

```text
TMR
MotionCritic
Constraint Metrics
Physical Metrics
```

同时执行。

MLLM：

```text
Semantic
Integrity
Temporal
Frequency
```

也可以按 Check 并行。

最后统一 Aggregation。

---

# 10.56 `VerifierFinding`

所有方向统一输出父 Schema。

```python
class VerifierFinding(BaseModel):

    finding_id: str

    check_id: str

    direction: str

    evaluator: str

    status: Literal[
        "pass",
        "fail",
        "warning",
        "uncertain",
        "error",
    ]

    severity: Literal[
        "critical",
        "major",
        "minor",
        "info",
    ]

    target_segments: list[int]

    body_parts: list[str]

    source_spec_ids: list[str]

    expected: dict

    observed: dict

    metrics: dict

    confidence: float | None

    evidence_handles: list[str]

    diagnostic_code: str | None

    message: str
```

---

# 10.57 Diagnostic Code

Verifier 不直接给：

```text
recommended_repair
```

只提供客观：

```text
diagnostic_code
```

第一版例如：

```text
SEMANTIC_MAIN_ACTION_MISMATCH

SEMANTIC_BODY_PART_MISMATCH

SEMANTIC_DIRECTION_MISMATCH

EVENT_MISSING

EVENT_ORDER_VIOLATION

EVENT_FREQUENCY_MISMATCH

MOTION_UNNATURAL

CONSTRAINT_POSITION_ERROR

CONSTRAINT_CONTACT_ERROR

CONSTRAINT_TRAJECTORY_ERROR

KEYFRAME_POSE_MISMATCH

PHYSICS_FOOT_SKATING

PHYSICS_GROUND_PENETRATION

MOTION_DISCONTINUITY

PRESERVATION_FAILURE
```

Diagnosis Agent 再将这些 Code 映射到：

```text
COMPILE
RETRIEVE
CONSTRAINT
KEYFRAME
GENERATE
```

---

# 10.58 Threshold 管理

禁止把 Threshold 分散 hard-code 在 Verifier 文件。

统一：

```text
configs/verifier_thresholds.yaml
```

---

## 10.58.1 Threshold 来源

优先级：

```text
1. User explicit requirement

2. VerificationSpec / KeyframeSpec

3. Benchmark definition

4. Calibrated verifier threshold profile
```

---

## 10.58.2 Learned Score Threshold

例如：

```text
TMR

MotionCritic
```

必须用：

```text
held-out validation data
```

校准。

不能因为某个 Demo：

```text
MotionCritic = 4.1
```

就把：

```text
4.0
```

当成通用 pass threshold。

---

# 10.59 Semantic Threshold Calibration

建议构造：

```text
positive text-motion pairs

hard negative pairs

body-part swapped negatives

direction swapped negatives
```

在验证集上选择：

```text
precision / recall tradeoff
```

得到：

```text
tmr_threshold_v1
```

---

# 10.60 Event Threshold

Event 默认不使用连续 score threshold。

直接使用结构条件：

```text
Integrity:
missing_event == 0

Temporal:
order_violation == 0

Frequency:
count relation satisfied
```

这比：

```text
event score > 0.7
```

更容易解释。

---

# 10.61 Constraint / Keyframe Threshold

完全使用：

```text
VerificationSpec
KeyframeSpec
```

提供的 tolerance。

Verifier 不重新调参。

---

# 10.62 Physical Threshold

通过：

```text
HumanML3D valid motion

GEM baseline generation

少量人工检查
```

校准：

```text
skating speed

penetration depth

jerk
```

并版本化保存。

---

# 10.63 Aggregation：不要用单一加权平均

最终验收不使用：

\[
Score =
w_1 semantic
+
w_2 naturalness
+
...
\]

然后：

```text
Score > 0.8
→ pass
```

原因：

```text
严重 Constraint Failure
可能被高 Semantic Score 抵消
```

这是不可接受的。

---

# 10.64 Pass / Fail Aggregation

使用：

> **Required-Check Gating**

规则：

```python
overall_pass = (
    verification_status == "complete"
    and
    all(
        check.status == "pass"
        for check in required_checks
    )
)
```

---

## 10.64.1 Critical Failure

任意：

```text
critical check == fail
```

则：

```text
overall_pass = false
```

---

## 10.64.2 Required Uncertain

如果：

```text
required check == uncertain
```

则：

```text
verification_status = incomplete
overall_pass = false
```

不能把：

```text
不知道
```

当成通过。

---

## 10.64.3 Required Evaluator Error

如果：

```text
required evaluator error
```

尝试有限次数 Infrastructure Retry。

仍失败：

```text
verification_status = incomplete
overall_pass = false
```

---

## 10.64.4 Optional Warning

```text
optional check == warning
```

不阻止：

```text
overall_pass
```

但写入 Report。

---

# 10.65 Severity

推荐：

```text
critical

major

minor

info
```

---

## Critical

例如：

```text
main action wrong

required event missing

explicit constraint fail

required keyframe fail

severe preservation failure
```

---

## Major

例如：

```text
motion clearly unnatural

strong foot skating

important style mismatch
```

---

## Minor

例如：

```text
moderate jerk

small non-required physical artifact
```

---

# 10.66 `VerificationReport`

最终输出：

```python
class VerificationReport(BaseModel):

    verification_id: str

    champion_candidate_id: str

    tournament_id: str

    status: Literal[
        "complete",
        "incomplete",
        "invalid",
        "failed_service",
    ]

    overall_pass: bool

    plan: VerificationPlan

    findings: list[VerifierFinding]

    passed_check_ids: list[str]

    failed_check_ids: list[str]

    warning_check_ids: list[str]

    uncertain_check_ids: list[str]

    failure_summary: VerificationFailureSummary

    display_scores: dict

    evidence_fingerprint: str

    runtime: VerificationRuntime
```

---

# 10.67 `VerificationFailureSummary`

```python
class VerificationFailureSummary(BaseModel):

    primary_directions: list[str]

    critical_failures: list[str]

    affected_segments: list[int]

    affected_body_parts: list[str]

    failed_constraint_ids: list[str]

    failed_keyframe_ids: list[str]

    failed_event_ids: list[str]

    diagnostic_codes: list[str]
```

该 Summary 直接作为：

```text
Diagnosis Agent
```

的输入之一。

---

# 10.68 `display_scores`

为了 UI / Debug 可以保存：

```json
{
  "semantic_tmr": 0.82,
  "motioncritic": 3.91,
  "trajectory_rmse_m": 0.04,
  "foot_skating_ratio": 0.07
}
```

但：

> `display_scores` 只用于观察，不用于统一加权决定 `overall_pass`。

---

# 10.69 Planner State 写回

Verifier 完成后：

```python
state.latest_verification = {
    "verification_id":
        report.verification_id,

    "overall_pass":
        report.overall_pass,

    "status":
        report.status,

    "passed":
        report.passed_check_ids,

    "failed":
        report.failed_check_ids,

    "critical_failures":
        report.failure_summary.critical_failures,

    "affected_segments":
        report.failure_summary.affected_segments,

    "diagnostic_codes":
        report.failure_summary.diagnostic_codes,
}
```

详细 Finding 放：

```text
Verification Store
```

Planner Context 只读 Summary。

---

# 10.70 与第 3 单元 Planner 的连接

如果：

```text
overall_pass = true
```

后续：

```text
Diagnosis
→ no critical failure
→ Planner
→ ACCEPT
```

如果：

```text
overall_pass = false
```

则：

```text
VerificationReport
→ Diagnosis
→ Planner
```

Planner 下一轮可能：

```text
COMPILE_MOTION

RETRIEVE_REFERENCE

BUILD_CONSTRAINT

BUILD_KEYFRAME

GENERATE
```

但 Multi-Verifier 不直接选择这些 Action。

---

# 10.71 与第 11 单元 Diagnosis 的接口

Diagnosis 输入：

```text
Original Motion Plan

VerificationFailureSummary

Relevant VerifierFinding[]

Generation Metadata

Repair History
```

Verifier 必须保证 Finding 已经定位到尽可能具体的：

```text
segment

event

body part

constraint

keyframe
```

这样 Diagnosis 不需要重新观看整个 Motion 才能猜 Failure。

---

# 10.72 Verification Service

建议：

```text
Orchestrator
     │
     ▼
Verification Service
     │
     ├── VerificationPlanBuilder
     ├── Candidate Store
     ├── Render Cache
     │
     ├── TMR Service
     ├── MotionCritic Service
     ├── Deterministic Metric Workers
     ├── MLLM Event/Semantic Provider
     │
     └── Verification Store
```

---

# 10.73 对外接口

```python
class VerificationService:

    async def verify(
        self,
        request: VerificationRequest,
    ) -> VerificationReport:
        ...
```

---

# 10.74 Model Lifecycle

以下模型不应该每次 Verification 重新加载：

```text
TMR

MotionCritic

MLLM Client
```

服务启动：

```text
load model
warmup
ready
```

Request Runtime：

```text
load candidate
run selected checks
aggregate
save report
```

---

# 10.75 Evidence Cache

第 9 单元已经可能计算：

```text
TMR

MotionCritic

Constraint quick metrics

Render
```

如果：

```text
model version
input candidate
metric config
```

完全相同，

Verifier 可以复用：

```text
raw evidence
```

但仍必须重新应用：

```text
absolute threshold
pass/fail rule
```

Tournament 的：

```text
relative winner
```

不能直接变成 Verifier 的：

```text
pass
```

---

# 10.76 Verifier Cache Key

```text
candidate_id

check type

source spec version

threshold profile

evaluator version

render profile

prompt version
```

任一变化：

```text
cache invalid
```

---

# 10.77 MLLM Verifier 稳定性

对于：

```text
Semantic

Integrity

Temporal

Frequency
```

需要防止：

```text
单次 LLM 随机判断
```

V1 推荐：

```text
temperature = 0 / deterministic mode
+
strict structured schema
+
task-specific prompt
```

---

## 10.77.1 Uncertain

MLLM 可以输出：

```text
uncertain
```

而不是被迫：

```text
pass / fail
```

发生 uncertain：

```text
尝试更密集 Render
→ rerun once
```

如果仍 uncertain：

```text
required check
→ incomplete
```

---

# 10.78 Infrastructure Retry

Verifier Service 可以对：

```text
timeout
invalid schema
temporary model error
```

做 Infrastructure Retry。

必须保持：

```text
same candidate

same specification

same threshold

same prompt version
```

它不是 Agent Retry。

---

# 10.79 内部代码结构

建议：

```text
verification/
│
├── schemas.py
├── service.py
├── plan_builder.py
├── aggregator.py
├── store.py
├── cache.py
│
├── technical.py
│
├── semantic/
│   ├── tmr.py
│   ├── mllm.py
│   └── prompt.py
│
├── event/
│   ├── integrity.py
│   ├── temporal.py
│   ├── frequency.py
│   ├── storyboard.py
│   └── prompts.py
│
├── naturalness/
│   ├── motioncritic.py
│   ├── adapter.py
│   └── kinematic.py
│
├── constraint/
│   └── verifier.py
│
├── keyframe/
│   └── verifier.py
│
├── physical/
│   ├── foot_skating.py
│   ├── penetration.py
│   └── smoothness.py
│
├── preservation/
│   └── verifier.py
│
└── thresholds/
    ├── loader.py
    └── calibration.py
```

---

# 10.80 `VerifierPlanBuilder` 伪代码

```python
def build_verification_plan(
    request,
    motion_spec,
    constraints,
    keyframes,
):

    checks = []

    checks.append(
        technical_check()
    )

    checks.append(
        semantic_check()
    )

    checks.append(
        naturalness_check()
    )

    events = extract_required_events(
        motion_spec
    )

    if len(events) >= 2:
        checks.append(
            integrity_check(
                events
            )
        )

    if has_required_event_order(
        motion_spec
    ):
        checks.append(
            temporal_check(
                motion_spec
            )
        )

    for event in events:

        if event.repetition is not None:
            checks.append(
                frequency_check(
                    event
                )
            )

    for spec in constraints:

        checks.append(
            constraint_check(
                spec
            )
        )

    for keyframe in keyframes:

        checks.append(
            keyframe_check(
                keyframe
            )
        )

    if requires_physical_check(
        motion_spec,
        constraints,
    ):
        checks.append(
            physical_check(
                required=True
            )
        )

    elif request.verifier_policy.enable_optional_checks:
        checks.append(
            physical_check(
                required=False
            )
        )

    if request.generation_scope == "segment":

        checks.append(
            preservation_check(
                previous_candidate_id=
                    request.previous_candidate_id,

                target_segments=
                    request.target_segments,
            )
        )

    return VerificationPlan(
        ...
    )
```

---

# 10.81 Verification 主流程伪代码

```python
async def verify_motion(
    request: VerificationRequest,
) -> VerificationReport:

    candidate = candidate_store.load(
        request.champion_candidate_id
    )

    motion_spec = motion_spec_store.load(
        request.motion_spec_id
    )

    constraint_specs = load_constraint_verification_specs(
        request.constraint_verification_specs
    )

    keyframes = load_keyframes(
        request.active_keyframe_ids
    )

    plan = build_verification_plan(
        request=request,
        motion_spec=motion_spec,
        constraints=constraint_specs,
        keyframes=keyframes,
    )

    technical = run_technical_verifier(
        candidate
    )

    if technical.status != "pass":

        return build_invalid_report(
            request=request,
            plan=plan,
            technical=technical,
        )

    cheap_checks = select_checks(
        plan,
        stages=[
            "deterministic",
            "learned_motion",
        ],
    )

    cheap_findings = await run_checks_parallel(
        cheap_checks,
        candidate,
    )

    render_checks = select_render_checks(
        plan
    )

    render_handles = await build_required_renders(
        candidate=candidate,
        checks=render_checks,
    )

    mllm_findings = await run_mllm_checks_parallel(
        checks=render_checks,
        candidate=candidate,
        renders=render_handles,
        motion_spec=motion_spec,
    )

    findings = (
        [technical]
        + cheap_findings
        + mllm_findings
    )

    return aggregate_verification(
        request=request,
        plan=plan,
        findings=findings,
    )
```

---

# 10.82 Constraint Verifier 伪代码

```python
def verify_constraint(
    candidate,
    verification_spec,
):

    metric = METRIC_REGISTRY[
        verification_spec.metric_type
    ]

    value = metric.compute(
        motion=candidate,
        target=verification_spec.target,
        frames=verification_spec.frame_indices,
        body_parts=verification_spec.body_parts,
    )

    passed = (
        value
        <= verification_spec.pass_threshold
    )

    return VerifierFinding(
        direction="constraint",
        status=(
            "pass"
            if passed
            else "fail"
        ),
        expected={
            "threshold":
                verification_spec.pass_threshold
        },
        observed={
            "value": value
        },
        ...
    )
```

---

# 10.83 Event Frequency 伪代码

```python
async def verify_frequency(
    candidate,
    event_spec,
):

    storyboard = await event_renderer.render_dense(
        candidate=candidate,
        segment_id=event_spec.segment_id,
    )

    finding = await event_mllm.verify_frequency(
        storyboard=storyboard,
        event=event_spec.action,
        expected_count=event_spec.repetition,
    )

    return finding
```

---

# 10.84 Physical Metric 伪代码

```python
def compute_foot_skating(
    joints,
    fps,
    ground_y,
):

    foot = joints[
        ...,
        FOOT_IDS,
        :
    ]

    velocity = (
        foot[1:]
        - foot[:-1]
    ) * fps

    height = (
        foot[..., 1]
        - ground_y
    )

    contact = (
        height < CONTACT_HEIGHT_THRESHOLD
    )

    horizontal_speed = norm(
        velocity[..., [0, 2]],
        dim=-1,
    )

    skating = (
        horizontal_speed
        * contact[:-1]
    )

    return {
        "mean_speed":
            skating.sum()
            / contact[:-1].sum().clamp_min(1),

        "max_speed":
            skating.max(),

        "frame_ratio":
            (
                skating
                > SKATING_SPEED_THRESHOLD
            ).float().mean(),
    }
```

实际阈值从：

```text
threshold config
```

读取，不 hard-code 在函数中。

---

# 10.85 Aggregation 伪代码

```python
def aggregate_verification(
    request,
    plan,
    findings,
):

    required = {
        check.check_id
        for check in plan.checks
        if check.required
    }

    by_check = group_findings(
        findings
    )

    incomplete = False

    failed_required = []

    for check_id in required:

        result = by_check[
            check_id
        ]

        if result.status in [
            "uncertain",
            "error",
        ]:
            incomplete = True

        elif result.status == "fail":
            failed_required.append(
                check_id
            )

    if incomplete:
        status = "incomplete"
        overall_pass = False

    else:
        status = "complete"
        overall_pass = (
            len(failed_required) == 0
        )

    return VerificationReport(
        status=status,
        overall_pass=overall_pass,
        ...
    )
```

---

# 10.86 V1 实现顺序

## Step 1：Schema + Plan Builder

先实现：

```text
VerificationRequest

VerificationPlan

VerifierCheck

VerifierFinding

VerificationReport
```

并测试 Verifier Direction Selection。

---

## Step 2：Technical / Constraint

优先完成最确定性的：

```text
Technical Validator

Constraint VerificationSpec
```

---

## Step 3：TMR Semantic

完成：

```text
Champion Motion
→ TMR Adapter
→ similarity
```

以及 Threshold Calibration。

---

## Step 4：MotionCritic

完成：

```text
GEM SMPL
→ MotionCriticAdapter
→ naturalness score
```

---

## Step 5：AToM Event Verifier

按顺序：

```text
Integrity

Temporal

Frequency
```

每个使用独立 Prompt 与 Output Schema。

---

## Step 6：Physical Metrics

实现：

```text
foot skating

ground penetration

jerk / smoothness
```

---

## Step 7：Keyframe

实现：

```text
pose / joint error
```

---

## Step 8：Preservation

连接：

```text
scope="segment"
```

局部 Retry。

---

## Step 9：Aggregator

实现：

```text
required gate

critical failure

optional warning

incomplete state
```

---

## Step 10：Service / Cache

完成：

```text
parallel execution

model reuse

render cache

evidence cache

result store
```

---

# 10.87 V1 最低实现范围

V1 必须完成：

```text
VerifierPlanBuilder

Technical Validator

TMR Semantic Verifier

Selective Semantic MLLM

AToM Integrity Verifier

AToM Temporal Verifier

AToM Frequency Verifier

MotionCritic Adapter + Verifier

Constraint Verifier

Keyframe Verifier

Foot Skating

Ground Penetration

Smoothness / Jerk

Segment Preservation

Required-check Aggregator

VerificationReport

Diagnosis Handoff
```

V1 可以暂缓：

```text
Physics Simulator

PP-Motion learned metric

PhyMotion

ReAlign step-aware reward as verifier

multi-model verifier ensemble

learned verifier router

RL-trained verifier
```

---

# 10.88 单元测试要求

至少覆盖：

```text
Single Simple Action

Multi-Event Integrity

Explicit Temporal Order

Frequency = 3

Body-Part Semantic Error

Direction Semantic Error

Natural Motion

Jittery Motion

Joint Target Pass

Joint Target Fail

Contact Pass

Contact Fail

Root Trajectory Pass

Keyframe Pass

Keyframe Fail

Foot Skating

Ground Penetration

Segment Preservation

Required Verifier Uncertain

Optional Verifier Warning

Evaluator Timeout

Evaluator Schema Error

Cache Reuse

Overall Pass Aggregation
```

---

## Case 1：Simple Walk

```text
Input:
walk forward

Expected Plan:
technical
semantic
naturalness
physical

No:
frequency
constraint
keyframe
preservation
```

---

## Case 2：Wave Three Times

```text
Input:
wave right hand three times

Expected:
semantic
event_frequency
naturalness

Frequency:
expected_count = 3
```

---

## Case 3：Three Ordered Events

```text
Input:
wave, then walk, then sit

Expected:
event_integrity
event_temporal
```

---

## Case 4：Constraint

```text
Constraint:
right wrist <= 5cm from target

Observed:
8cm

Expected:
Constraint Finding = fail

overall_pass = false
```

---

## Case 5：Tournament Winner Still Fails

```text
Tournament Champion:
best among K

But:
Frequency expected 3
Observed 2

Expected:
Verifier fail

Diagnosis receives:
EVENT_FREQUENCY_MISMATCH
```

---

## Case 6：Segment Retry

```text
scope = segment

Target segment improved

But outside segment changed strongly

Expected:
preservation = fail
overall_pass = false
```

---

# 10.89 Verifier 评估指标

系统级：

```text
Verification Latency

Per-Verifier Latency

MLLM Cost

Render Cost

Cache Hit Rate

Evaluator Error Rate

Incomplete Verification Rate
```

准确性：

```text
Verifier / Human Agreement

Semantic Precision / Recall

Event Integrity Accuracy

Temporal Order Accuracy

Frequency Count Accuracy

Constraint Pass/Fail Accuracy

MotionCritic / Human Correlation

Physical Metric / Human Artifact Agreement
```

Agent 效果：

```text
Verifier Failure Localization Accuracy

Diagnosis Routing Accuracy

Repair Success After Verifier Failure

False Accept Rate

False Reject Rate

Average Verification Cost / Generation Round
```

---

# 10.90 最重要的指标：False Accept Rate

因为 Planner 的：

```text
ACCEPT
```

完全依赖：

```text
overall_pass
```

所以 Verifier Evaluation 不能只看：

```text
平均 score correlation
```

必须重点观察：

> **明显失败 Motion 被错误判为通过的比例。**

即：

```text
False Accept Rate
```

这是比：

```text
overall score accuracy
```

更重要的系统指标。

---

# 10.91 Ablation

建议至少比较：

```text
TMR only

MLLM Semantic only

TMR + MLLM

MotionCritic only

MotionCritic + Physical Metrics

Generic Event Prompt

AToM-style Task-specific Event Verifiers

Single Weighted Score

Required-Check Gating

All Verifiers Always On

VerifierPlanBuilder Dynamic Selection
```

重点验证：

```text
Dynamic Selection
```

是否在降低成本的同时保持验收质量。

---

# 10.92 参考工作与本模块对应关系

| 工作 | 借鉴内容 | 本模块位置 |
|---|---|---|
| VISTA | Specialized Critics | Multi-Verifier 总体设计 |
| VISTA | Critic specialization rather than one scalar judge | Verifier Directions |
| AToM | Integrity / Temporal / Frequency | Event Verifier |
| AToM | Task-specific visual formatting / scoring instructions | Event Storyboard + Prompt |
| MotionCritic | Human-perceptual motion quality | Naturalness Verifier |
| TMR | Text-motion shared embedding | Semantic Core Signal |
| PP-Motion | Physical + perceptual fidelity should be separated | Physical Verifier design |
| ReAlign | Text-alignment / motion-quality reward separation | Future shared reward interface |

---

# 10.93 当前模块完成标准

Multi-Verifier 完成后，必须稳定实现：

```text
Tournament Champion
        ↓
VerificationRequest
        ↓
VerifierPlanBuilder
        ↓
Selected Required / Optional Checks
        ↓
┌─────────────────────────────┐
│ Semantic                    │
│ Event: Integrity/Temporal   │
│ Event: Frequency            │
│ Naturalness                 │
│ Constraint                  │
│ Keyframe                    │
│ Physical                    │
│ Preservation                │
└─────────────────────────────┘
        ↓
Structured VerifierFinding[]
        ↓
Required-Check Aggregation
        ↓
VerificationReport
        ↓
FailureSummary
        ↓
Diagnosis
        ↓
Planner
```

并保证：

```text
Verifier 不选择 Candidate。

Verifier 不修改 Motion Plan。

Verifier 不重新生成 Motion。

Verifier 不直接执行 Repair。

Verifier 不决定 Planner Action。

Verifier Direction 由结构化任务规则自动选择，
不依赖另一个自由 LLM Router。

Constraint / Keyframe 直接复用已有 Spec，
不重新解释自然语言。

Event Alignment 按 Integrity / Temporal / Frequency 分开验证。

Naturalness 与 Physical Feasibility 分开验证。

最终 overall_pass 不使用可互相抵消的加权总分，
而使用 Required-Check Gating。

只有 complete 且所有 required checks 通过，
Planner 才可能 ACCEPT。
```

# 10.94 Heading Continuity Completion Addendum

Multi-Verifier 还必须保证：

```text
HeadingContinuitySpec 存在时启用 deterministic-first heading check；
anchor 来自对应 segment，而不是硬编码 world forward；
explicit turn window 正确 exempt；
UNCOMMANDED_HEADING_DRIFT 可阻止 overall_pass；
threshold 来自 config/calibration，不由 LLM 决定。
```
