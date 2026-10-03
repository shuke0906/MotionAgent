# MotionAgent for GEM：模块实现指南

> 本文件从总实现指南中拆分，对应第 5 单元。内容保持模块边界独立，其输入输出接口需与相邻模块的 Schema 保持一致。

# 5. Motion Compiler

## Phase 7 Composite Timeline Extension

`MotionSegment` retains the existing action/body-parts/repetition/direction
fields and adds `angle_deg`, `parent_segment_id`, `continuation_of`,
`simultaneous_with`, `explicit_duration_s` and resolved `duration_s`.
Relationships refer to segment IDs. Concurrent windows can overlap their
parent; sequential stages still cover the complete timeline without gaps.

For the required six-second, 30 fps prompt, the compiler emits walk `[0,120)`,
right-hand wave (repeat 2) `[60,120)`, left turn (90 degrees) `[120,150)` and
sit_down `[150,180)`. The explicit initial two seconds set the continuation
start; unspecified wave duration uses the existing action weight, and
unspecified terminal transitions share the remaining time equally. This is
a deterministic timing convention, not a claim that the prompt fixes every
duration. Simple existing single-segment simultaneous syntax stays supported.

`GenerationRequest.motion_spec` and persisted candidate request metadata
retain these relationships. GEM receives compound captions for concurrent
intent, partitioned only at the compiler's existing segment boundaries.
Camera/image placeholders have exactly `total_frames` rows. DSL overlap is
preserved; text-window count need not equal semantic-segment count.

### Deterministic Semantic Compilation Compatibility Update

The current executable backend is the bounded deterministic parser in
`semantic_parser.py`; the LLM-based parser described below remains the broader
design, not an implementation claim. A shared lexical table normalizes
imperative, third-person, gerund and past-tense forms of supported actions.
Conjunction splitting recognizes action verbs without splitting coordinated
body parts such as `left and right hands`.

Initial compilation records `source_text`, `source_start`, and `source_end`
on each parsed segment. These offsets reference the unchanged original input.
Temporal Resolver uses that clause, not the first occurrence of an action in
the whole prompt. Counts, continuous modes and alternation therefore remain
clause-scoped. Explicit concurrent child segments receive a `simultaneous`
temporal relation to their parent. Existing inline `secondary_actions` syntax
remains compatible.
When both primary and secondary actions carry independent counts, or a
concurrent clause has an explicit continuous mode, the existing child-segment
representation is used so temporal attributes cannot migrate between actions.

`validate_request_coverage` rejects supported source actions lost during initial
compilation and checks recorded count, body-part and direct turn-direction
provenance. This is bounded lexical coverage, not a universal natural-language
understanding guarantee. Targeted revision retains existing segment provenance;
its full-plan offsets do not become offsets into the revision instruction.
For legacy artifacts without source spans, existing structured or legacy
repetition counts remain authoritative rather than rebinding to the first
matching verb in the original request.
Caption validation checks all active actions using shared verb morphology.

`gem_text_compiler.py` partitions at existing start/end frames and composes one
caption from all active segments in each interval. It does not invent timings,
expand repetitions, apply body-part masks, add keyframes, or modify GEM.
`GEMTextCondition.segment_bounds` maps semantic IDs to original frame ranges.
Generation preflight and segment-range lookup use that map, with caption-index
fallback only for legacy conditions lacking it. GEM's native text API still
receives captions and normalized windows, not a new conditioning tensor.

Reproducibility validation must record prompt, actual sampler seed, checkpoint
and source hashes, software/GPU settings, and fresh-inference comparisons.
Cache hits are not evidence of deterministic inference. Compound conditioning
is an input representation change, not proof of better motion or exact cycles.

本节定义 Motion Compiler 的**实现规范**。

Motion Compiler 的职责是：

> **将开放自然语言编译成结构化、时间对齐、GEM-aware 的 Motion Specification，并生成可直接交给 GEM Adapter 的 Text Condition。**

Motion Compiler 是：

```text
LLM-driven
+
Rule-validated
+
Planner-controlled
```

的 Structured Tool / Subgraph。

它不拥有全局决策权。

### LangGraph 执行边界

V1 中 Motion Compiler 由 `compile_motion` LangGraph node wrapper 调用，但 Compiler 的 domain implementation 保持 framework-agnostic：

```text
PlannerDecision
→ LangGraph route
→ compile_motion node
→ CompilerRequestBuilder
→ Motion Compiler service
→ typed result / artifact persist / StateReducer
→ planner
```

Compiler 不读取 LangGraph topology，也不决定下一 node。

---

## 5.1 职责边界

Motion Compiler 负责：

```text
理解用户动作语义
拆解时序动作
识别 body part / direction / speed / style / repetition
生成 Human Motion DSL
生成 Timeline
识别 Retrieval / Constraint / Keyframe 候选需求
生成 GEM-friendly Caption
构造 GEM Text Condition
```

Motion Compiler 不负责：

```text
直接调用 Retrieval Tool
直接计算 IK / FK
直接生成数值 Constraint
直接构造 observed_motion_3d
直接构造 motion_mask_3d
直接生成 Keyframe
直接运行 GEM
直接进行 Verify
直接进行 Retry
```

如果 Compiler 发现：

```text
rare motion
explicit contact
root trajectory
whole-body keyframe
```

只输出对应的 `routing_hint`。

之后统一返回 Planner，由 Planner 决定是否执行：

```text
RETRIEVE_REFERENCE
BUILD_CONSTRAINT
BUILD_KEYFRAME
GENERATE
```

---

## 5.2 调用关系

Motion Compiler 只通过 Planner 调用。

主链路：

```text
Planner
   │
   └── COMPILE_MOTION
          │
          ▼
   Motion Compiler
          │
          ├── MotionSpecification
          ├── GEMTextCondition
          └── RoutingHints
          │
          ▼
    Update State
          │
          ▼
       Planner
```

Planner 再决定下一步。

例如：

```text
Compiler
→ retrieval_candidate
→ Planner
→ RETRIEVE_REFERENCE
→ Update State
→ Planner
→ COMPILE_MOTION(mode="revise", focus=["gem_caption"])
```

因此 Retrieval 可以辅助第二次 Caption Refinement，但 Retrieval 不在 Compiler 内部自动执行。

Retry 也不属于 Compiler。

如果生成失败：

```text
Generate
→ Tournament
→ Verify
→ Diagnosis
→ Planner
```

Planner 再决定：

```text
重新 COMPILE_MOTION
或 BUILD_CONSTRAINT
或 RETRIEVE_REFERENCE
或 BUILD_KEYFRAME
或直接重新 GENERATE
```

---

## 5.3 总体执行流程

Motion Compiler 内部固定执行以下流程：

```text
User Request / Existing Plan
          │
          ▼
1. Semantic Parsing
          │
          ▼
2. Temporal Resolution
          │
          ▼
3. Human Motion DSL Normalization
          │
          ▼
4. Timeline Compilation
          │
          ▼
5. Control Intent Detection
          │
          ▼
6. GEM Caption Optimization
          │
          ▼
7. Validation
          │
          ▼
8. GEM Text Compilation
          │
          ▼
CompilerResult
```

其中：

```text
LLM
→ 负责语义理解与 Caption Rewrite

Rules / Validator
→ 负责格式、时间、枚举、长度和逻辑一致性
```

不采用纯规则，也不完全依赖 LLM。

### 5.3.1 Temporal Resolver

Temporal Resolver 位于 Semantic Parser 之后，负责把自然语言中的时间模式转成结构化语义。它只修改 Motion DSL，不调用 GEM、不运行 diffusion、不验证最终动作是否真的满足重复次数。

支持的第一版语义：

```json
{
  "action": "wave",
  "temporal_constraint": {
    "type": "repetition",
    "mode": "cycle",
    "count": 3
  }
}
```

连续动作：

```json
{
  "action": "walk",
  "temporal_mode": "continuous"
}
```

交替关系：

```json
{
  "action": "wave",
  "temporal_relation": {
    "type": "alternating",
    "marker": "alternately"
  }
}
```

第一版确定性覆盖：

```text
once
twice
three times
four times
repeatedly
continuously
keep walking / keep waving
alternately / alternating
```

旧字段 `repetition` 可作为兼容摘要保留，但新的 source of truth 是
`temporal_constraint`。

---

# 5.4 输入接口

主接口：

```python
def compile_motion(
    state: MotionAgentState,
    mode: Literal["initial", "revise"],
    target_segments: list[int] | None = None,
    focus: list[str] | None = None,
) -> CompilerResult:
    ...
```

输入来源：

```text
state.user_request
state.motion_plan
state.retrieved_references
state.latest_verification
state.diagnosis
```

其中：

### `mode="initial"`

第一次建立 Motion Specification。

### `mode="revise"`

只修改已有计划中的目标部分。

例如：

```python
compile_motion(
    state=state,
    mode="revise",
    target_segments=[1],
    focus=["timeline", "frequency"],
)
```

禁止 revise 时无理由重建全部 Segment。

---

# 5.5 Semantic Parser

Semantic Parser 使用 LLM Structured Output。

目标：

> 将自然语言中的动作语义转成规范字段，而不是直接生成最终 Caption。

输入：

```text
Original User Request
Existing Motion Specification（如果 revise）
Target Segment（如果局部修改）
Verifier / Diagnosis Feedback（如果 repair）
```

输出：

```python
SemanticMotionPlan
```

示例：

用户：

```text
一个人一瘸一拐向前走，三秒后右手扶住桌子，然后转身慢慢坐下。
```

输出：

```json
{
  "actions": [
    {
      "action": "walk",
      "body_parts": ["full_body"],
      "direction": "forward",
      "speed": "normal",
      "style": ["limping", "asymmetric"]
    },
    {
      "action": "reach_and_support",
      "body_parts": ["right_arm", "right_hand"],
      "interaction": {
        "type": "contact",
        "target": "table"
      }
    },
    {
      "action": "turn_and_sit",
      "body_parts": ["full_body"],
      "speed": "slow"
    }
  ],
  "explicit_timing": [
    {
      "event": "right_hand_contact",
      "time_s": 3.0
    }
  ]
}
```

Semantic Parser 不允许：

```text
自行添加用户没有要求的动作
自行添加数值坐标
自行求解关节角
自行决定 GEM Tensor
```

---

# 5.6 Human Motion DSL

Motion Compiler 必须先输出统一的中间表示，不直接从 User Prompt 跳到 GEM Caption。

采用：

> **Semi-Open Human Motion DSL**

其中：

```text
action
→ open vocabulary

modifier / structural fields
→ controlled vocabulary
```

推荐 Segment Schema：

```python
class MotionSegment(BaseModel):
    segment_id: int

    action: str
    secondary_actions: list[str]

    body_parts: list[str]

    direction: str | None
    speed: str | None
    style: list[str]

    repetition: int | None
    temporal_constraint: dict | None
    temporal_mode: str | None
    temporal_relation: dict | None
    orientation: str | None
    transition: str | None

    interaction: dict | None

    duration_weight: float

    start_s: float | None
    end_s: float | None

    start_frame: int | None
    end_frame: int | None

    gem_caption: str | None

    confidence: float
```

重复动作不得只写成：

```json
{
  "repeat": 3
}
```

应归一化为：

```json
{
  "temporal_constraint": {
    "type": "repetition",
    "mode": "cycle",
    "count": 3
  }
}
```

`repetition` 字段只作为 legacy summary / verifier compatibility 使用。下游需要区分离散重复、连续执行、交替关系时必须读取 `temporal_constraint`、`temporal_mode` 和 `temporal_relation`。

---

## 5.6.0 Cross-Segment Continuity Policy

显式语义之外，Compiler 必须维护 Segment 之间的默认运动连续性。V1 首先实现 root heading / facing continuity。

新增结构：

```python
class HeadingContinuitySpec(BaseModel):
    mode: Literal[
        "free",
        "inherit_previous",
        "explicit",
        "follow_trajectory",
    ]

    anchor_segment_id: int | None
    explicit_facing: str | None
    source: Literal[
        "user_explicit",
        "continuity_default",
        "trajectory",
        "unknown",
    ]
```

`MotionSegment` 增加：

```python
heading_continuity: HeadingContinuitySpec
```

注意：Compiler 只产生**语义级 expectation**，不在这里计算数值 yaw、SMPL root rotation 或 tolerance。

### 默认继承规则

如果前一 Segment 已经建立明显 heading，例如：

```text
walk forward
```

而下一 Segment：

```text
wave right hand
sit down
reach forward
stand still
```

没有显式：

```text
turn
rotate
spin
face left/right/backward
follow a curved trajectory
```

则：

```text
mode = inherit_previous
anchor_segment_id = previous_segment_id
source = continuity_default
```

例如：

```text
walk forward
→ wave the right hand
→ sit down
```

应编译为：

```text
segment 0: heading established by forward locomotion
segment 1: inherit_previous(segment 0)
segment 2: inherit_previous(segment 1)
```

### 何时不能自动继承

以下情况不得强行加 heading continuity：

```text
用户明确要求 turn / rotate / face another direction
动作本身以旋转为核心（例如 spin / pirouette）
trajectory 明确要求 follow_tangent
语义不确定，无法建立可靠 heading anchor
```

此时使用：

```text
explicit
follow_trajectory
free
```

之一。

### Caption Materialization

为了在纯文本 GEM 阶段也尽量发挥作用，Caption Optimizer 可以把已确定的 continuity expectation 轻量写入 caption，例如：

```text
"the person waves the right hand while maintaining the same facing direction"
"the person sits down without turning away"
```

但它只能表达已存在于 DSL 的 continuity policy，不能自行创造新的方向要求。

这种 phrase 是结构性 continuity cue，不是新的用户动作。

## 5.6.1 `body_parts`

第一版固定为：

```text
full_body

head
torso
pelvis

left_arm
right_arm
left_hand
right_hand

left_leg
right_leg
left_foot
right_foot
```

---

## 5.6.2 `direction`

第一版允许：

```text
forward
backward
left
right
up
down
clockwise
counterclockwise
none
```

---

## 5.6.3 `speed`

第一版允许：

```text
very_slow
slow
normal
fast
very_fast
```

---

## 5.6.4 `transition`

第一版允许：

```text
continuous
abrupt
hold
pause
```

---

## 5.6.5 Simultaneous Action

如果用户要求：

```text
一边走一边挥右手
```

第一版不要生成两个重叠 Segment。

应表示为：

```json
{
  "action": "walk",
  "secondary_actions": [
    "wave right hand"
  ],
  "body_parts": [
    "full_body",
    "right_arm"
  ]
}
```

生成 Caption：

```text
a person walks forward while waving the right hand
```

第一版保持：

```text
一个时间区间
→ 一个主 Motion Segment
```

避免引入复杂的 time × body-part textual control。

---

# 5.7 Timeline Compiler

Timeline Compiler 使用：

```text
LLM 提取 explicit timing / duration relation
+
Deterministic Rule 计算具体时间
```

不允许 LLM 随意生成最终 frame boundary。

---

## 5.7.1 显式时间

用户明确说：

```text
3 秒后
持续 2 秒
第 5 秒
```

必须优先保留。

例如：

```json
{
  "event": "right_hand_contact",
  "time_s": 3.0
}
```

---

## 5.7.2 隐式时间

没有明确秒数时，LLM 只输出：

```text
duration_weight
```

例如：

```text
walk   -> 3
reach  -> 1
sit    -> 2
```

如果：

```text
total_duration = 6 s
```

由 Timeline Compiler 确定性计算：

```text
walk   0-3s
reach  3-4s
sit    4-6s
```

---

## 5.7.3 Timeline Hard Rules

必须满足：

```text
start_0 = 0

end_last = total_duration

segment.start < segment.end

Segment 默认不重叠

相邻 Sequential Segment 连续

所有 frame index 合法

最后一个 end_frame == total_frames
```

计算：

```python
start_frame = round(start_s * fps)
end_frame = round(end_s * fps)
```

最后统一做 boundary correction，保证 frame 不重复、不丢失。

---

# 5.8 Control Intent Detector

该模块负责识别：

> 当前需求是否仅靠 Text 即可，还是存在需要 Planner 后续处理的控制需求。

它只输出候选，不执行任何 Tool。

输出：

```python
list[ControlIntent]
```

Schema：

```python
class ControlIntent(BaseModel):
    intent_type: Literal[
        "retrieval_candidate",
        "constraint_candidate",
        "keyframe_candidate"
    ]

    segment_id: int

    reason_code: str

    details: dict
```

---

## 5.8.1 Retrieval Candidate

典型条件：

```text
rare motion
rare style
specialized gait
specialized sport / dance motion
模型语义连续失败
需要 reference pose / trajectory
```

示例：

```json
{
  "intent_type": "retrieval_candidate",
  "segment_id": 0,
  "reason_code": "RARE_STYLE",
  "details": {
    "query_hint": "asymmetric limping gait",
    "preferred_type": "motion"
  }
}
```

---

## 5.8.2 Constraint Candidate

典型条件：

```text
明确位置
明确接触
明确 joint target
root trajectory
固定某个 body part
可测量几何关系
```

示例：

```json
{
  "intent_type": "constraint_candidate",
  "segment_id": 1,
  "reason_code": "EXPLICIT_CONTACT",
  "details": {
    "constraint_type": "contact",
    "body_part": "right_wrist",
    "target": "table",
    "time_s": 3.0
  }
}
```

---

## 5.8.3 Keyframe Candidate

典型条件：

```text
某一时间必须出现完整姿态
动作边界需要明确 whole-body state
最终必须稳定处于某姿态
```

示例：

```json
{
  "intent_type": "keyframe_candidate",
  "segment_id": 2,
  "reason_code": "WHOLE_BODY_END_STATE",
  "details": {
    "time_s": 5.8,
    "description": "stable seated whole-body pose"
  }
}
```

---

## 5.8.4 Intent Detection 实现

第一版使用：

```text
LLM Structured Classification
+
Deterministic Rules
```

例如明显关键词：

```text
"touch"
"contact"
"stay fixed"
"reach position"
"follow a circle"
"at exactly 3 seconds"
```

可作为 rule feature。

最终以结构化 LLM 结果为主，Rule Validator 检查明显遗漏和非法分类。

---

# 5.9 GEM Caption Optimizer

Caption Optimizer 的目标：

> 将 Human Motion DSL 转换成 GEM 更容易利用的 Motion Caption。

参考思路：

```text
LAMP
→ 先得到结构化 Motion Program

RAPO
→ 再把描述改写成 generator-aware Prompt
```

但本项目只保留 Motion-relevant 信息。

允许：

```text
action
body part
direction
speed
gait
motion style
repetition
transition
motion-relevant interaction
```

禁止无关扩写：

```text
clothing
color
lighting
cinematic style
background description
camera language
irrelevant atmosphere
```

---

## 5.9.1 无 Retrieval 的 Caption

第一次编译：

```text
Motion DSL
→ Instruction-based Rewrite
→ Draft GEM Caption
```

例如：

```json
{
  "action": "walk",
  "direction": "forward",
  "speed": "slow",
  "style": [
    "limping",
    "asymmetric"
  ]
}
```

输出：

```text
a person slowly walks forward with an asymmetric limping gait
```

---

## 5.9.2 有 Retrieval 的 Caption

如果 State 中已经存在 Planner 获取的 Retrieved Caption：

```text
Motion DSL
+
Retrieved HumanML3D-style Captions
→ Retrieval-grounded Rewrite
→ Refined GEM Caption
```

例如：

```text
Retrieved:
"a person walks with a limp"
"a person moves forward with an uneven gait"

DSL:
walk + forward + limping + asymmetric

↓

"a person walks forward with an asymmetric limping gait"
```

禁止：

```text
照抄 unrelated retrieved semantics
加入 reference 中但用户没有要求的动作
```

---

## 5.9.3 Caption Candidate

第一版可以生成 2 个候选：

```text
Candidate A
→ Retrieval-grounded

Candidate B
→ Instruction-only
```

选择规则：

```text
semantic preservation
+
training-caption similarity
+
format validity
-
hallucination
-
excessive length
```

第一版不要求训练专门 Selector。

---

# 5.10 Caption Validation

Caption 进入 GEM 前必须执行 Validator。

检查：

```text
语义是否覆盖 DSL
是否遗漏 repetition
是否遗漏 body part
是否改变 direction
是否增加不存在的动作
是否包含无关视觉信息
是否过长
```

当前 GEM 配置使用：

```text
T5-3B
max_text_len = 50
encoded_text_dim = 1024
```

因此第一版要求：

```text
target:
<= 30~35 T5 tokens

hard:
不得超过 50 tokens
```

如果超过目标长度：

```text
compress caption
```

不能直接依赖 tokenizer 静默截断。

---

# 5.11 GEM Text Compiler

这是 Motion Compiler 与 GEM Adapter 的直接接口。

输出：

```python
class GEMTextCondition(BaseModel):
    captions: list[str]

    window_start: list[float]
    window_end: list[float]

    total_frames: int
```

---

## 5.11.1 时间窗口编译

对于：

```text
duration = 6s
fps = 30
total_frames = 180
```

Motion Plan：

```text
0-3s
3-4s
4-6s
```

转换成：

```text
0-90
90-120
120-180
```

再转换成 GEM 的 normalized window：

```python
window_start = start_frame / total_frames
window_end = end_frame / total_frames
```

得到：

```json
{
  "captions": [
    "a person walks forward with an asymmetric limping gait",
    "the person reaches forward with the right arm",
    "the person turns and slowly sits down"
  ],
  "window_start": [
    0.0,
    0.5,
    0.6667
  ],
  "window_end": [
    0.5,
    0.6667,
    1.0
  ],
  "total_frames": 180
}
```

---

# 5.12 与 GEM 的实际接口

GEM 当前支持：

```python
multi_text_data = {
    "vid": [],
    "caption": [],
    "text_ind": [],
    "window_start": [],
    "window_end": [],
}
```

因此 `gem_text_compiler.py` 负责把 `GEMTextCondition` 转成：

```python
def build_multi_text_data(condition: GEMTextCondition):

    return {
        "vid": [
            f"segment_{i}"
            for i in range(len(condition.captions))
        ],

        "caption": condition.captions,

        "text_ind": list(
            range(len(condition.captions))
        ),

        "window_start": torch.tensor(
            condition.window_start,
            dtype=torch.float32,
        ),

        "window_end": torch.tensor(
            condition.window_end,
            dtype=torch.float32,
        ),
    }
```

最终：

```text
Motion Compiler
      ↓
GEMTextCondition
      ↓
GemAdapter
      ↓
multi_text_data
      ↓
GEM
```

---

# 5.13 GEM 内部 Text 路径

当前 GEM Text Conditioning 路径保持不修改：

```text
Segment Caption
      ↓
T5 Tokenizer
      ↓
Frozen T5-3B
      ↓
1024-D Text Embedding
      ↓
GEM Text Projection
      ↓
Cross-Attention
      ↓
Motion Tokens
```

第一阶段：

```text
不替换 T5
不训练新的 Text Encoder
不修改 GEM Cross-Attention
```

Motion Compiler 只负责在 GEM 之前提供更好的：

```text
Caption
+
Temporal Window
```

---

## 5.13.1 Multi-Text Window

GEM 会将：

```text
window_start
window_end
```

转换为 Motion Frame 区间。

每个 Text Caption 只对自己的 Motion 时间窗口提供 Cross-Attention Condition。

因此：

```text
Timeline Compiler
```

不是纯 Agent Metadata。

它会直接决定：

```text
哪一段 Motion Frame
读取哪一个 Text Caption
```

这是 Motion Compiler 与 GEM 的主要连接点。

---

# 5.14 Compiler 输出

主输出：

```python
class CompilerResult(BaseModel):

    motion_spec: MotionSpecification

    gem_text_condition: GEMTextCondition

    routing_hints: list[ControlIntent]

    changed_segments: list[int]

    warnings: list[str]
```

其中：

```python
class MotionSpecification(BaseModel):

    original_request: str

    duration_s: float
    fps: int
    total_frames: int

    segments: list[MotionSegment]

    control_intents: list[ControlIntent]
```

---

# 5.15 Planner 如何使用 CompilerResult

Compiler 执行完成后：

```python
state.motion_plan = result.motion_spec
state.gem_text_condition = result.gem_text_condition
state.routing_hints = result.routing_hints
```

然后返回 Planner。

Planner 决策示例：

### 只有 Text

```text
routing_hints = []
→ GENERATE
```

### Rare Motion

```text
retrieval_candidate
→ RETRIEVE_REFERENCE
```

### Contact

```text
constraint_candidate
→ BUILD_CONSTRAINT
```

### Whole-body State

```text
keyframe_candidate
→ BUILD_KEYFRAME
```

如果 Retrieval 执行完成：

```text
Retrieved Reference
→ State
→ Planner
→ COMPILE_MOTION(mode="revise", focus=["gem_caption"])
```

再得到 refined caption。

---

# 5.16 Semantic Parser Prompt

第一版固定使用以下约束：

```text
You are the Motion Specification Compiler.

Convert the user's motion request into a structured human-motion specification.

Do not generate SMPL parameters.
Do not calculate numerical constraints.
Do not call external tools.
Do not invent actions, repetitions, targets, body parts,
or timing not supported by the user request.

Decompose sequential actions into temporal segments.

Keep simultaneous body actions inside the same segment.

Represent:
- action
- body parts
- direction
- speed
- style
- repetition
- orientation
- transition
- interaction
- duration weight

Preserve explicit timing exactly.

When timing is not explicit, output relative duration weights
instead of arbitrary absolute timestamps.

Identify requirements that may need:
- external retrieval
- numerical constraint
- whole-body keyframe

but do not execute those operations.

Return only the required structured schema.
```

---

# 5.17 Caption Optimizer Prompt

第一版固定使用：

```text
You are the GEM Motion Caption Optimizer.

Convert one structured human-motion segment into a concise
caption suitable for GEM's text-conditioned human motion generation.

Describe only motion-relevant information.

Prioritize:
- action
- body part
- direction
- speed
- gait or motion style
- repetition
- transition
- motion-relevant interaction

Do not add:
- clothing
- color
- lighting
- cinematic style
- background appearance
- camera language
- unrelated emotions
- actions not present in the Motion DSL

Use simple action-centered language similar to HumanML3D captions.

Preserve all required motion semantics.

Keep the caption concise and below GEM's text-token limit.
```

如果存在 Retrieved Caption：

```text
Retrieved training-style caption examples:

1. ...
2. ...
3. ...

Use these examples only as linguistic and motion-description references.
Do not copy unrelated actions or introduce new semantics.
```

---

# 5.18 Validators

必须实现：

```python
validate_schema()
validate_timeline()
validate_motion_semantics()
validate_caption_length()
validate_segment_coverage()
validate_control_intents()
validate_temporal_constraints()
```

---

## `validate_schema()`

检查：

```text
enum 合法
required field 存在
segment_id 唯一
repetition > 0
temporal_constraint.count > 0 when present
temporal_constraint uses repetition/cycle for discrete action cycles
duration_weight > 0
```

---

## `validate_timeline()`

检查：

```text
Segment 无非法重叠
Timeline 覆盖完整 Duration
Frame Boundary 连续
最后一个 End Frame 正确
```

---

## `validate_motion_semantics()`

检查：

```text
Caption 未丢失主要 Action
Caption 未改变 Body Part
Caption 未改变 Direction
Caption 未改变 Repetition
Caption 未丢失 structured temporal_constraint / temporal_mode
Caption 未添加新 Motion
```

---

## `validate_caption_length()`

检查 T5 token length。

如果超过：

```text
target token limit
```

则触发一次 deterministic / LLM compression。

---

## 5.18.1 `validate_heading_continuity()`

Compiler Validator 新增确定性检查：

```text
1. inherit_previous 必须引用存在且早于当前 Segment 的 anchor；
2. 用户显式 turn / rotate 时不得仍标记 inherit_previous；
3. follow_trajectory 只能在 trajectory intent 存在时启用；
4. explicit 必须有明确 facing 语义来源；
5. free 不得在 caption 中偷偷加入固定朝向；
6. continuation phrase 不得覆盖用户显式 orientation change。
```

Validator 不检查生成 Motion 是否真的保持朝向；那属于 Tournament / Verifier。

# 5.19 内部代码结构

建议：

```text
compiler/
│
├── motion_compiler.py
│
├── schemas.py
│
├── semantic_parser.py
│
├── temporal_resolver.py
│
├── motion_dsl.py
│
├── timeline_compiler.py
│
├── control_intent.py
│
├── caption_optimizer.py
│
├── gem_text_compiler.py
│
└── validators.py
```

职责：

```text
motion_compiler.py
→ 主入口与流程编排

schemas.py
→ Pydantic Schema

semantic_parser.py
→ Natural Language → Structured Semantics

temporal_resolver.py
→ repetition / continuous / alternation semantics → structured temporal fields

motion_dsl.py
→ Normalize / Canonicalize

timeline_compiler.py
→ 时间与 Frame 计算

control_intent.py
→ Retrieval / Constraint / Keyframe Candidate

caption_optimizer.py
→ GEM-aware Caption Rewrite

gem_text_compiler.py
→ GEMTextCondition / multi_text_data

validators.py
→ 全部确定性检查
```

---

# 5.20 主执行伪代码

```python
def compile_motion(
    state,
    mode,
    target_segments=None,
    focus=None,
):

    semantic_plan = semantic_parser.parse(
        request=state.user_request,
        existing_plan=state.motion_plan,
        mode=mode,
        target_segments=target_segments,
        focus=focus,
        diagnosis=state.diagnosis,
    )

    semantic_plan = temporal_resolver.resolve(
        request=state.user_request,
        semantic_plan=semantic_plan,
    )

    dsl_plan = motion_dsl.normalize(
        semantic_plan
    )

    timeline = timeline_compiler.compile(
        dsl_plan,
        total_duration=state.task.target_duration,
        fps=state.task.fps,
    )

    control_intents = detect_control_intents(
        timeline
    )

    captions = []

    for segment in timeline.segments:

        retrieved_context = get_retrieved_context(
            state=state,
            segment_id=segment.segment_id,
        )

        caption = caption_optimizer.optimize(
            segment=segment,
            retrieved_context=retrieved_context,
        )

        captions.append(caption)

    validate_schema(
        timeline
    )

    validate_timeline(
        timeline
    )

    validate_motion_semantics(
        original_request=state.user_request,
        timeline=timeline,
        captions=captions,
    )

    captions = enforce_caption_length(
        captions
    )

    gem_text_condition = gem_text_compiler.compile(
        timeline=timeline,
        captions=captions,
    )

    return CompilerResult(
        motion_spec=build_motion_spec(
            state,
            timeline,
            control_intents,
        ),
        gem_text_condition=gem_text_condition,
        routing_hints=control_intents,
        changed_segments=get_changed_segments(...),
        warnings=[],
    )
```

---

# 5.21 第一版实现范围

V1 必须完成：

```text
Natural Language
→ Semantic Motion Plan
→ Human Motion DSL
→ Timeline
→ Control Intent
→ GEM Caption
→ GEMTextCondition
→ multi_text_data
```

V1 使用：

```text
Strong LLM
+
Strict Structured Output
+
Rule Validator
+
HumanML3D Caption Retrieval Context（如果 Planner 已调用 Retrieval）
```

V1 暂时不做：

```text
专门训练 Motion Compiler
复杂 Relation Graph
专门训练 RAPO-style Refactoring Model
Learned Caption Selector
```

---

# 5.22 单元测试要求

至少覆盖：

```text
Single Action

Sequential Actions

Simultaneous Body-Part Actions

Explicit Time

Implicit Time

Repeated Action

Rare Motion Style

Joint / Contact Requirement

Root Trajectory Requirement

Whole-Body Keyframe Requirement

Long Prompt

Ambiguous Prompt

Revise One Segment
```

示例：

```text
Input:
walk forward

Expected:
1 segment
no routing hint
```

```text
Input:
wave the right hand three times and sit down

Expected:
2 segments
repetition = 3
```

```text
Input:
at 3 seconds place the right hand on the table

Expected:
constraint_candidate
type = contact
body_part = right_wrist
time_s = 3
```

```text
Input:
walk with an unusual staggering gait

Expected:
style stored in DSL
retrieval_candidate = true
no numerical constraint automatically created
```

---

## 5.22.1 Cross-Segment Heading Continuity Tests

至少新增以下 deterministic fixture：

### Case A：自然继承

```text
Input:
walk forward, wave the right hand, then sit down

Expected:
segment 0 establishes forward heading
segment 1 = inherit_previous
segment 2 = inherit_previous
caption 不包含 turn
```

### Case B：显式转向

```text
Input:
walk forward, turn right, then sit down

Expected:
turn segment = explicit orientation change
sit segment inherits the NEW heading
```

### Case C：旋转动作

```text
Input:
walk forward, spin once, then stop

Expected:
spin segment 不被 inherit_previous 锁死
```

### Case D：trajectory

```text
Input:
walk along a circle while facing the tangent direction

Expected:
mode = follow_trajectory
```

### Case E：未知

如果没有可靠 heading anchor：

```text
Expected:
mode = free
```

不能为了提高稳定性而虚构固定朝向。

# 5.23 当前模块完成标准

Motion Compiler 完成后，必须能够稳定实现：

```text
User Request
      ↓
Human Motion DSL
      ↓
Timeline
      ↓
Routing Hints
      ↓
GEM-friendly Caption
      ↓
GEM multi_text_data
```

并保证：

```text
Compiler 只负责 Semantic Compilation。

是否 Retrieval / Constraint / Keyframe
由 Planner 决定。

数值 Constraint 的真正执行
由 Constraint Compiler 负责。

Retry / Regeneration
由 Planner 根据 Verify / Diagnosis 结果决定。
```

## 5.24 Cross-Segment Continuity Completion Addendum

在本次 continuity 扩展后，Motion Compiler 的完成标准还必须包括：

```text
无显式转向时，可稳定生成 heading inheritance；
显式 turn / spin / trajectory orientation 不被错误覆盖；
continuity cue 与 DSL 同源；
Compiler 不生成数值 yaw / tolerance；
heading continuity fixtures 全部通过。
```
