# MotionAgent for GEM：模块实现指南

> 本文件对应 `BUILD_KEYFRAME` 的独立实现规范。  
> 该模块位于 `07_constraint_compiler.md` 与 `08_gem_generation_tool.md` 之间，负责把“某个时间点必须出现完整 Whole-Body State”的要求编译成可供 GEM 使用的 Keyframe Condition，并生成与第 10 单元一致的 VerificationSpec。

# 7B. Keyframe Tool

> Phase 7 scope revision: retain `KeyframeSpec`, pose handles, hard masks and
> verification specifications through the generation pipeline. A valid input
> interface does not guarantee exact keyframe forcing by Frozen GEM. Forcing
> is deferred to Guided Generation; errors are retained as research metrics.

Keyframe Tool 的职责是：

> **把 Planner 已经确认的 Whole-Body State Requirement，转换成一个可执行、可验证、可与 Constraint Composer 合并的 KeyframeSpec。**

它是：

```text
Planner-controlled
+
Deterministic / Retrieval-assisted
+
GEM-aware
```

的 Tool / Structured Subgraph。

它不是 Planner，也不直接运行 GEM。

### LangGraph 执行边界

V1 中 Keyframe Tool 由 `keyframe` LangGraph node wrapper 调用。Source resolution、IK、GEM encode 与 VerificationSpec 属于 domain service；LangGraph 只负责 lifecycle/routing，Tool 不拥有全局控制权。

---

## 7B.1 在整体 Planner 中的定位

Keyframe Tool 只由：

```text
BUILD_KEYFRAME
```

Action 调用。

```text
Motion Compiler
      │
      └── keyframe_candidate
                │
                ▼
             Planner
                │
        BUILD_KEYFRAME
                │
                ▼
          Keyframe Tool
                │
             KeyframeSpec
                │
                ▼
          Update State
                │
                ▼
             Planner
                │
                ▼
             GENERATE
```

Repair 情况：

```text
Verifier
→ KEYFRAME_POSE_MISMATCH
→ Diagnosis
→ BUILD_KEYFRAME Proposal
→ Planner
→ BUILD_KEYFRAME(mode="replace")
```

---

# 7B.2 为什么 Keyframe 不能完全并入 Constraint

Constraint 主要描述：

```text
局部 Joint Target
Contact
Trajectory
Fixed Joint
局部 Pose
```

Keyframe 描述：

```text
某一时间点
完整 Whole-Body State
```

例如：

```text
最后必须稳定坐下
第 3 秒必须处于深蹲状态
动作边界必须出现完整起始姿势
```

如果把 Whole-Body Keyframe 拆成几十个局部 Constraint：

```text
Joint 1
Joint 2
Joint 3
...
```

会增加：

```text
接口复杂度
冲突概率
验证难度
```

因此语义层保持 Keyframe 独立。

底层可以复用：

```text
GEM Encoder
HardMotionCondition
Feature Mask
ConstraintComposer / ConditionAssembler
```

---

# 7B.3 参考方法

## GMD：Sparse Keyframe Guidance

Guided Motion Diffusion 指出，稀疏 spatial signal / keyframe 在 diffusion 中容易被忽略，因此需要显式的 imputation / guidance 机制。

本项目 V1 不重训练 GEM，也不直接复现 GMD 的 dense guidance 网络，而是利用 GEM 已存在的：

```text
observed_motion_3d
+
motion_mask_3d
```

做 hard keyframe anchoring。

后续如果单帧硬锚导致 transition 不自然，可以在 Guided Generation 中增加：

```text
pose reward
transition reward
```

而不是先改 GEM 权重。

---

## KeyMotion：Keyframe → Infill 思想

KeyMotion 强调关键 pose 对整段动作结构的重要性，并采用 keyframe-first / in-filling 的生成思想。

MotionAgent 不训练 KeyMotion 模型，但借鉴：

```text
先明确关键 whole-body pose
再让生成模型完成中间 motion
```

这正对应：

```text
KeyframeSpec
→ GEM inpainting / conditioned generation
```

---

## Sparse Keyframe Motion Generation

ICCV 2025 的 sparse-keyframe motion work进一步说明：

```text
少量关键状态
```

可以作为高信息密度的 Motion Control Representation。

本项目因此把 Keyframe 作为独立 Control Primitive，而不是仅仅作为 Prompt 文本。

---

# 7B.4 V1 设计原则

V1 不训练任何新的 Keyframe Network。

使用：

```text
Reference Pose
Existing Candidate Pose
IK-refined Pose
+
GEM hard conditioning
```

完成 Keyframe Control。

---

# 7B.5 输入接口：`KeyframeRequest`

主接口：

```python
def build_keyframe(
    state: MotionAgentState,
    request: KeyframeRequest,
) -> KeyframeBuildResult:
    ...
```

Schema：

```python
class KeyframeRequest(BaseModel):

    mode: Literal[
        "add",
        "replace",
        "remove",
    ]

    keyframe_id: str | None

    target_segment: int

    time: float

    description: str

    source_preference: Literal[
        "retrieval",
        "ik",
        "generation",
        "auto",
    ]

    reference_ids: list[str] = []

    source_candidate_id: str | None = None

    pose_targets: list[KeyframePoseTarget] = []

    control_root_orientation: bool = True

    control_root_translation: bool = False

    temporal_tolerance_frames: int | None = None
```

该 Schema 与第 3 单元 Planner Payload 保持一致：

```json
{
  "mode": "add",
  "time": 5.8,
  "description": "a stable seated whole-body pose",
  "source_preference": "retrieval"
}
```

Request Builder 负责补充：

```text
target_segment
reference_ids
candidate context
threshold profile
```

---

# 7B.6 `mode`

## add

新增 Keyframe。

## replace

替换已有 Keyframe。

必须提供：

```text
keyframe_id
```

旧 Keyframe 标记：

```text
superseded
```

而不是原地覆盖 Artifact。

## remove

删除当前 active Keyframe。

只更新 State，不需要重新构造 Tensor。

---

# 7B.7 Target Time

Planner 使用：

```text
time
```

单位：

```text
second
```

Time Resolver：

```python
target_frame = round(
    request.time * state.task.fps
)
```

必须满足：

```text
0 <= target_frame < total_frames
```

且默认属于：

```text
target_segment
```

---

# 7B.8 Temporal Tolerance

Keyframe 的语义通常不是：

```text
必须精确在 frame 174
```

而可能是：

```text
最后附近稳定坐下
```

因此 Verification 应允许：

```text
± N frames
```

第一版来源优先级：

```text
1. User explicit timing tolerance
2. Keyframe Request
3. Config default
```

统一配置：

```text
configs/keyframes.yaml
```

---

# 7B.9 `KeyframePoseTarget`

如果用户 / 上游已有明确 Pose Anchor，可传：

```python
class KeyframePoseTarget(BaseModel):
    body_part: str

    target_type: Literal[
        "position",
        "rotation",
        "pose_reference",
    ]

    values: dict

    coordinate_frame: str | None
```

这主要用于：

```text
IK refinement
```

而不是让 Planner 自己生成 joint rotation。

---

# 7B.10 Source Selection

Keyframe Tool 必须明确：

> **Keyframe 的关键问题不是“写一句描述”，而是“得到一个真实可执行 Whole-Body Pose”。**

V1 支持四种 source preference。

---

# 7B.11 `source_preference="retrieval"`

使用已有：

```text
RetrievedReference
```

优先：

```text
retrieval_type = pose
```

其次：

```text
retrieval_type = motion
```

从 Motion 中选择合适 Frame。

Keyframe Tool 不主动调用 Retrieval。

如果没有可用 Reference：

```text
status = needs_reference
```

并返回：

```text
routing_hint = RETRIEVE_REFERENCE
purpose = keyframe_source
```

由 Planner 决定下一步。

---

# 7B.12 `source_preference="generation"`

V1 中 `generation` 的定义是：

> **从已经存在的 MotionCandidate 中选取 / 修正一个 Pose。**

Keyframe Tool 不在内部重新调用 GEM。

必须存在：

```text
source_candidate_id
```

或当前：

```text
state.generation.champion_candidate_id
```

典型 Repair：

```text
当前 Candidate 最终坐姿接近目标
但还不够稳定
↓
抽取最佳 seated frame
↓
构造更明确 Keyframe
↓
下一轮 inpainting
```

---

# 7B.13 `source_preference="ik"`

IK 不能从一句：

```text
"deep squat"
```

凭空推导完整 Whole-Body Pose。

IK 适合：

```text
已有 Base Pose
+
少量明确 Pose Targets
```

例如：

```text
retrieved sitting pose
+
right wrist target
+
pelvis height target
```

流程：

```text
Base Pose
→ IK refinement
→ FK validate
→ Whole-Body Pose
```

如果没有 Base Pose：

```text
status = needs_reference
```

而不是强行从 T-pose 求一个复杂 Keyframe。

---

# 7B.14 `source_preference="auto"`

V1 推荐优先级：

```text
1. explicit pose/reference
2. existing retrieved pose
3. retrieved motion frame
4. existing generated candidate frame
5. IK refine reliable base pose
```

如果全部不可用：

```text
needs_reference
```

不允许 LLM 直接生成 SMPL 参数。

---

# 7B.15 Keyframe Source Resolver

接口：

```python
class KeyframeSourceResolver:

    def resolve(
        self,
        state,
        request,
    ) -> KeyframeSourceResult:
        ...
```

输出：

```python
class KeyframeSourceResult(BaseModel):
    status: Literal[
        "resolved",
        "needs_reference",
        "needs_candidate",
        "unsupported",
    ]

    source_type: str | None

    pose_handle: str | None

    source_reference_ids: list[str]

    source_candidate_id: str | None

    confidence: float | None
```

---

# 7B.16 从 Retrieved Motion 选 Pose

如果 Retrieval 返回整段 Motion：

```text
不要默认使用最后一帧。
```

例如：

```text
"stable seated pose"
```

应该在 Retrieved Motion 中找到：

```text
最符合 seated + low velocity
```

的 Frame。

V1 可用简单：

```text
semantic phase hint
+
joint velocity
+
root velocity
```

选择。

---

# 7B.17 Stable Pose Selector

对于：

```text
stable seated
stable standing
hold pose
```

定义：

```text
pose semantic similarity
+
low local joint velocity
+
low root velocity
```

接口：

```python
def select_stable_pose_frame(
    motion_handle,
    target_description,
    candidate_window=None,
) -> PoseFrameSelection:
    ...
```

---

# 7B.18 Candidate Pose Selection

从已生成 Candidate 中选 Pose 时：

```text
Keyframe Tool
```

读取：

```text
pred_body_params_global
```

而不是 incam pose。

原因：

```text
Pure Text Camera 是 synthetic camera
```

核心 Motion Control 应使用 global representation。

---

# 7B.19 Keyframe Canonical Pose

最终必须构造：

```python
class CanonicalKeyframePose(BaseModel):
    body_pose_handle: str

    global_orient_handle: str

    transl_handle: str | None

    betas_handle: str | None

    source_type: str
```

实际 Tensor 保存在 Keyframe Store / Artifact Store。

---

# 7B.20 Body Shape

Keyframe 默认：

```text
不控制 betas
```

因为 Keyframe 表示：

```text
Pose State
```

不是：

```text
Character Identity / Shape
```

如果是 Segment Regeneration：

```text
betas
```

由第 8 单元 identity preservation 机制固定。

---

# 7B.21 Root Orientation

Whole-Body Keyframe 默认：

```text
control_root_orientation = true
```

例如：

```text
turn around and sit
```

最终坐姿方向是 whole-body state 的一部分。

---

# 7B.22 Root Translation

默认：

```text
control_root_translation = false
```

原因：

```text
"stable seated pose"
```

并不一定指定世界坐标。

只有：

```text
明确 world / scene position
reference position required
```

才固定 root translation。

这能减少 over-constraint。

---

# 7B.23 输出接口：`KeyframeBuildResult`

```python
class KeyframeBuildResult(BaseModel):

    status: Literal[
        "compiled",
        "needs_reference",
        "needs_candidate",
        "conflict",
        "unsupported",
        "error",
    ]

    keyframe: KeyframeSpec | None

    warnings: list[str]

    routing_hints: list[str]
```

---

# 7B.24 `KeyframeSpec`

```python
class KeyframeSpec(BaseModel):

    keyframe_id: str

    target_segment: int

    target_time_s: float
    target_frame: int

    temporal_tolerance_frames: int

    description: str

    source_type: Literal[
        "retrieval_pose",
        "retrieval_motion",
        "candidate",
        "ik_refined",
        "explicit_pose",
    ]

    source_reference_ids: list[str]
    source_candidate_id: str | None

    pose_handle: str

    control_body_pose: bool
    control_root_orientation: bool
    control_root_translation: bool

    hard_condition_handle: str

    verification_spec: KeyframeVerificationSpec

    status: Literal[
        "active",
        "superseded",
        "removed",
    ]

    metadata: dict
```

---

# 7B.25 `KeyframeVerificationSpec`

```python
class KeyframeVerificationSpec(BaseModel):

    keyframe_id: str

    target_frame: int
    temporal_tolerance_frames: int

    pose_handle: str

    metrics: list[
        Literal[
            "joint_rotation_error",
            "joint_position_error",
            "root_orientation_error",
            "root_position_error",
        ]
    ]

    rotation_threshold_deg: float | None
    position_threshold_m: float | None

    control_root_orientation: bool
    control_root_translation: bool
```

第 10 单元 Keyframe Verifier 必须直接读取该 Spec。

---

# 7B.26 GEM Keyframe 编译

Keyframe Tool 不直接拼 151-D。

正确流程：

```text
CanonicalKeyframePose
      ↓
SMPL Params
      ↓
GEM EnDecoder.encode()
      ↓
151-D normalized feature
      ↓
FeatureMaskBuilder
      ↓
HardMotionCondition
```

与第 7 单元使用同一：

```text
GEMConstraintEncoder
GEMFeatureMapper
```

---

# 7B.27 Keyframe Feature Mask

目标 Frame：

```text
t = target_frame
```

默认 Mask：

```text
body_pose = 1

global_orient = 1

global_orient_gv = 1

betas = 0

local_transl_vel = 0
```

如果：

```text
control_root_translation = true
```

则不能仅 mask：

```text
local_transl_vel at one frame
```

因为 translation 是序列积分结果。

这类 Keyframe 应优先：

```text
root position RewardSpec
```

或与：

```text
root trajectory constraint
```

共同编译。

V1 不伪造单帧 local velocity 来代表绝对位置。

---

# 7B.28 为什么 Root Position 不能简单 Hard Mask

GEM 表示中 root translation 使用：

```text
local_transl_vel
```

而不是：

```text
absolute xyz
```

所以：

```text
某帧 pelvis 必须在 (x,y,z)
```

更接近：

```text
Constraint Compiler root / joint position target
```

而不是简单 Keyframe Mask。

因此 Keyframe Tool 的核心是：

```text
whole-body pose state
```

不是任意世界坐标状态。

---

# 7B.29 HardMotionCondition

最终：

```python
class HardMotionCondition:
    values: torch.Tensor  # [L, 151]
    mask: torch.Tensor    # [L, 151]
```

只有：

```text
target frame
```

对应 Pose Features 为 1。

其余 Frame：

```text
mask = 0
```

---

# 7B.30 Anchor Window

V1 不建议把目标 Pose 在：

```text
±10 frames
```

全部 hard freeze。

否则：

```text
motion 会停顿
```

推荐：

```text
单帧 / 极窄 anchor
+
文本 transition
```

如果 transition 不稳定：

```text
Phase II
→ pose similarity soft reward around target
```

---

# 7B.31 多 Keyframe

系统允许：

```text
多个 Keyframe
```

例如：

```text
start pose
mid pose
end pose
```

每个生成独立：

```text
HardMotionCondition
```

进入：

```text
ConditionAssembler
```

统一合并。

---

# 7B.32 Keyframe 与 Constraint 冲突

例如：

```text
Keyframe:
right arm down at frame 90

Constraint:
right wrist at high target at frame 90
```

必须检测：

```text
same frame
same features / kinematic target
incompatible values
```

禁止：

```text
last write wins
```

返回：

```text
status = conflict
```

交回 Planner。

---

# 7B.33 Keyframe Condition Composer

Keyframe 与 Constraint 统一进入：

```text
GenerationConditionAssembler
```

建议底层共享：

```text
constraints/composer.py
```

输入：

```text
Hard Conditions from Constraints
Hard Conditions from Keyframes
```

输出唯一：

```text
observed_motion_3d
motion_mask_3d
```

---

# 7B.34 与第 8 单元 Generation 的接口

Generator 不读取：

```text
Keyframe description
```

只读取：

```text
hard_motion_condition_handle
```

以及用于 metadata 的：

```text
active_keyframe_ids
```

因此：

```text
Keyframe Tool
→ Compile once
→ Generator consume
```

---

# 7B.35 与第 10 单元 Verifier 的接口

Verifier 读取：

```text
KeyframeVerificationSpec
```

在：

```text
target_frame ± temporal_tolerance
```

范围内寻找最佳匹配 Frame。

计算：

```text
joint rotation error
joint position error
root orientation error（如启用）
```

绝不通过：

```text
重新理解 description
```

来决定是否通过。

---

# 7B.36 Keyframe Pass Rule

示例：

```text
rotation error <= configured threshold
AND
position error <= configured threshold
```

如果：

```text
control_root_translation = false
```

则 root position 不进入 pass rule。

---

# 7B.37 与 Retrieval 的连接

典型链路：

```text
Motion Compiler
→ keyframe_candidate
→ Planner
→ RETRIEVE_REFERENCE(
     type="pose",
     purpose="keyframe_source"
  )
→ State
→ Planner
→ BUILD_KEYFRAME
→ Keyframe Tool
```

Keyframe Tool 不自己调用 Retrieval。

---

# 7B.38 与 Diagnosis 的连接

第 11 单元出现：

```text
KEYFRAME_POSE_MISMATCH
```

可产生：

```text
BUILD_KEYFRAME(mode="replace")
```

Proposal。

Keyframe Tool 再决定：

```text
重新选择 Reference
重新 IK refine
重新生成 Hard Condition
```

---

# 7B.39 Keyframe Store

大型 Pose Tensor 不进入 State。

```text
keyframes/store.py
```

保存：

```text
CanonicalKeyframePose
SMPL Pose Tensor
Encoded GEM Pose
HardMotionCondition
```

State 保存：

```text
KeyframeSummary
```

---

# 7B.40 Source Provenance

Keyframe 必须记录来源：

```text
reference ID
candidate ID
frame ID
IK settings
```

这样可以调试：

```text
为什么这个 Keyframe 很奇怪？
```

---

# 7B.41 `KeyframeMetadata`

建议：

```python
class KeyframeMetadata(BaseModel):
    source_type: str

    source_reference_ids: list[str]
    source_candidate_id: str | None
    source_frame: int | None

    ik_used: bool
    ik_residual_m: float | None

    encoder_version: str

    config_version: str
```

---

# 7B.42 IK Refinement

如果 Source Pose 接近目标，但有明确 Joint Anchor：

```text
reference pose
→ CCD IK
→ FK validation
```

复用第 7 单元：

```text
solve_pose_ik()
```

禁止维护第二套 IK。

---

# 7B.43 IK 失败

如果：

```text
residual > tolerance
```

则：

```text
status = needs_reference
```

或保留原 Reference Pose，附 warning。

不允许：

```text
失败 IK
→ hard keyframe
```

---

# 7B.44 Keyframe 质量预检查

在编译为 Hard Condition 前：

```text
FK valid
no NaN / Inf
joint rotations valid
pose dimensions correct
source frame valid
```

可额外检查：

```text
gross ground penetration
extreme joint outlier
```

避免坏 Reference 进入 GEM。

---

# 7B.45 输出状态写回

成功：

```python
state.conditions.keyframes.append(
    KeyframeSummary(...)
)
```

正式实现通过：

```text
State Reducer
```

更新，而不是 Tool 自己直接 mutate State。

旧 replace target：

```text
status = superseded
```

---

# 7B.46 Planner Skill Card

建议：

```text
planner_skills/keyframe.md
```

内容：

```text
WHEN TO USE

Use BUILD_KEYFRAME when a specific time or motion boundary
requires a meaningful whole-body state.

Examples:
- final stable seated pose;
- deep squat at a specific time;
- whole-body turning pose at a boundary.

DO NOT USE

Do not use for:
- one local joint target;
- contact only;
- root trajectory;
- repetition;
- general style.

SOURCE

Prefer:
1. explicit / retrieved pose;
2. retrieved motion frame;
3. existing candidate pose;
4. IK refinement on a reliable base pose.

Never invent SMPL parameters from text.

IMPORTANT

The Keyframe Tool does not run GEM and does not call Retrieval.
```

---

# 7B.47 内部代码结构

```text
keyframes/
│
├── schemas.py
├── compiler.py
├── source_resolver.py
├── pose_selector.py
├── candidate_pose.py
├── ik_refiner.py
├── gem_encoder.py
├── feature_mask.py
├── verifier_spec.py
├── validators.py
└── store.py
```

其中：

```text
gem_encoder.py
feature_mask.py
ik_refiner.py
```

优先调用：

```text
constraints/
```

共享实现，而不是复制。

---

# 7B.48 主流程伪代码

```python
def build_keyframe(
    state,
    request,
):

    validate_request(
        state,
        request,
    )

    if request.mode == "remove":
        return remove_keyframe(...)

    target_frame = resolve_target_frame(
        request.time,
        fps=state.task.fps,
    )

    source = source_resolver.resolve(
        state=state,
        request=request,
    )

    if source.status != "resolved":
        return KeyframeBuildResult(
            status=source.status,
            keyframe=None,
            routing_hints=build_routing_hints(
                source
            ),
            warnings=[],
        )

    pose = keyframe_store.load_pose(
        source.pose_handle
    )

    if request.pose_targets:
        pose = refine_pose_with_ik(
            pose=pose,
            targets=request.pose_targets,
        )

    validate_pose(
        pose
    )

    encoded_pose = gem_encoder.encode_keyframe_pose(
        pose=pose,
        adapter_context=
            state.gem_adapter_context,
    )

    hard_condition = build_keyframe_hard_condition(
        encoded_pose=encoded_pose,
        target_frame=target_frame,
        total_frames=state.task.total_frames,
        control_root_orientation=
            request.control_root_orientation,
        control_root_translation=
            request.control_root_translation,
    )

    condition_handle = store_hard_condition(
        hard_condition
    )

    verification_spec = build_keyframe_verification_spec(
        request=request,
        pose_handle=source.pose_handle,
        target_frame=target_frame,
    )

    keyframe = KeyframeSpec(
        ...,
        hard_condition_handle=condition_handle,
        verification_spec=verification_spec,
    )

    return KeyframeBuildResult(
        status="compiled",
        keyframe=keyframe,
        warnings=[],
        routing_hints=[],
    )
```

---

# 7B.49 Keyframe Hard Condition 伪代码

```python
def build_keyframe_hard_condition(
    encoded_pose,
    target_frame,
    total_frames,
    control_root_orientation,
    control_root_translation,
):

    values = torch.zeros(
        total_frames,
        151,
    )

    mask = torch.zeros(
        total_frames,
        151,
    )

    values[target_frame] = encoded_pose

    body_slice = feature_mapper.get_group_slice(
        "body_pose"
    )

    mask[
        target_frame,
        body_slice[0]:body_slice[1]
    ] = 1

    if control_root_orientation:

        for feature in [
            "global_orient",
            "global_orient_gv",
        ]:
            s, e = feature_mapper.get_group_slice(
                feature
            )
            mask[target_frame, s:e] = 1

    # absolute root translation is not represented as a
    # direct single-frame feature; handle separately.

    return HardMotionCondition(
        values=values,
        mask=mask,
    )
```

---

# 7B.50 V1 实现顺序

## Step 1

实现 `KeyframeRequest / KeyframeSpec / Result`。

## Step 2

实现 Retrieved Pose / Motion source resolver。

## Step 3

实现 Existing Candidate pose extraction。

## Step 4

复用 GEM Encoder 构造单帧 hard pose mask。

## Step 5

实现 VerificationSpec。

## Step 6

实现 multi-keyframe conflict detection。

## Step 7

增加 IK refinement。

## Step 8

增加 Repair / replace / remove。

---

# 7B.51 V1 最低实现范围

必须：

```text
KeyframeRequest

add / replace / remove

retrieved pose source

retrieved motion frame source

existing candidate source

GEM pose encode

single-frame whole-body hard mask

root orientation control

KeyframeVerificationSpec

Keyframe Store

Constraint / Keyframe conflict detection
```

可以暂缓：

```text
learned pose generator

keyframe-specific diffusion model

dense GMD guidance

physics-aware keyframe optimization

absolute root-position hard keyframe
```

---

# 7B.52 单元测试要求

至少：

```text
Add Keyframe

Replace Keyframe

Remove Keyframe

Retrieved Pose Source

Retrieved Motion Frame Source

Generated Candidate Source

No Reference

IK Refinement Success

IK Refinement Failure

Target Frame Boundary

Root Orientation On / Off

Body Shape Not Masked

Constraint Conflict

Multiple Keyframes

GEM Encode / Decode Pose Consistency

VerificationSpec Correct
```

---

# 7B.53 评估指标

模块级：

```text
Keyframe Compile Success Rate

Source Resolution Success Rate

IK Success Rate

IK Residual

Keyframe Conflict Rate

Compilation Latency
```

Generation 后：

```text
Keyframe Joint Rotation Error

Keyframe Joint Position Error

Temporal Match Offset

Keyframe Satisfaction Rate

Naturalness Regression After Keyframe
```

---

# 7B.54 模块完成标准

Keyframe Tool 完成后必须稳定实现：

```text
Planner BUILD_KEYFRAME
        ↓
KeyframeRequest
        ↓
Source Resolution
        ↓
Whole-Body Pose
        ↓
Optional IK Refinement
        ↓
GEM Encode
        ↓
HardMotionCondition
        ↓
KeyframeVerificationSpec
        ↓
KeyframeSpec
        ↓
State
        ↓
Planner
```

并保证：

```text
Keyframe Tool 不调用 GEM。

Keyframe Tool 不主动调用 Retrieval。

Keyframe Tool 不把一句自然语言直接变成 SMPL 数值。

Whole-Body Keyframe 与局部 Constraint 职责分开。

Keyframe 与 Constraint 在底层共享 Hard Condition Infrastructure。

Generator 与 Verifier 使用同一个 KeyframeSpec。

V1 不需要重新训练 GEM。
```
