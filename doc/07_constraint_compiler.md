# MotionAgent for GEM：模块实现指南

> 本文件从总实现指南中拆分，对应第 7 单元。内容保持模块边界独立，其输入输出接口需与相邻模块的 Schema 保持一致。

# 7. Constraint Compiler

本节定义 Constraint Compiler 的**实现规范**。

Constraint Compiler 的职责是：

> **将 Planner 已经确认需要显式控制的 Motion Requirement，编译成可执行的数值约束，并生成统一的 Generation / Guidance / Verification Constraint Specification。**

Constraint Compiler 是：

```text
Planner-controlled
+
Deterministic / Numerical
+
GEM-aware
```

的 Tool / Structured Subgraph。

它不是顶层 Agent，不负责决定“是否应该使用 Constraint”。

### LangGraph 执行边界

V1 中 Constraint Compiler 由 `constraint` LangGraph node wrapper 调用。FK / IK / 151-D encoding / RewardSpec / VerificationSpec 都保持 framework-agnostic；node 仅负责 request/commit lifecycle，并在完成后返回 Planner。

---

## 7.1 在整体 Planner 中的定位

Constraint 只能由 Planner 的：

```text
BUILD_CONSTRAINT
```

Action 调用。

典型链路：

```text
Motion Compiler
      │
      └── constraint_candidate
                │
                ▼
             Planner
                │
        BUILD_CONSTRAINT
                │
                ▼
       Constraint Compiler
                │
        CompiledConstraint
                │
                ▼
          Update State
                │
                ▼
             Planner
                │
        ┌───────┴────────┐
        ▼                ▼
    GENERATE         继续补条件
```

生成失败后的修改链路：

```text
GEM
 ↓
Tournament
 ↓
Verifier
 ↓
Diagnosis:
constraint failure
 ↓
Planner
 ↓
BUILD_CONSTRAINT(
    mode="replace"
)
 ↓
Constraint Compiler
 ↓
Updated Constraint
 ↓
Planner
 ↓
GENERATE
```

Constraint Compiler 执行完成后，不自动调用 GEM，也不自动 Retry。

---

# 7.2 职责边界

Constraint Compiler 负责：

```text
解析 Planner 的结构化 ConstraintRequest
检查目标是否可数值化
解析时间区间
解析坐标系
解析 body part / joint
加载必要的 Retrieved Reference
执行 FK / IK
构造 Root Trajectory
构造 Contact / Fixed-Joint 数值目标
决定 Hard Condition / Soft Reward 的执行表示
将可硬约束部分编码到 GEM 151-D Motion Space
生成 motion_mask_3d
生成 RewardSpec
生成 VerificationSpec
检查多个 Constraint 是否冲突
返回 CompiledConstraint
```

Constraint Compiler 不负责：

```text
判断是否应该使用 Constraint
修改用户语义
修改 GEM Caption
主动调用 Retrieval
主动调用 Keyframe Tool
执行 Whole-Body Keyframe Planning
运行 GEM
执行 Diffusion Guidance
执行 DNO Noise Optimization
执行 ReAlign
执行最终 Verify
执行 Retry
```

其中：

```text
是否需要 Constraint
→ Planner

Constraint 数值怎么构造
→ Constraint Compiler

Soft Constraint 怎么影响 Sampling
→ Guided Generator

Constraint 是否满足
→ Verifier
```

---

# 7.3 V1 Constraint 类型

为了和第 3 部分 Planner Action Space 保持一致，V1 正式支持：

```text
joint_target
body_part_pose
root_trajectory
contact
fixed_joint
```

暂时不把以下内容放进 V1 Constraint Compiler：

```text
Temporal Order
Action Frequency
General Motion Semantics
Whole-Body Keyframe
Generic Collision / Obstacle Avoidance
```

对应关系：

```text
Temporal Order / Frequency
→ Motion Compiler + Event Verifier

General Semantics
→ Motion Compiler

Whole-Body Keyframe
→ BUILD_KEYFRAME

复杂 Obstacle / Collision
→ Phase II Soft Constraint / Guided Generation
```

这样避免不同模块职责重叠。

---

# 7.4 参考方法的具体使用方式

## 7.4.1 Retrieval-Guided DNO

主要借鉴：

```text
Relational Task Parsing
→ 找出真正困难的 Constraint

Difficult Constraint
→ Retrieval Motion Prior

Constraint Reward
→ Training-free Optimization
```

本项目不把其全部流程放进 Constraint Compiler。

拆分为：

```text
Motion Compiler
→ constraint_candidate

Planner
→ BUILD_CONSTRAINT / RETRIEVE_REFERENCE

Retrieval Tool
→ Reference Motion

Constraint Compiler
→ Numerical Constraint / RewardSpec

Guided Generator
→ DNO-style / ReAlign-style optimization
```

因此 Constraint Compiler 只负责：

> **把约束定义成可计算、可执行、可验证的形式。**

DNO 的：

```text
retrieved-noise initialization
reward-guided mask
diffusion-noise optimization
```

属于后续 Guided Generator，不在本模块内部实现。

---

## 7.4.2 NEWTON Scientific Computation Tool

主要借鉴：

> LLM / Planner 决定需要什么控制，确定性程序计算具体数值。

在本项目中对应：

```text
Planner:
"right wrist contact"
      ↓
Constraint Compiler:
resolve target
solve IK
compute FK
build mask
```

或：

```text
Planner:
"root follows a circle"
      ↓
Constraint Compiler:
generate trajectory
compute orientation policy
encode root motion
```

因此 V1 不再让 LLM 自己生成：

```text
joint XYZ
rotation matrix
151-D tensor
trajectory samples
```

这些均由 Python / Torch Numerical Tool 生成。

---

# 7.5 输入接口：`ConstraintRequest`

主接口：

```python
def compile_constraint(
    state: MotionAgentState,
    request: ConstraintRequest,
) -> ConstraintCompileResult:
    ...
```

Schema：

```python
class ConstraintRequest(BaseModel):

    mode: Literal[
        "add",
        "replace",
        "remove",
    ]

    constraint_id: str | None

    target_segment: int

    constraint_type: Literal[
        "joint_target",
        "body_part_pose",
        "root_trajectory",
        "contact",
        "fixed_joint",
    ]

    requested_strength: Literal[
        "hard",
        "soft",
    ]

    time_spec: TimeSpec

    body_parts: list[str]

    target: ConstraintTarget | None

    reference_ids: list[str]

    tolerance: ConstraintTolerance | None

    priority: int = 0
```

---

## 7.5.1 `mode`

### `add`

新建 Constraint。

### `replace`

替换已经存在的 Constraint。

必须提供：

```text
constraint_id
```

### `remove`

删除 Constraint。

该操作不重新编译 Tensor，只更新 State / Constraint Store。

---

## 7.5.2 `target_segment`

必须是：

```text
state.motion_plan.segments
```

中的合法 Segment。

所有时间必须限制在对应 Segment 或合法全局 Timeline 中。

---

## 7.5.3 `requested_strength`

表示 Planner / 用户希望的控制强度：

```text
hard
soft
```

注意：

> `requested_strength="hard"` 不代表 Compiler 必须无条件生成 hard mask。

如果约束本身无法唯一转换成 Motion State，例如：

```text
right hand touches table
```

但没有：

```text
table geometry
target XYZ
reference motion
```

则不能伪造一个 Hard Condition。

Compiler 必须返回：

```text
needs_geometry
needs_reference
```

或编译成：

```text
effective_mode="guided_reward"
```

并明确记录原因。

禁止静默“猜一个坐标”。

---

# 7.6 时间接口：`TimeSpec`

所有 Constraint 必须显式绑定时间。

```python
class TimeSpec(BaseModel):

    type: Literal[
        "point",
        "window",
        "segment",
    ]

    time_s: float | None

    start_s: float | None
    end_s: float | None

    segment_relative: bool = False
```

---

## `point`

例如：

```text
第 3 秒右手到达目标
```

```json
{
  "type": "point",
  "time_s": 3.0
}
```

---

## `window`

例如：

```text
3.0s - 3.5s 保持接触
```

```json
{
  "type": "window",
  "start_s": 3.0,
  "end_s": 3.5
}
```

---

## `segment`

整个 Segment 有效。

例如：

```text
整个站立阶段左脚保持不动
```

---

## Time Resolver

统一转成 Frame：

```python
class ResolvedTimeSpec(BaseModel):

    frame_indices: list[int]

    start_frame: int
    end_frame: int
```

计算规则：

```python
frame = round(time_s * fps)
```

并强制：

```text
frame >= 0
frame < total_frames
frame 属于 target segment
```

---

# 7.7 Target Schema

不能在系统内部使用：

```text
target = "table_surface"
```

这种没有数值定义的字符串直接生成 Hard Constraint。

必须先规范成：

```python
class ConstraintTarget(BaseModel):

    target_type: Literal[
        "point",
        "trajectory",
        "plane",
        "reference_pose",
        "reference_motion",
        "relative_anchor",
        "self_anchor",
        "scene_anchor",
    ]

    coordinate_frame: Literal[
        "world",
        "root_local",
        "gravity_aligned",
        "scene_local",
    ]

    values: dict

    geometry_handle: str | None

    reference_id: str | None
```

---

# 7.8 Coordinate Resolver

所有数值 Constraint 必须经过统一坐标解析。

禁止各模块自行解释：

```text
x / y / z
```

建议由：

```text
constraints/coordinates.py
```

提供：

```python
resolve_point(...)
transform_world_to_root(...)
transform_root_to_world(...)
resolve_scene_anchor(...)
```

V1 统一使用：

> **GemAdapter 定义的 canonical motion coordinate system。**

纯 Text Generation 时：

```text
initial root
default static camera
gravity direction
```

必须和 GemAdapter 使用完全相同的定义。

---

## 7.8.1 Scene Object Target

例如：

```text
right hand touches table
```

如果系统只有 Human Motion，没有 Scene Geometry：

```text
table
```

本身不包含可执行 XYZ。

因此：

### 有 Scene Geometry

```text
scene_anchor
→ resolve table surface / contact point
→ numerical target
```

### 有 Retrieved Reference

```text
reference_motion
→ extract plausible wrist target / relative target
```

### 两者都没有

返回：

```text
status = needs_geometry
```

或者保留为：

```text
semantic / soft verification requirement
```

禁止 LLM 自行生成：

```text
table height = 0.8m
```

之类的假设并当成事实。

---

# 7.9 输出接口：`ConstraintCompileResult`

```python
class ConstraintCompileResult(BaseModel):

    status: Literal[
        "compiled",
        "compiled_soft",
        "needs_reference",
        "needs_geometry",
        "conflict",
        "unsupported",
        "error",
    ]

    constraint: CompiledConstraint | None

    warnings: list[str]

    routing_hints: list[str]
```

---

## 7.9.1 `CompiledConstraint`

```python
class CompiledConstraint(BaseModel):

    constraint_id: str

    target_segment: int

    constraint_type: str

    requested_strength: str

    effective_mode: Literal[
        "hard_condition",
        "selection_reward",
        "guided_reward",
    ]

    resolved_time: ResolvedTimeSpec

    target_spec: ConstraintTarget | None

    hard_condition_handle: str | None

    reward_spec: RewardSpec | None

    verification_spec: VerificationSpec

    source_reference_ids: list[str]

    tolerance: ConstraintTolerance

    priority: int

    metadata: dict
```

这是 Constraint 的：

> **Single Source of Truth**

同一个 `CompiledConstraint` 同时服务：

```text
Generation
Guidance
Verification
```

禁止三处分别重新解释 User Constraint。

---

# 7.10 三种执行模式

Constraint 编译完成后，最终映射为三种执行方式之一。

```text
hard_condition
selection_reward
guided_reward
```

---

## 7.10.1 `hard_condition`

适合：

```text
已知局部 Pose
已知 Joint Rotation
已知 Root Motion
已知 Reference Pose
可以通过 IK 得到稳定唯一解
```

输出：

```text
observed_motion_3d
+
motion_mask_3d
```

---

## 7.10.2 `selection_reward`

适合：

```text
约束可以计算
但不适合固定成唯一 Pose
```

例如：

```text
foot remains fixed
right wrist close to target plane
root trajectory
```

V1 即使还没有 Guided Sampling，也可以：

```text
GEM × K
→ evaluate constraint cost
→ Tournament / Verifier
→ select better candidate
```

因此 Soft Constraint 在 V1 仍然有实际作用。

---

## 7.10.3 `guided_reward`

用于：

```text
ReAlign-style Sampling Guidance
DNO-style Noise Optimization
```

Constraint Compiler 只生成：

```text
RewardSpec
```

真正计算 gradient / 修改 diffusion noise：

```text
Guided Generator
```

负责。

---

# 7.11 Hard vs Soft 的选择规则

推荐默认策略：

```text
如果目标可以唯一或近似唯一映射到 Motion Feature
→ hard_condition

如果存在大量合法 Pose 都满足目标
→ soft reward

如果强行 Hard Mask 会过度限制动作
→ soft reward

如果目标 geometry 未解析
→ 不允许 hard condition

如果 IK residual 太大
→ 不允许强制 hard condition

如果 Retrieved Motion 提供可靠 Pose
→ 可以升级为 hard condition
```

例如：

```text
"3 秒时使用这个 reference pose"
→ hard

"3 秒时右手在某个 XYZ"
→ IK 后可 hard / soft

"右手接触桌面"
→ 通常 soft

"左脚保持不动"
→ 通常 soft

"沿指定 world trajectory"
→ 视 root orientation 是否已解析
```

---

# 7.12 GEM Motion Representation

当前 GEM SMPL Motion Representation 为：

```text
151-D / frame
```

布局：

```text
0   : 126   body_pose
126 : 136   betas
136 : 142   global_orient
142 : 148   global_orient_gv
148 : 151   local_transl_vel
```

其中：

```text
body_pose
= 21 joints × 6D rotation
```

Constraint Compiler 不应手工拼接并归一化这 151 维。

正确流程：

```text
Constraint Target
      ↓
SMPL Parameters / Reference Sequence
      ↓
GEM EnDecoder.encode()
      ↓
normalized 151-D representation
      ↓
Feature Mask
```

原因：

```text
GEM 自己负责 rotation conversion
GEM 自己负责 local translational velocity
GEM 自己负责 normalization
避免坐标系和统计量不一致
```

---

# 7.13 GEM Feature Mapping

必须建立：

```python
GEMFeatureMap
```

使用 GEM 自身：

```python
endecoder.obs_indices_dict
```

当前映射：

```python
{
    "body_pose": (0, 126),
    "betas": (126, 136),
    "global_orient": (136, 142),
    "global_orient_gv": (142, 148),
    "local_transl_vel": (148, 151),
}
```

不要在多个文件中重复 hard-code。

统一封装：

```python
class GEMFeatureMapper:

    def get_group_slice(self, name):
        ...

    def get_body_joint_slice(self, smpl_joint_id):
        ...
```

对于 SMPL body joint：

```text
root joint = 0
body pose joint = 1 ... 21
```

对应 body_pose 6D：

```python
start = (joint_id - 1) * 6
end = start + 6
```

但正式实现必须通过：

```text
joint registry
```

解析 Joint Name，不允许业务代码直接写数字。

---

# 7.14 Joint Registry

建议统一：

```text
constraints/joints.py
```

维护：

```python
SMPLJointRegistry
```

需要至少支持：

```text
pelvis
left_hip
right_hip
left_knee
right_knee
left_ankle
right_ankle
left_foot
right_foot
left_shoulder
right_shoulder
left_elbow
right_elbow
left_wrist
right_wrist
```

当前 GEM 后处理代码已经使用：

```text
L_Ankle = 7
L_Foot = 10
R_Ankle = 8
R_Foot = 11
L_Wrist = 20
R_Wrist = 21
```

正式实现时统一从 Registry 读取。

---

# 7.15 Hard Motion Condition

Hard Condition 的内部表示：

```python
class HardMotionCondition:

    values: torch.Tensor
    # [L, 151]

    mask: torch.Tensor
    # [L, 151]

    source_constraint_id: str
```

其中：

```text
mask = 1
→ 该 Motion Feature 被强制固定

mask = 0
→ 由 GEM 自由生成
```

最终多个 Hard Constraint 合并为：

```text
observed_motion_3d
motion_mask_3d
```

---

# 7.16 GEM 当前 Hard Conditioning Hook

GEM denoiser 当前已经存在：

```python
if motion_mask_3d is not None:
    xt = (
        xt * (1 - motion_mask_3d)
        + observed_motion_3d * motion_mask_3d
    )
```

因此每次 denoiser forward 时：

```text
mask == 1
```

的维度会被 `observed_motion_3d` 覆盖。

这允许：

```text
sparse frame condition
partial body condition
root-motion condition
```

而不需要第一阶段重新训练 GEM。

---

# 7.17 GEM Adapter 必须做的最小 Patch

当前公开 GEM 的：

```python
GEM.predict()
```

没有把：

```text
observed_motion_3d
motion_mask_3d
```

从 demo `data` 自动复制到 inference `batch`。

因此 V1 的 `gem_adapter.py` 必须增加这一层。

推荐最小 Patch：

```python
if "observed_motion_3d" in data:

    batch["observed_motion_3d"] = (
        data["observed_motion_3d"]
        [None]
        .cuda()
    )

    batch["motion_mask_3d"] = (
        data["motion_mask_3d"]
        [None]
        .cuda()
    )

    batch["rm_text_flag"] = torch.zeros(
        1,
        device=batch["device"],
        dtype=torch.bool,
    )
```

之后继续调用：

```text
create_condition_mask()
→ pipeline.forward()
```

`GEMDiffusion.forward_test()` 已经会将这些字段传给 denoiser。

---

## 7.17.1 V1 不修改 `in_attr`

当前：

```text
configs/pipeline/dual_mode.yaml
```

默认没有：

```text
observed_motion_3d
```

作为 `in_attr`。

因此 V1 只使用：

> **denoiser hard overwrite / inpainting hook**

不尝试新增：

```text
observed_motion_3d feature embedder
```

到训练条件中。

如果后续将它加入 `in_attr`：

```text
需要对应训练过的 embedder 权重
可能需要 retraining / finetuning
```

属于 Phase II，不在 V1 进行。

---

## 7.17.2 `rm_text_flag`

V1 默认：

```text
rm_text_flag = False
```

即：

```text
Text Condition
+
Constraint Condition
```

同时存在。

除非后续做明确 Ablation，不应因为加入 Hard Constraint 而关闭 Text。

---

# 7.18 Hard Condition 的构造原则

不要直接：

```text
Constraint
→ 手填 151-D
```

推荐：

```text
Constraint
      ↓
Reference SMPL State / Sequence
      ↓
GEM-Compatible Coordinate Conversion
      ↓
EnDecoder.encode()
      ↓
151-D normalized motion
      ↓
FeatureMaskBuilder
      ↓
HardMotionCondition
```

---

# 7.19 `body_part_pose`

这是最直接的 Hard Constraint 类型。

例如：

```text
右臂在第 3 秒采用某个 Reference Pose
```

流程：

```text
Reference Pose
      ↓
SMPL Pose
      ↓
EnDecoder.encode()
      ↓
取 right_arm 对应 rotation dimensions
      ↓
mask only those dimensions
```

不要：

```text
mask 整个 151-D frame
```

除非 Planner 使用的是：

```text
BUILD_KEYFRAME
```

而不是 `BUILD_CONSTRAINT`。

---

## 7.19.1 Body-Part Joint Set

例如：

```text
right_arm
```

可展开：

```text
right_shoulder
right_elbow
right_wrist
```

最终 Mask：

```text
只覆盖这些 Joint 的 6D rotation
```

其他：

```text
left arm
legs
root
body shape
```

仍由 GEM 自由生成。

---

# 7.20 `joint_target`

例如：

```text
3 秒时 right_wrist 到达目标 XYZ
```

GEM 151-D 中没有：

```text
right_wrist_xyz
```

因此必须：

```text
Joint XYZ
    ↓
IK
    ↓
SMPL Joint Rotations
    ↓
GEM EnDecoder
```

---

## 7.20.1 Joint Target 流程

```text
ConstraintRequest
      ↓
resolve target XYZ
      ↓
choose base pose
      ↓
FK
      ↓
IK
      ↓
validate residual
      ↓
SMPL pose
      ↓
GEM encode
      ↓
mask affected kinematic chain
```

---

## 7.20.2 Base Pose 来源

优先级：

```text
1. Retrieved Motion Reference
2. Existing Keyframe / Reference Pose
3. Existing accepted neighboring pose
4. Canonical pose
```

V1 不建议直接从完全 neutral T-pose 对复杂目标做大范围 IK，并无条件作为 Hard Constraint。

如果没有可靠 Base Pose：

```text
requested hard joint_target
```

可以返回：

```text
needs_reference
```

或者：

```text
compiled_soft
```

---

# 7.21 IK 实现

GEM 当前 repo 已有：

```text
endecoder.fk_v2()
CCD_IK
process_ik()
```

可以直接复用。

当前后处理已经定义：

```text
left / right leg chain
left / right hand chain
wrist / foot joint ids
```

因此 V1 的：

```python
solve_pose_ik(...)
```

优先封装现有 `CCD_IK`。

接口：

```python
class IKSolveResult(BaseModel):

    success: bool

    body_pose: torch.Tensor

    residual_m: float

    affected_joint_ids: list[int]

    iterations: int
```

主函数：

```python
def solve_pose_ik(
    base_pose,
    target_joint,
    target_position,
    target_rotation=None,
    tolerance_m=0.03,
) -> IKSolveResult:
    ...
```

---

## 7.21.1 IK Validation

IK 完成后必须重新：

```text
FK
```

验证：

```python
residual = norm(
    solved_joint_xyz
    - target_xyz
)
```

如果：

```text
residual <= tolerance
```

才允许转成 Hard Condition。

否则：

```text
status = needs_reference
```

或：

```text
compiled_soft
```

禁止将失败的 IK 结果强制写入 GEM。

---

# 7.22 `root_trajectory`

Root Trajectory 不能简单理解为：

```text
直接写 XYZ 到 GEM
```

因为 GEM Motion Representation 主要使用：

```text
local_transl_vel
```

而不是绝对 world translation。

同时：

```text
local_transl_vel
```

依赖：

```text
root orientation
```

因此必须区分两种情况。

---

## 7.22.1 Orientation 已知

例如：

```text
沿圆形轨迹走
并始终沿轨迹切线方向朝前
```

可以：

```text
Trajectory Builder
      ↓
world translation
+
root orientation
      ↓
build synthetic SMPL sequence
      ↓
EnDecoder.encode()
      ↓
global orientation features
+
local_transl_vel
```

然后 Hard Mask：

```text
global_orient / global_orient_gv
local_transl_vel
```

---

## 7.22.2 Orientation 未知

例如用户只说：

```text
沿 S 形轨迹移动
```

但没有规定身体朝向。

此时不要随意固定：

```text
global orientation
```

更适合：

```text
RewardSpec:
root world trajectory error
```

即：

```text
effective_mode = selection_reward
```

或：

```text
guided_reward
```

由 GEM 自己选择合理朝向。

---

# 7.23 Trajectory Builder

建议：

```text
constraints/trajectory.py
```

支持：

```text
line
circle
arc
spline
waypoints
reference_trajectory
```

接口：

```python
class TrajectorySpec(BaseModel):

    type: str

    duration_s: float

    points: list[list[float]] | None

    radius: float | None

    center: list[float] | None

    orientation_policy: Literal[
        "free",
        "fixed",
        "follow_tangent",
    ]
```

函数：

```python
build_root_trajectory(
    spec,
    fps,
) -> RootTrajectory
```

输出：

```python
class RootTrajectory:
    positions_world: torch.Tensor
    orientations_world: torch.Tensor | None
```

---

## 7.23.1 Cross-Segment Heading Continuity Control

`HeadingContinuitySpec` 来自 Motion Compiler，本身不是数值 Constraint。只有在：

```text
Verifier 已确认 uncommanded heading drift
或
Planner 明确要求将 continuity expectation 升级为 measurable control
```

时，Constraint Compiler 才把它编译为 root-orientation control。

V1 **不新增顶层 Constraint Type**，复用：

```text
root_trajectory
```

的 orientation channel，并允许 orientation-only root control：

```python
class TrajectorySpec(BaseModel):
    type: str
    duration_s: float
    points: list[list[float]] | None
    radius: float | None
    center: list[float] | None

    orientation_policy: Literal[
        "free",
        "fixed",
        "follow_tangent",
        "inherit_anchor",
    ]

    anchor_segment_id: int | None = None
```

当只约束 heading 而不约束 world translation 时：

```text
points = None
orientation_policy = inherit_anchor / fixed
```

### Soft-first 原则

默认生成：

```text
RewardSpec:
root_heading_continuity
```

而不是直接 hard mask 整段 `global_orient`。

原因：

```text
自然动作需要小幅 torso / pelvis / root rotation；
完全固定 root orientation 容易导致僵硬或破坏 sit / reach 等动作。
```

推荐 metric：从 SMPL root orientation 得到 canonical forward vector，投影到 ground plane，和 anchor heading 比较 angular drift。

```text
anchor_heading
= anchor segment terminal window 的 robust mean / median heading

heading_error(t)
= angle(projected_forward(t), anchor_heading)
```

具体 tolerance 不由 Compiler/LLM 猜测，必须来自 config / calibration。

### Hard control 仅在以下情况允许

```text
用户显式要求固定朝向；
可靠 reference/root orientation 已知；
或 soft / selection control 连续失败且 hard condition 不会过约束。
```

此时才考虑编码：

```text
global_orient
global_orient_gv
```

的部分 hard condition。

# 7.24 `contact`

Contact 是最容易被错误 Hard-Code 的 Constraint。

例如：

```text
right hand touches table
```

这并不唯一确定：

```text
shoulder rotation
elbow rotation
torso rotation
root position
```

因此默认：

> **Contact 优先编译为 Soft Constraint。**

---

## 7.24.1 Contact Target 已知

如果：

```text
target point / plane
```

已知，可定义：

\[
E_{contact}
=
E_{position}
+
\lambda_v E_{velocity}
\]

例如静态接触：

\[
E_{position}
=
\frac{1}{T}
\sum_t
\|p_{joint}(t)-p_{target}(t)\|_2^2
\]

\[
E_{velocity}
=
\frac{1}{T-1}
\sum_t
\|p_{joint}(t+1)-p_{joint}(t)\|_2^2
\]

即：

```text
靠近目标
+
接触期间不要乱滑
```

---

## 7.24.2 Plane Contact

如果目标是：

```text
floor
wall
table plane
```

可用：

```text
point + normal
```

定义 Signed Distance：

```python
distance_to_plane(
    joint_position,
    plane_point,
    plane_normal,
)
```

---

## 7.24.3 Contact Hard Condition

只有在：

```text
有明确 Reference Pose / Motion
+
IK residual 合格
+
Hard Mask 不会固定过多自由度
```

时，才允许将 Contact 转成 Hard Local Pose Constraint。

否则保持：

```text
selection_reward / guided_reward
```

---

# 7.25 `fixed_joint`

例如：

```text
左脚在这一段保持不动
```

它本质上是：

> Joint Position 在时间上的低变化约束。

不一定需要预先知道绝对 XYZ。

可以使用：

```text
该窗口第一帧 Joint Position
```

作为动态 Anchor。

定义：

\[
E_{fixed}
=
\frac{1}{T}
\sum_t
\|p_j(t)-p_j(t_0)\|_2^2
\]

或者使用速度：

\[
E_{vel}
=
\frac{1}{T-1}
\sum_t
\|p_j(t+1)-p_j(t)\|_2^2
\]

因此：

```text
fixed_joint
```

默认属于：

```text
selection_reward
```

后续可用于：

```text
guided_reward
```

而不需要强行固定整个腿部 Pose。

---

# 7.26 `body_part_pose`

如果用户 / Retrieval 已提供局部参考姿态：

```text
right arm pose
left leg pose
torso orientation
```

可直接转为 Hard Constraint。

如果只有：

```text
"right arm raised"
```

这种语义描述：

```text
不要直接生成 Joint Rotation
```

应先：

```text
Motion Compiler / Retrieval
```

获得：

```text
Reference Pose
```

或保留 Text Condition。

---

# 7.27 RewardSpec

所有 Soft Constraint 统一编译为：

```python
class RewardSpec(BaseModel):

    reward_type: Literal[
        "joint_position",
        "root_trajectory",
        "contact",
        "fixed_joint",
        "pose_similarity",
    ]

    target: dict

    frame_indices: list[int]

    body_parts: list[str]

    weight: float

    tolerance: float | None

    reduction: Literal[
        "mean",
        "max",
        "terminal",
    ]
```

---

## 7.27.1 V1 用法

V1：

```text
RewardSpec
→ Candidate Scoring
→ Tournament / Verifier
```

即使还没有 Guided Sampling，也可以实际使用。

---

## 7.27.2 Phase II 用法

Phase II：

```text
RewardSpec
→ Guided Generator
→ differentiable constraint cost
→ ReAlign / DNO-style sampling optimization
```

因此 RewardSpec 必须从第一版就保持：

```text
可计算
可微分（如果可能）
与 Verification 定义一致
```

---

# 7.28 VerificationSpec

Constraint 编译时同时产生：

```python
class VerificationSpec(BaseModel):

    metric_type: str

    frame_indices: list[int]

    body_parts: list[str]

    target: dict

    pass_threshold: float

    units: str
```

例如：

```text
joint target
```

```json
{
  "metric_type": "joint_position_error",
  "body_parts": ["right_wrist"],
  "target": {
    "xyz": [0.4, 0.2, 0.9]
  },
  "pass_threshold": 0.05,
  "units": "meter"
}
```

同一个 target 不允许 Verifier 再自行重建。

---

# 7.29 Constraint Tolerance

统一 Schema：

```python
class ConstraintTolerance(BaseModel):

    position_m: float | None

    rotation_deg: float | None

    trajectory_rmse_m: float | None

    velocity_mps: float | None
```

Tolerance 来源优先级：

```text
1. User explicit requirement
2. Task / benchmark specification
3. Constraint config default
```

不要让 LLM 每次自由生成 threshold。

默认值统一放：

```text
configs/constraints.yaml
```

---

# 7.30 Constraint Computation Tool

内部 Numerical Tool 建议提供：

```python
resolve_joint_name(...)
resolve_target_geometry(...)

forward_kinematics(...)
solve_pose_ik(...)

build_root_trajectory(...)
sample_spline(...)
compute_tangent_orientation(...)

extract_reference_pose(...)
extract_reference_trajectory(...)

encode_smpl_condition(...)
build_feature_mask(...)

compute_joint_error(...)
compute_contact_error(...)
compute_fixed_joint_error(...)
compute_trajectory_error(...)
```

原则：

```text
Planner / Compiler
→ 决定问题类型

Numerical Tool
→ 给出确定结果
```

---

# 7.31 GEM Constraint Encoder

建议增加：

```text
constraints/gem_encoder.py
```

接口：

```python
def encode_smpl_constraint(
    smpl_sequence: SMPLSequence,
    adapter_context: GEMAdapterContext,
) -> torch.Tensor:
    ...
```

必须复用：

```text
GEM EnDecoder.encode()
```

而不是复制其：

```text
rotation conversion
normalization
velocity conversion
```

逻辑。

---

## 7.31.1 Adapter Context

为了保证和 Generation 坐标一致：

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

Constraint 编译时使用和 Generation 完全相同的：

```text
camera / coordinate setup
```

---

# 7.32 Feature Mask Builder

接口：

```python
def build_feature_mask(
    total_frames: int,
    frame_indices: list[int],
    feature_groups: list[str],
    joint_ids: list[int] | None = None,
) -> torch.Tensor:
    ...
```

输出：

```text
[L, 151]
```

---

## 7.32.1 Body Pose Mask

例如只控制：

```text
right shoulder
right elbow
```

Mask：

```text
frame 90
body_pose[right_shoulder 6D] = 1
body_pose[right_elbow 6D] = 1
```

其余保持：

```text
0
```

---

## 7.32.2 Root Mask

Root Orientation：

```text
136:148
```

Root Translation Motion：

```text
148:151
```

Body Shape：

```text
126:136
```

默认永远：

```text
mask = 0
```

因为普通 Motion Constraint 不应修改人物体型。

---

# 7.33 Constraint Composer

多个 Constraint 不应各自直接改 GEM input。

生成前由：

```python
ConstraintComposer
```

统一合并。

接口：

```python
def compose_constraints(
    constraints: list[CompiledConstraint],
    total_frames: int,
) -> ConstraintBundle:
    ...
```

---

## 7.33.1 ConstraintBundle

```python
class ConstraintBundle(BaseModel):

    hard_condition_handle: str | None

    reward_specs: list[RewardSpec]

    verification_specs: list[VerificationSpec]

    active_constraint_ids: list[str]

    mask_density: float

    conflicts: list[ConstraintConflict]
```

---

# 7.34 Hard Tensor 合并

初始化：

```python
observed = torch.zeros(
    total_frames,
    151,
)

mask = torch.zeros(
    total_frames,
    151,
)
```

对每个 Hard Constraint：

```python
observed_i, mask_i = load(condition_handle)
```

检查重叠：

```python
overlap = (
    mask > 0
) & (
    mask_i > 0
)
```

---

## 7.34.1 无冲突

```python
observed[mask_i > 0] = observed_i[mask_i > 0]

mask = torch.maximum(
    mask,
    mask_i,
)
```

---

## 7.34.2 有冲突

如果：

```text
同一 frame
同一 feature
两个 Hard Constraint
要求不同数值
```

禁止：

```text
last write wins
```

必须返回：

```text
ConstraintConflict
```

---

# 7.35 Constraint Conflict

```python
class ConstraintConflict(BaseModel):

    constraint_ids: list[str]

    frame_indices: list[int]

    feature_indices: list[int]

    max_value_difference: float

    conflict_type: str
```

出现 Conflict：

```text
ConstraintComposer
→ conflicts != []
→ Generation Guard
→ 禁止 GENERATE
→ return Planner
```

Planner 再决定：

```text
replace constraint
remove constraint
switch hard → soft
change keyframe
```

---

# 7.36 Over-Constraint 检查

Hard Mask 太密会让 GEM 几乎没有生成空间。

因此需要记录：

```python
mask_density = mask.sum() / mask.numel()
```

并记录：

```text
number of constrained frames
number of constrained joints
long continuous hard windows
```

第一版不写死一个未经验证的通用阈值。

通过：

```text
configs/constraints.yaml
+
ablation
```

确定 warning / blocking threshold。

如果过度约束：

```text
status = conflict / warning
```

建议 Planner：

```text
reduce hard region
switch to soft
use keyframe instead
```

---

# 7.37 Retrieved Reference 的使用

如果：

```text
state.retrieved_references
```

已经有：

```text
constraint_source
motion_prior
```

Constraint Compiler 可以加载对应 Handle。

例如：

```text
Retrieved limping motion
      ↓
extract wrist / root trajectory
      ↓
Constraint Compiler
```

但：

> Constraint Compiler 不主动发起新的 Retrieval。

如果 Reference 不足：

```text
status = needs_reference
```

返回 Planner。

---

# 7.38 Keyframe 边界

Whole-Body Keyframe 的正式实现规范见 `07b_keyframe_tool.md`。

Whole-Body Keyframe 仍然属于：

```text
BUILD_KEYFRAME
```

而不是：

```text
BUILD_CONSTRAINT
```

但 Keyframe Tool 最终也可能复用：

```text
GEMConstraintEncoder
FeatureMaskBuilder
ConstraintComposer
```

来构造：

```text
observed_motion_3d
motion_mask_3d
```

区别在语义层：

```text
Constraint
→ 局部 / 几何 / 关系

Keyframe
→ 某一时刻的完整 Whole-Body State
```

底层 Tensor Infrastructure 可以共享。

---

# 7.39 Generation 接口

Planner 执行：

```text
GENERATE
```

时：

```text
state.constraints
      ↓
ConstraintComposer
      ↓
ConstraintBundle
      ↓
GemGenerator
```

Generator 输入：

```python
GenerationRequest(
    ...,
    constraint_bundle=constraint_bundle,
)
```

---

## 7.39.1 Normal Generation

```text
hard_condition
→ 直接注入 GEM

reward_specs
→ 用于 Candidate Selection / Verify
```

---

## 7.39.2 Guided Generation

```text
hard_condition
→ 继续注入 GEM

reward_specs
→ 同时交给 Guided Sampler
```

因此：

```text
Normal / Guided
```

不需要重新编译 Constraint。

---

# 7.40 Verifier 接口

Verifier 不重新解释自然语言 Constraint。

直接读取：

```text
CompiledConstraint.verification_spec
```

例如：

```text
joint_target
→ joint_position_error

root_trajectory
→ trajectory_rmse

contact
→ contact_distance + contact_velocity

fixed_joint
→ joint_drift

body_part_pose
→ rotation / pose error
```

结果必须返回对应：

```text
constraint_id
```

例如：

```json
{
  "constraint_id": "constraint_03",
  "metric": "joint_position_error",
  "value": 0.28,
  "threshold": 0.05,
  "status": "fail"
}
```

这样 Diagnosis 可以精准告诉 Planner：

```text
哪一个 Constraint 失败
```

---

# 7.41 Constraint 类型的具体执行映射

| Constraint Type | 默认执行模式 | Hard 实现 | Soft / Reward 实现 |
|---|---|---|---|
| `joint_target` | soft / conditional hard | IK → chain rotations → mask | joint XYZ distance |
| `body_part_pose` | hard | reference pose → joint 6D mask | pose rotation distance |
| `root_trajectory` | soft；orientation 已知时可 hard | encoded root orientation + local velocity | world root trajectory RMSE |
| `contact` | soft | reliable IK/reference pose 时局部 hard | contact distance + velocity |
| `fixed_joint` | soft | reference pose sequence 时可 hard | joint drift / velocity |

该表作为 V1 默认行为。

---

# 7.42 Constraint Store

不要把大型 Tensor 直接放进 Planner Context。

建议：

```text
constraints/store.py
```

保存：

```text
HardMotionCondition Tensor
Reference SMPL
Derived Trajectory
```

State 只保存：

```json
{
  "constraint_id": "constraint_03",
  "type": "contact",
  "segment": 1,
  "requested_strength": "hard",
  "effective_mode": "guided_reward",
  "status": "active",
  "condition_handle": null,
  "reward_spec_id": "reward_03"
}
```

Planner 不需要读取：

```text
[L, 151]
```

Tensor。

---

# 7.43 Constraint Skill Card

建议：

```text
planner_skills/constraint.md
```

内容至少包括：

```text
WHEN TO USE

Use BUILD_CONSTRAINT for measurable motion requirements:
- joint target
- local body-part pose
- root trajectory
- contact
- fixed joint

DO NOT USE

Do not use for:
- general semantics
- style that has no numerical definition
- temporal ordering
- action frequency
- whole-body keyframes

HARD VS SOFT

Use hard only when a valid motion state can be constructed.

Use soft when many poses can satisfy the goal or hard masking
would over-constrain the motion.

GEOMETRY

Never invent object coordinates.

If target geometry is unavailable:
→ request reference / geometry
→ or keep the condition soft.

REPAIR

After a constraint failure:
- replace only the failed constraint;
- preserve passed constraints;
- prefer local changes over rebuilding all conditions.
```

---

# 7.44 内部代码结构

建议：

```text
constraints/
│
├── compiler.py
├── schemas.py
├── store.py
│
├── router.py
│
├── time_resolver.py
├── coordinates.py
├── joints.py
│
├── geometry.py
├── trajectory.py
├── kinematics.py
├── contact.py
│
├── reward.py
├── verifier_spec.py
│
├── gem_encoder.py
├── feature_mask.py
├── composer.py
│
└── validators.py
```

职责：

```text
compiler.py
→ 主入口

schemas.py
→ Request / Result / Constraint Schema

store.py
→ Tensor / Handle

router.py
→ constraint type → compiler

time_resolver.py
→ seconds / segment → frames

coordinates.py
→ coordinate conversion

joints.py
→ SMPL joint registry

geometry.py
→ point / plane / distance

trajectory.py
→ line / circle / spline / waypoints

kinematics.py
→ FK / IK

contact.py
→ contact objective

reward.py
→ RewardSpec / differentiable cost

verifier_spec.py
→ verification metric definition

gem_encoder.py
→ SMPL → GEM 151-D

feature_mask.py
→ per-frame / per-joint mask

composer.py
→ combine multiple constraints

validators.py
→ geometry / IK / conflict / over-constraint checks
```

---

# 7.45 主执行伪代码

```python
def compile_constraint(
    state: MotionAgentState,
    request: ConstraintRequest,
) -> ConstraintCompileResult:

    validate_request(
        state=state,
        request=request,
    )

    if request.mode == "remove":
        remove_constraint(
            state=state,
            constraint_id=request.constraint_id,
        )

        return ConstraintCompileResult(
            status="compiled",
            constraint=None,
            warnings=[],
            routing_hints=[],
        )

    segment = state.motion_plan.get_segment(
        request.target_segment
    )

    resolved_time = time_resolver.resolve(
        request.time_spec,
        segment=segment,
        fps=state.task.fps,
        total_frames=state.task.total_frames,
    )

    target_result = resolve_constraint_target(
        request=request,
        state=state,
        segment=segment,
    )

    if target_result.status == "needs_geometry":
        return ConstraintCompileResult(
            status="needs_geometry",
            constraint=None,
            warnings=target_result.warnings,
            routing_hints=[],
        )

    if target_result.status == "needs_reference":
        return ConstraintCompileResult(
            status="needs_reference",
            constraint=None,
            warnings=target_result.warnings,
            routing_hints=["RETRIEVE_REFERENCE"],
        )

    compiled = constraint_router.compile(
        constraint_type=request.constraint_type,
        request=request,
        segment=segment,
        resolved_time=resolved_time,
        target=target_result.target,
        state=state,
    )

    validate_compiled_constraint(
        compiled
    )

    save_constraint_payloads(
        compiled
    )

    return ConstraintCompileResult(
        status=compiled.status,
        constraint=compiled.constraint,
        warnings=compiled.warnings,
        routing_hints=compiled.routing_hints,
    )
```

---

# 7.46 `joint_target` 伪代码

```python
def compile_joint_target(
    request,
    target,
    resolved_time,
    state,
):

    target_xyz = coordinate_resolver.resolve_point(
        target
    )

    base_pose = choose_base_pose(
        state=state,
        request=request,
    )

    if base_pose is None:
        return needs_reference()

    ik_result = solve_pose_ik(
        base_pose=base_pose,
        target_joint=request.body_parts[0],
        target_position=target_xyz,
        tolerance_m=request.tolerance.position_m,
    )

    reward_spec = build_joint_position_reward(
        ...
    )

    verifier_spec = build_joint_position_verifier(
        ...
    )

    if (
        request.requested_strength == "hard"
        and ik_result.success
    ):

        encoded = gem_encoder.encode_pose(
            ik_result.body_pose,
            context=state.gem_adapter_context,
        )

        mask = feature_mask.for_kinematic_chain(
            joint_ids=ik_result.affected_joint_ids,
            frames=resolved_time.frame_indices,
        )

        condition_handle = store_hard_condition(
            values=encoded,
            mask=mask,
        )

        effective_mode = "hard_condition"

    else:

        condition_handle = None
        effective_mode = "selection_reward"

    return CompiledConstraint(...)
```

---

# 7.47 `root_trajectory` 伪代码

```python
def compile_root_trajectory(
    request,
    target,
    resolved_time,
    state,
):

    trajectory = build_root_trajectory(
        target,
        fps=state.task.fps,
    )

    reward_spec = build_trajectory_reward(
        trajectory
    )

    verifier_spec = build_trajectory_verifier(
        trajectory
    )

    if trajectory.orientations_world is not None:

        smpl_reference = build_root_reference_sequence(
            trajectory
        )

        encoded = gem_encoder.encode_sequence(
            smpl_reference,
            context=state.gem_adapter_context,
        )

        mask = feature_mask.for_features(
            frames=resolved_time.frame_indices,
            features=[
                "global_orient",
                "global_orient_gv",
                "local_transl_vel",
            ],
        )

        condition_handle = store_hard_condition(
            encoded,
            mask,
        )

        effective_mode = "hard_condition"

    else:

        condition_handle = None
        effective_mode = "selection_reward"

    return CompiledConstraint(...)
```

---

# 7.48 Constraint Composer 伪代码

```python
def compose_constraints(
    constraints,
    total_frames,
):

    observed = torch.zeros(
        total_frames,
        151,
    )

    mask = torch.zeros(
        total_frames,
        151,
    )

    rewards = []
    verifiers = []
    conflicts = []

    for constraint in constraints:

        if constraint.hard_condition_handle:

            condition = constraint_store.load(
                constraint.hard_condition_handle
            )

            new_conflicts = detect_overlap_conflict(
                observed,
                mask,
                condition.values,
                condition.mask,
            )

            conflicts.extend(
                new_conflicts
            )

            if not new_conflicts:
                observed[
                    condition.mask > 0
                ] = condition.values[
                    condition.mask > 0
                ]

                mask = torch.maximum(
                    mask,
                    condition.mask,
                )

        if constraint.reward_spec:
            rewards.append(
                constraint.reward_spec
            )

        verifiers.append(
            constraint.verification_spec
        )

    return ConstraintBundle(
        hard_condition_handle=save_if_needed(
            observed,
            mask,
        ),
        reward_specs=rewards,
        verification_specs=verifiers,
        active_constraint_ids=[
            x.constraint_id
            for x in constraints
        ],
        mask_density=float(
            mask.mean()
        ),
        conflicts=conflicts,
    )
```

---

# 7.49 V1 实现顺序

建议按以下顺序完成。

## Step 1：GEM Hard Hook

先实现：

```text
observed_motion_3d
motion_mask_3d
GemAdapter injection
```

并测试：

```text
单帧局部 Pose Mask
```

---

## Step 2：Feature Mapper

实现：

```text
151-D mapping
joint → 6D body-pose slice
root feature mask
```

---

## Step 3：Reference Pose Hard Constraint

先实现最稳定的：

```text
body_part_pose
```

测试局部 Arm / Leg Pose。

---

## Step 4：FK / IK

复用 GEM：

```text
fk_v2
CCD_IK
```

实现：

```text
joint_target
```

---

## Step 5：Soft Reward

实现：

```text
joint position error
fixed joint drift
contact distance
trajectory RMSE
```

供：

```text
Tournament / Verifier
```

使用。

---

## Step 6：Root Trajectory

先实现：

```text
line
circle
waypoints
```

再增加：

```text
spline
```

---

## Step 7：Constraint Composer

支持：

```text
multiple active constraints
conflict detection
mask density
```

---

## Step 8：Guided Generation 接口

先只输出：

```text
RewardSpec
```

真正 DNO / ReAlign Guidance 在后续 Generation 模块完成。

---

# 7.50 V1 最低实现范围

V1 必须完成：

```text
ConstraintRequest

Time Resolver

Joint Registry

Coordinate Resolver

body_part_pose hard condition

joint_target + IK

root_trajectory reward

contact reward

fixed_joint reward

GEM 151-D Encoder

motion_mask_3d

observed_motion_3d

ConstraintComposer

VerificationSpec

Constraint Store
```

V1 可以暂缓：

```text
arbitrary object collision

full scene physics

learned IK

learned constraint router

DNO noise initialization

gradient-guided optimization

observed_motion_3d feature-embedder retraining
```

---

# 7.51 单元测试要求

至少覆盖：

```text
Single Joint Target

Body-Part Pose

Sparse Frame Mask

Root Trajectory

Contact Point

Contact Plane

Fixed Foot

Retrieved Reference Constraint

Unresolved Scene Object

IK Failure

Constraint Replace

Constraint Remove

Multiple Compatible Constraints

Conflicting Hard Constraints

Over-Constrained Mask

Coordinate Transform

GEM Encode / Decode Consistency
```

---

## Case 1：Joint Target

```text
Input:
right wrist at known XYZ at t=3s

Expected:
target resolved
IK success
affected hand-chain mask only
VerificationSpec created
```

---

## Case 2：Contact Without Geometry

```text
Input:
right hand touches the table

No scene geometry
No retrieved reference

Expected:
status = needs_geometry
or compiled semantic / soft fallback

Never invent table XYZ
```

---

## Case 3：Fixed Foot

```text
Input:
keep left foot fixed during segment 2

Expected:
soft fixed_joint reward
joint drift verifier
no whole-leg hard freeze by default
```

---

## Case 4：Root Circle

```text
Input:
walk along circle
orientation_policy = follow_tangent

Expected:
world trajectory
root orientation
GEM encoded root features
hard root mask possible
trajectory verifier
```

---

## Case 5：Conflicting Constraints

```text
Constraint A:
right arm pose X at frame 90

Constraint B:
right arm pose Y at frame 90

Expected:
ConstraintConflict
GENERATE blocked
return Planner
```

---

## Case 6：Heading Continuity

```text
Input:
segment 0 establishes forward heading
segment 1 heading_continuity = inherit_previous
Planner requests measurable continuity control
```

Expected：

```text
reuse root_trajectory orientation channel
position remains unconstrained
soft root_heading_continuity RewardSpec created
VerificationSpec created
no full-body hard freeze
```

并测试：

```text
explicit turn
→ 不得继续沿用旧 anchor
```

# 7.52 Constraint 模块评估指标

模块级别至少记录：

```text
Constraint Compile Success Rate

Hard / Soft Routing Accuracy

IK Success Rate

IK Residual

Constraint Conflict Rate

Mask Density

Compilation Latency
```

生成结果记录：

```text
Joint Position Error

Root Trajectory RMSE

Contact Distance

Contact Velocity

Fixed Joint Drift

Pose Rotation Error
```

Agent 层记录：

```text
Constraint Repair Success Rate

Constraint Failure → Correct Repair Routing Rate

Average Constraint Repair Rounds

Hard Constraint Success vs Soft Constraint Success

Constraint Cost / Generation Round
```

---

# 7.53 当前模块完成标准

Constraint Compiler 完成后，必须稳定实现：

```text
Planner BUILD_CONSTRAINT
        ↓
ConstraintRequest
        ↓
Time / Target / Coordinate Resolution
        ↓
Constraint-specific Compilation
        ↓
Hard Condition and/or RewardSpec
        ↓
VerificationSpec
        ↓
CompiledConstraint
        ↓
State
        ↓
Planner
```

生成时：

```text
CompiledConstraint[]
        ↓
ConstraintComposer
        ↓
ConstraintBundle
        │
        ├── observed_motion_3d
        ├── motion_mask_3d
        ├── RewardSpec[]
        └── VerificationSpec[]
        ↓
Generator / Verifier
```

并保证：

```text
Constraint Compiler 只负责数值约束编译。

是否建立 Constraint
由 Planner 决定。

Whole-Body Keyframe
由 BUILD_KEYFRAME 负责。

Temporal / Frequency
由 Motion Compiler + Event Verifier 负责。

Retrieval
由 Planner 单独调用。

Guided Diffusion Optimization
由 Guided Generator 负责。

Verifier 直接复用同一个 VerificationSpec，
不重新解释用户要求。
```

# 7.54 Heading Continuity Completion Addendum

Constraint Compiler 还必须保证：

```text
orientation-only control 可以复用 root_trajectory；
默认 soft-first；
position 可保持 free；
explicit turn 会更新 anchor；
hard root mask 不得成为默认行为。
```
