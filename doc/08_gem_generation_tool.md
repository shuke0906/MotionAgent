# MotionAgent for GEM：模块实现指南

> 本文件从总实现指南中拆分，对应第 8 单元。内容保持模块边界独立，其输入输出接口需与相邻模块的 Schema 保持一致。

# 8. GEM Generation Tool

## Phase 7 Infrastructure Scope (Revised)

Phase 7 completes the agentic generation infrastructure. `ConditionBundle`
means information provided to the generator, not a guarantee of constraint
satisfaction. The full module design below includes future research goals;
its preservation/satisfaction requirements are deferred for the Phase 7 gate.

Current path:

```text
MotionSpecification -> ConditionCompiler -> ConditionBundle
-> GenerationRequest -> GEM Adapter -> Frozen GEM -> CandidateStore
```

Temporal semantics from Motion Compiler, including `temporal_constraint`,
`temporal_mode` and `temporal_relation`, are preserved in
`GenerationRequest.motion_spec`. This interface guarantees only:

```text
Natural language
        ↓
structured temporal representation
        ↓
GenerationRequest
```

It does not claim that Frozen GEM will execute an exact repetition count.
Exact event-frequency satisfaction remains future verifier, guided generation
or repair-loop work.

Concurrent semantic segments may project to a single compound caption.
`GEMTextCondition.segment_bounds` retains their original ID-to-frame mapping;
semantic IDs must not be inferred from caption indices. Only preflight target
validation and segment-range lookup consume this mapping. Legacy conditions
without it retain their previous index-based interpretation. No GEM model,
diffusion, mask semantics or sampling algorithm changes are implied.

The gate checks schemas, masks, keyframe specifications, adapter forwarding,
request metadata, candidate persistence, K sampling, cache, worker health and
typed failures. Candidate metadata persists the effective `GenerationRequest`,
including the composed condition handle, verification specifications and
keyframe specifications. Motion tensors and poses remain in external stores.

Hard-condition error, keyframe error and outside-segment preservation error
remain collected as research metrics. Report each interface as PASS/FAIL and
each satisfaction result as `DEFERRED (requires guided generation)`, with a
separate `tolerance_met` measurement. Satisfaction is not a Phase 7 blocker.
`constraint_safe` disables GEM postprocessing; it does not enforce targets.

Future insertion point:

```text
ConditionBundle -> Guided Sampling Module -> Constraint Satisfaction
```

The existing `GENERATE.strategy` boundary remains available but guided
generation is feature-gated. Phase 7 does not modify diffusion sampling,
the denoiser, GEM/T5 weights, losses or the sampling loop. Exact keyframe
forcing, true diffusion-time inpainting and guided generation are deferred.
Tournament, Verifier and Repair are outside this phase.

本节定义 GEM Generation Tool 的**实现规范**。

Generation Tool 的职责是：

> **接收已经由 Motion Compiler、Constraint Compiler、Keyframe Tool 等模块编译完成的条件，构造合法的 GEM inference request，稳定地产生一个或多个可复现的 Motion Candidate，并将重型 Tensor 结果以 Handle 的形式返回给后续 Tournament / Verifier。**

Generation Tool 是：

```text
Planner-controlled
+
GPU Execution Tool
+
GEM Adapter
+
Candidate Sampler
```

它不是 Agent，不负责判断用户真正想要什么，也不负责选择最终 Candidate。

### LangGraph 执行边界

V1 中 GEM Generation 由 `generation` LangGraph node 触发，但 GPU inference 仍由独立 GEM worker/service 执行。Generation node 提交 `GenerationResult` 后具有固定 edge：

```text
generation → tournament → verifier → diagnosis → planner
```

LangGraph 不持有 GEM tensor；GraphState 只保存 generation/candidate handle 与轻量 metadata。

---

## 8.1 在整体 Planner 中的定位

Generation Tool 只能由 Planner 的：

```text
GENERATE
```

Action 触发。

整体链路：

```text
Planner
   │
   └── GENERATE
          │
          ▼
GenerationRequestBuilder
          │
          ▼
Generation Preflight
          │
          ▼
      GEM Generator
          │
      K Candidates
          │
          ▼
   GenerationResult
          │
          ▼
      Update State
          │
          ▼
 ── automatic subgraph ──
          │
          ▼
 Candidate Tournament
          │
          ▼
      Multi-Verifier
          │
          ▼
       Diagnosis
          │
          ▼
        Planner
```

Generation Tool 执行完成后：

```text
不自动选 Champion
不自动 Accept
不自动修改 Prompt
不自动修改 Constraint
不自动 Retry
```

这些由后续模块或 Planner 负责。

---

## 8.2 职责边界

Generation Tool 负责：

```text
从 MotionAgentState 构造 GenerationRequest
执行生成前合法性检查
加载 / 复用 GEM Model
构造 Pure-Text GEM Input
构造 multi_text_data
注入 Hard Motion Condition
合并 Keyframe / Constraint Hard Condition
控制 Random Seed
生成 K 个 Candidate
支持 Full Generation
支持 Segment Inpainting / Regeneration
控制 GEM Post-processing Policy
解码 Motion Representation
保存 Candidate Tensor
记录完整生成 Metadata
处理 GPU / OOM / Runtime Error
为 Guided Generation 保留统一接口
```

Generation Tool 不负责：

```text
重新理解 User Prompt
生成 Motion DSL
主动 Retrieval
主动构造 Constraint
主动构造 Keyframe
决定哪个 Candidate 最好
执行 VISTA Tournament
执行最终 Constraint Verification
决定是否 Retry
决定是否 ACCEPT
```

---

## 8.3 当前 GEM 实现基础

本项目第一版直接复用当前 GEM-SMPL inference path。

当前 GEM 代码已经提供：

```text
multi-text temporal conditioning
T5-3B text encoding
DDIM sampling
Classifier-Free Guidance
151-D motion representation
global / in-camera SMPL decode
static-joint post-processing
IK post-processing
```

当前默认 Generation 配置中：

```text
sampler = DDIM
test timestep respacing = 50
CFG guidance = 2.5
DDIM eta = 0
```

这些值属于：

```text
checkpoint / GEM config
```

Generation Tool 第一版不在代码中重复 hard-code。

统一从：

```text
GEM Model Config
```

读取，并写入 Generation Metadata。

---

## 8.3.1 长序列

GEM denoiser 内部：

```text
max_len
```

不是严格的 Sequence Length 上限。

当：

```text
L > max_len
```

时，当前实现会建立局部 attention mask。

因此当前 demo 可以使用：

```text
300 frames
```

的 text segment。

但是服务层仍然必须配置：

```text
max_request_frames
```

控制：

```text
GPU memory
latency
request budget
```

该限制属于系统配置，不直接等同于 GEM 网络中的 `max_len`。

---

# 8.4 Generation 的输入不是 Planner JSON

Planner 输出：

```json
{
  "action": "GENERATE",
  "payload": {
    "strategy": "normal",
    "scope": "full",
    "num_candidates": 4
  }
}
```

不能直接交给 GEM。

Orchestrator 必须先执行：

```text
PlannerDecision
+
MotionAgentState
      ↓
GenerationRequestBuilder
      ↓
GenerationRequest
```

这样 Planner 不需要知道 GEM Tensor / Camera / Mask 等底层字段。

---

# 8.5 `GenerationRequest`

主接口：

```python
def generate_motion(
    request: GenerationRequest,
) -> GenerationResult:
    ...
```

推荐 Schema：

```python
class GenerationRequest(BaseModel):

    generation_id: str

    strategy: Literal[
        "normal",
        "guided",
    ]

    scope: Literal[
        "full",
        "segment",
    ]

    target_segments: list[int] | None

    fps: int
    total_frames: int

    text_condition: GEMTextCondition

    condition_bundle: GenerationConditionBundle

    previous_candidate_id: str | None

    num_candidates: int

    seeds: list[int]

    postprocess_policy: Literal[
        "gem_default",
        "none",
        "constraint_safe",
    ]

    output_policy: GenerationOutputPolicy

    timeout_s: float | None
```

---

## 8.5.1 `strategy`

### `normal`

使用 GEM 当前标准：

```text
DDIM
+
CFG
+
Compiled Conditions
```

### `guided`

使用：

```text
GEM
+
RewardSpec
+
Guided Sampling
```

第一版如果 Guided Sampler 尚未启用：

```text
strategy="guided"
```

必须在 Preflight 阶段返回：

```text
guided_generation_not_available
```

禁止静默退化为 normal。

---

## 8.5.2 `scope`

### `full`

重新生成完整 Motion Sequence。

### `segment`

只允许目标 Segment 发生变化，其余已经通过的 Motion 尽量保持不变。

`segment` 不等于：

```text
单独生成一个短 Clip
然后直接拼回原 Motion
```

V1 实现采用：

> **Full-sequence masked inpainting**

后文详细定义。

---

## 8.5.3 `num_candidates`

Planner 指定：

```text
K
```

第一版建议从配置读取默认值，例如：

```text
candidate_count_default
```

而不是把：

```text
K = 4
```

写死在 Generator 内部。

---

## 8.5.4 `seeds`

必须满足：

```text
len(seeds) == num_candidates
```

所有 Candidate 必须有独立 Seed。

如果 Planner 未显式指定 Seed，由：

```text
GenerationRequestBuilder
```

生成并写入 Request。

Seed 一旦生成，不允许在同一次 Job 中变化。

---

# 8.6 `GenerationConditionBundle`

Generator 不直接读取：

```text
Motion Compiler 内部对象
Constraint Compiler 内部对象
Keyframe Tool 内部对象
```

统一转换成：

```python
class GenerationConditionBundle(BaseModel):

    text_condition_id: str

    hard_motion_condition_handle: str | None

    reward_specs: list[RewardSpec]

    active_constraint_ids: list[str]

    active_keyframe_ids: list[str]

    active_reference_ids: list[str]

    condition_fingerprint: str
```

---

## 8.6.1 Text

来自：

```text
Motion Compiler
→ GEMTextCondition
```

---

## 8.6.2 Hard Motion Condition

来自：

```text
07 Constraint Compiler
→ ConstraintBundle

07B Keyframe Tool
→ KeyframeSpec
→ Keyframe HardMotionCondition
```

GenerationRequestBuilder 不直接把整个 `KeyframeSpec` 传给 GEM；它只读取 Keyframe Tool 已编译好的 HardMotionCondition / Verification metadata，并与 Constraint Hard Condition 合并。这样 `KeyframeSpec` 是上游 canonical artifact，而 Generator 消费的是其 GEM-ready condition view。


两者需要在进入 Generator 前合并成唯一：

```text
observed_motion_3d
motion_mask_3d
```

---

## 8.6.3 RewardSpec

来自：

```text
Constraint Compiler
```

V1：

```text
用于 Candidate Scoring / Verifier
```

Phase II：

```text
同时用于 Guided Sampling
```

---

## 8.6.4 Reference

普通 `strategy="normal"`：

```text
Retrieved Reference 不直接进入 GEM
```

它应先通过：

```text
Motion Compiler
Constraint Compiler
Keyframe Tool
```

转换成 GEM 能使用的 Condition。

只有：

```text
strategy="guided"
```

并且 Guided Method 明确支持 Motion Prior 时，才允许读取：

```text
active_reference_ids
```

---

# 8.7 `GenerationRequestBuilder`

建议：

```text
generators/request_builder.py
```

主接口：

```python
def build_generation_request(
    state: MotionAgentState,
    decision: PlannerDecision,
) -> GenerationRequest:
    ...
```

执行：

```text
1. 获取当前 Motion Specification
2. 获取 GEMTextCondition
3. 获取 Active Constraints
4. 获取 Active Keyframes
5. 合并 Hard Conditions
6. 收集 RewardSpec
7. 解析 Full / Segment Scope
8. 生成 Seeds
9. 选择 Postprocess Policy
10. 创建 Request Fingerprint
```

---

# 8.8 Generation Preflight

任何 GPU inference 前必须先执行：

```python
def validate_generation_request(
    request: GenerationRequest,
    state: MotionAgentState,
) -> GenerationPreflightReport:
    ...
```

Schema：

```python
class GenerationPreflightReport(BaseModel):

    status: Literal[
        "ready",
        "blocked",
    ]

    errors: list[str]

    warnings: list[str]

    estimated_frames: int

    hard_mask_density: float

    condition_fingerprint: str
```

---

## 8.8.1 必须阻止的情况

```text
Motion Plan 不存在

GEMTextCondition 不存在

total_frames <= 0

Caption / Window 数量不一致

Temporal Window 非法

Constraint Conflict 未解决

Keyframe / Constraint Hard Mask 冲突

GENERATE budget == 0

num_candidates <= 0

Seeds 数量错误

Segment scope 但没有 previous_candidate

Segment ID 非法

Guided strategy 但 Guided Sampler 未启用

Request 超过 max_request_frames
```

Preflight 失败：

```text
禁止进入 GPU Worker
```

返回 Planner / Orchestrator。

---

# 8.9 GEM Adapter Context

所有 Generation、Constraint、Keyframe 必须共享同一：

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

这样可以保证：

```text
Constraint 编译坐标
Generation 坐标
Verification 坐标
```

一致。

---

# 8.10 Pure-Text GEM Adapter

当前官方 `demo_smpl.py` 要求至少输入一个 Video，主要是为了从 Video 取得：

```text
reference width / height
camera intrinsics
```

Text Segment 本身实际上使用：

```text
zero kp2d
zero image feature
false visual masks
static camera
```

因此 MotionAgent 不应为了 Text-to-Motion 强行准备一个假的 Video 文件。

应实现：

```text
PureTextGEMAdapter
```

直接构造 GEM 所需的 synthetic non-visual inputs。

---

## 8.10.1 Pure Text Data

对于：

```text
L = total_frames
```

构造：

```python
kp2d = torch.zeros(
    L,
    17,
    3,
)

f_imgseq = torch.zeros(
    L,
    1024,
)

has_img_mask = torch.zeros(
    L,
    dtype=torch.bool,
)

has_2d_mask = torch.zeros(
    L,
    dtype=torch.bool,
)

has_cam_mask = torch.zeros(
    L,
    dtype=torch.bool,
)

bbx_xys = torch.zeros(
    L,
    3,
)
```

---

## 8.10.2 Static Camera

使用 GEM 当前工具：

```text
estimate_K(width, height)
```

构造：

```text
R_w2c = Identity
cam_angvel = static
cam_tvel = 0
K_fullimg = repeated estimated K
```

其中：

```text
width
height
```

从：

```text
GEMAdapterContext
```

读取。

不要为了 Pure Text 每次依赖外部 Video。

---

# 8.11 `multi_text_data`

来自：

```text
Motion Compiler
→ GEMTextCondition
```

转换成当前 GEM 所需：

```python
multi_text_data = {
    "vid": [...],
    "caption": [...],
    "text_ind": [...],
    "window_start": ...,
    "window_end": ...,
}
```

顶层：

```text
caption
```

使用：

```text
第一个 active caption
```

保持和当前 GEM demo 行为一致。

真正分段 conditioning 由：

```text
multi_text_data
```

控制。

---

# 8.12 Text Embedding Cache

当前 `GEM.predict()` 会在：

```text
multi_text_data
```

缺少：

```text
text_embed
```

时调用 T5 编码，并直接将结果写回该 Dict。

工业实现中不能依赖共享 Mutable Dict。

建议：

```text
Generation Request
→ immutable

TextEmbeddingCache
→ 独立管理
```

Cache Key：

```text
model_version
+
caption text
+
max_text_len
```

输出：

```text
encoded text tensor handle
```

---

## 8.12.1 Request-local `multi_text_data`

每个 Generation Job 创建自己的：

```text
multi_text_data
```

禁止多个并发 Request 共用同一个可变 Dict。

这样避免：

```text
跨 Request Tensor 污染
GPU Tensor 生命周期冲突
并发 Mutation
```

---

# 8.13 Hard Motion Condition 接入

如果：

```text
condition_bundle.hard_motion_condition_handle != None
```

Generator 从 Constraint / Keyframe Store 加载：

```text
observed_motion_3d
motion_mask_3d
```

形状必须是：

```text
[L, 151]
```

Adapter 加 Batch 维：

```text
[1, L, 151]
```

---

## 8.13.1 当前 GEM 最小 Patch

如第 7 部分所述，当前公开 `GEM.predict()` 没有自动从 `data` 复制：

```text
observed_motion_3d
motion_mask_3d
```

因此 `gem_adapter.py` 必须在 inference batch 构造时加入：

```python
if "observed_motion_3d" in data:

    batch["observed_motion_3d"] = (
        data["observed_motion_3d"]
        [None]
        .to(device)
    )

    batch["motion_mask_3d"] = (
        data["motion_mask_3d"]
        [None]
        .to(device)
    )

    batch["rm_text_flag"] = torch.zeros(
        1,
        device=device,
        dtype=torch.bool,
    )
```

保持：

```text
Text Condition
+
Hard Motion Condition
```

共同生效。

---

# 8.14 Keyframe 接入

Generation Tool 不关心 Keyframe 如何产生。

Keyframe Tool 最终必须提供与 Constraint 一致的：

```text
HardMotionCondition
```

或：

```text
RewardSpec
```

因此 Generator 只消费合并后的：

```text
GenerationConditionBundle
```

而不增加：

```text
special keyframe-specific GEM API
```

这样可以避免 Generator 与 Keyframe 实现强耦合。

---

# 8.15 Full Generation

`scope="full"`：

```text
Motion Specification
+
Text Condition
+
Hard Conditions
+
Random Noise
      ↓
GEM
      ↓
Full Motion Candidate
```

所有 Candidate 使用完全相同：

```text
Motion Specification
Condition Bundle
Checkpoint
Sampler Config
```

唯一默认变化是：

```text
Seed / Initial Noise
```

这样 Candidate 差异才能解释为：

```text
sampling diversity
```

而不是输入条件变化。

---

# 8.16 Segment Regeneration

`scope="segment"` 必须要求：

```text
previous_candidate_id != None
```

V1 不采用：

```text
单独生成 segment
→ 后处理拼接
```

因为容易造成：

```text
root discontinuity
velocity jump
pose discontinuity
body shape mismatch
```

采用：

> **Full-sequence constrained inpainting**

---

## 8.16.1 Segment Inpainting

假设需要重新生成：

```text
segment [s:e)
```

加载上一轮 Candidate 的：

```text
motion_repr
```

得到：

```text
previous_motion
[L, 151]
```

构造：

```text
preserve_mask
[L, 151]
```

规则：

```text
segment 外
→ mask = 1

目标 segment 内
→ mask = 0
```

然后再合并当前：

```text
Constraint / Keyframe Hard Mask
```

目标 Segment 内的显式 Hard Constraint 仍保持：

```text
mask = 1
```

---

## 8.16.2 Body Shape

Segment Regeneration 时：

```text
betas
```

默认在完整 Sequence 上保持上一轮值。

即：

```text
betas mask = 1
for all frames
```

避免：

```text
同一个人的体型在 segment 边界变化
```

---

## 8.16.3 Boundary Context

至少保证：

```text
s - 1
e
```

等相邻已通过 Frame 被保留。

可选增加：

```text
boundary_context_frames
```

在 segment 两侧提供固定上下文。

具体数值通过配置和实验确定，不在逻辑代码中硬编码。

---

# 8.17 Candidate Generation

当前公开：

```text
GEM.predict()
```

是：

```text
B = 1
```

的 inference wrapper。

因此 V1 不假设它支持：

```text
一次 batch 生成 K 个 Candidate
```

第一版采用：

```text
一个 GEM GPU Worker
+
同一 Request
+
K 个 Seed
+
串行 Candidate Sampling
```

---

## 8.17.1 Candidate Loop

```text
seed_1
→ GEM
→ candidate_1

seed_2
→ GEM
→ candidate_2

...

seed_K
→ GEM
→ candidate_K
```

Candidate 可以逐个落盘，避免全部长期占用 GPU Memory。

---

## 8.17.2 后续 Batch 优化

Phase II 可以重构：

```text
GEM.predict()
```

支持：

```text
B = K
```

但必须重新验证：

```text
multi_text_data batch semantics
condition masks
hard conditions
random noise
memory
postprocess
```

在完成这些验证前，不将 Batch-K 作为 V1 假设。

---

# 8.18 Seed 与可复现性

当前 GEM sampler 内部使用：

```text
torch.randn_like(...)
```

生成初始 Noise。

V1 的 Seed 机制要求：

> 每个 GPU Worker 串行执行 Candidate，以避免并发 Request 竞争同一个 CUDA RNG 状态。

每个 Candidate：

```python
with torch.random.fork_rng(
    devices=[device_id]
):

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)

    pred = gem_model.predict(...)
```

并记录：

```text
seed
PyTorch version
CUDA version
checkpoint
GEM config hash
```

---

## 8.18.1 更强的 Seed 方案

Phase II 建议 Patch GEM：

```text
forward_test(inputs)
```

支持显式：

```text
initial_noise
```

优先级：

```python
if "initial_noise" in inputs:
    noise = inputs["initial_noise"]
else:
    noise = torch.randn_like(motion)
```

这样：

```text
Seed
→ RequestBuilder
→ explicit noise tensor
```

比全局 RNG 更适合：

```text
并发
Batch Generation
严格复现
```

---

# 8.19 GEM Sampling

当前 GEM Generation path：

```text
Condition Assembly
      ↓
Diffusion Input
      ↓
Classifier-Free Guidance
      ↓
DDIM Sampling
      ↓
pred_x
      ↓
EnDecoder.decode()
      ↓
SMPL Parameters
```

V1 不修改：

```text
DDIM schedule
CFG implementation
GEM denoiser weights
```

除非是专门的 Generation Ablation。

---

## 8.19.1 Sampler Config

所有 Sampling 参数必须来自：

```text
GenerationConfig
+
GEM checkpoint config
```

记录：

```python
class SamplerMetadata(BaseModel):

    sampler: str

    timestep_respacing: str

    guidance_scale: float

    ddim_eta: float
```

Candidate 必须保存该 Metadata。

---

# 8.20 Post-processing Policy

当前 GEM Pipeline 在：

```text
postproc=True
```

时可能执行：

```text
static joint correction
+
IK post-processing
```

它会修改：

```text
global translation
body pose
```

这对普通 GEM Motion 可能有益，但可能破坏 MotionAgent 已经施加的 Hard Constraint。

因此 Generation Tool 必须显式管理 Post-processing。

---

## 8.20.1 `gem_default`

适合：

```text
无 Hard Constraint
普通完整生成
```

允许：

```text
GEM 原生 postproc
```

---

## 8.20.2 `none`

直接返回：

```text
raw GEM decoded output
```

不执行 GEM post-processing。

---

## 8.20.3 `constraint_safe`

如果存在：

```text
Hard Constraint
Segment Inpainting
Keyframe Hard Condition
```

V1 默认：

```text
postproc=False
```

防止后处理破坏固定 Motion State。

未来可单独实现：

```text
constraint-preserving postprocess
```

但必须再次检查所有 Hard Constraint。

---

# 8.21 Generation Output

Generator 不直接返回巨大的 Tensor 给 Planner。

主输出：

```python
class GenerationResult(BaseModel):

    generation_id: str

    status: Literal[
        "success",
        "partial",
        "failed",
    ]

    candidates: list[MotionCandidateSummary]

    failed_candidates: list[CandidateFailure]

    condition_fingerprint: str

    runtime: GenerationRuntime

    preflight: GenerationPreflightReport
```

---

# 8.22 `MotionCandidate`

重型结果存 Candidate Store。

```python
class MotionCandidate(BaseModel):

    candidate_id: str

    generation_id: str

    seed: int

    motion_repr_handle: str

    smpl_global_handle: str

    smpl_incam_handle: str | None

    static_conf_handle: str | None

    raw_output_handle: str | None

    metadata: CandidateMetadata
```

---

## 8.22.1 Core Output

MotionAgent 的核心 Candidate 使用：

```text
pred_body_params_global
```

而不是：

```text
pred_body_params_incam
```

因为 Pure Text 使用的是 synthetic camera。

`pred_body_params_incam` 可以保留用于：

```text
debug
render
```

但不能作为核心 Motion Evaluation 表示。

---

## 8.22.2 `motion_repr`

同时保存：

```text
pred_x
[L, 151]
```

因为后续：

```text
Segment Inpainting
Hard Condition
Debug
Ablation
```

都可能直接复用 normalized GEM Motion Representation。

---

# 8.23 Candidate Store

建议：

```text
generators/candidate_store.py
```

V1 可使用本地文件：

```text
artifacts/
  generations/
    {generation_id}/
      candidate_0.pt
      candidate_1.pt
      ...
```

Candidate Handle：

```text
generation://{generation_id}/{candidate_id}
```

Planner State 只保存 Summary。

---

## 8.23.1 Candidate Summary

```python
class MotionCandidateSummary(BaseModel):

    candidate_id: str

    seed: int

    status: str

    duration_s: float

    num_frames: int

    candidate_handle: str
```

Tournament / Verifier 根据 Handle 加载 Motion。

---

# 8.24 Candidate Fingerprint 与缓存

每个 Candidate 建立：

```text
candidate_fingerprint
```

由：

```text
checkpoint version
condition fingerprint
scope
target segments
seed
sampler config
postprocess policy
```

共同计算。

如果完全相同的 Request 再次提交：

```text
允许直接复用已有 Candidate
```

除非：

```text
force_regenerate=True
```

---

# 8.25 Condition Fingerprint

必须确保：

```text
Caption 改变
Constraint 改变
Keyframe 改变
Reference 改变
```

后：

```text
condition_fingerprint
```

发生变化。

这样：

```text
Candidate Cache
Duplicate Guard
Experiment Tracking
```

才可靠。

---

# 8.26 Model Lifecycle

禁止：

```text
每生成一个 Candidate
重新加载 GEM checkpoint
重新加载 T5
```

工业实现：

```text
Process Startup
      ↓
Load GEM
      ↓
Load T5
      ↓
Move to GPU
      ↓
eval()
      ↓
Warmup
      ↓
Ready
```

然后持续处理 Request。

---

## 8.26.1 `GEMModelManager`

```python
class GEMModelManager:

    def load():
        ...

    def warmup():
        ...

    def get_model():
        ...

    def health():
        ...
```

记录：

```text
model_version
checkpoint_path
checkpoint_hash
device
loaded_at
```

---

# 8.27 GPU Worker 模型

建议运行结构：

```text
Planner / Orchestrator
       │
       ▼
Generation Executor
       │
       ▼
   GPU Job Queue
       │
       ▼
    GEM Worker
       │
       ▼
 Candidate Store
```

第一版：

> **一个 GPU / 一个 GEM Worker / Worker 内串行 sampling。**

这样有利于：

```text
RNG 稳定
显存管理
模型复用
故障隔离
```

---

## 8.27.1 多 GPU

后续：

```text
GPU0 Worker
GPU1 Worker
GPU2 Worker
...
```

Request 在 Job 层分配。

同一 K Candidate Job 可以：

```text
单 GPU 串行
```

或者后续：

```text
按 Candidate 分发多个 GPU
```

但 Candidate 的：

```text
condition fingerprint
seed
model version
```

必须完全一致。

---

# 8.28 Runtime Error Handling

单个 Candidate 执行状态：

```python
class CandidateFailure(BaseModel):

    seed: int

    error_type: Literal[
        "oom",
        "timeout",
        "invalid_output",
        "nan",
        "runtime_error",
    ]

    message: str
```

---

## 8.28.1 Partial Success

例如：

```text
K = 4

candidate 1 success
candidate 2 success
candidate 3 OOM
candidate 4 success
```

返回：

```text
status = partial
candidates = 3
failed_candidates = 1
```

不要因为一个 Candidate 失败而丢弃全部 Job。

---

## 8.28.2 Infrastructure Retry

必须区分：

```text
Infrastructure Retry
```

和：

```text
Agent Semantic Retry
```

Infrastructure Retry：

```text
同一个 Request
同一个 Seed
同一个 Condition
```

只用于瞬时：

```text
CUDA allocation
worker transient failure
```

最多按配置重试。

它不允许：

```text
改 Prompt
改 Constraint
改 Seed
```

Semantic Retry 则必须回 Planner。

---

## 8.28.3 OOM

发生 OOM：

```text
记录错误
释放 request-local tensor
torch.cuda.empty_cache()
```

可按配置：

```text
same candidate retry once
```

如果仍失败：

```text
CandidateFailure(error_type="oom")
```

禁止 Generator 自己偷偷：

```text
减少 frame
修改 K
删除 Constraint
```

这些改变必须由上层决定。

---

# 8.29 Output Validation

每个 Candidate 生成后必须立即进行轻量技术检查：

```text
shape valid
frame count correct
no NaN
no Inf
SMPL keys present
motion_repr dimension == 151
```

接口：

```python
validate_candidate_output(
    candidate
)
```

这里只做：

```text
technical validation
```

不做：

```text
semantic quality evaluation
motion quality ranking
```

后者属于 Tournament / Verifier。

---

# 8.30 Guided Generation 接口

`strategy="guided"` 与 normal 使用同一个：

```text
GenerationRequest
MotionCandidate
GenerationResult
```

避免建立第二套 Agent 接口。

区别只在内部 Sampler。

---

## 8.30.1 Guided Sampler 输入

```python
class GuidanceContext(BaseModel):

    reward_specs: list[RewardSpec]

    semantic_reward_config: dict | None

    motion_reward_config: dict | None

    reference_ids: list[str]

    guidance_config: dict
```

---

## 8.30.2 ReAlign-style Guidance

ReAlign 的主要思想：

```text
noisy motion at timestep t
+
text
+
timestep-aware reward model
      ↓
reward
      ↓
reward gradient
      ↓
modify denoising trajectory
```

本项目中不能直接假设 ReAlign Reward Model 可以读取：

```text
GEM 151-D noisy representation
```

因为其训练 Motion Representation 可能不同。

因此需要独立：

```text
RewardAdapter
```

处理：

```text
GEM x_t
→ Reward Model Expected Motion Representation
```

或者：

```text
重新训练 GEM-representation Step-Aware Reward Model
```

在完成该验证前：

```text
semantic ReAlign guidance
```

不能视为已经可直接插拔。

---

## 8.30.3 Numerical Constraint Guidance

对于：

```text
joint target
contact
trajectory
fixed joint
```

更容易直接基于 GEM 输出构造 differentiable reward：

```text
x_t / predicted x0
      ↓
GEM decode
      ↓
FK
      ↓
Constraint Cost
      ↓
gradient
```

因此 Phase II 可以先实现：

```text
Numerical Reward Guidance
```

再接：

```text
semantic ReAlign reward
```

---

## 8.30.4 DNO-style Guidance

DNO 路径：

```text
Retrieved Motion Prior
      ↓
construct better initial noise / initialization
      ↓
Constraint Reward
      ↓
optimize diffusion noise
      ↓
GEM sampling
```

其中：

```text
Retrieval Tool
→ 只提供 Reference

Constraint Compiler
→ 提供 RewardSpec

Guided Generator
→ 执行 Noise Optimization
```

职责保持不变。

---

# 8.31 Guided Sampling 插入位置

当前 GEM：

```text
GEMDiffusion.forward_test()
      ↓
ddim_sample_loop_with_aux()
```

Phase II 推荐新增：

```text
generators/guided/gem_guided_sampler.py
```

而不是把 reward logic 写进：

```text
Motion Planner
Constraint Compiler
```

接口：

```python
def guided_sample(
    model,
    base_inputs,
    guidance_context,
    seed,
) -> GEMRawOutput:
    ...
```

---

# 8.32 Rendering 边界

Generation Tool 默认：

```text
不 Render
```

原因：

```text
Render 显著增加 latency
Tournament 可能只需要部分 representation / lightweight preview
大部分 K Candidate 最终会被淘汰
```

Candidate 先保存：

```text
SMPL
motion_repr
```

需要视觉评价时：

```text
Rendering Tool
```

单独生成：

```text
skeleton
mesh
video
```

---

# 8.33 Generation Metadata

每个 Candidate 必须保存：

```python
class CandidateMetadata(BaseModel):

    seed: int

    generation_id: str

    condition_fingerprint: str

    checkpoint_version: str

    gem_config_hash: str

    sampler: SamplerMetadata

    scope: str

    target_segments: list[int] | None

    postprocess_policy: str

    active_constraint_ids: list[str]

    active_keyframe_ids: list[str]

    active_reference_ids: list[str]

    runtime_ms: float

    peak_gpu_memory_mb: float | None
```

用于：

```text
Debug
Ablation
Reproducibility
Planner RL Dataset
```

---

# 8.34 Planner State 写回

Generation 完成后，State 只保存：

```json
{
  "generation_id": "gen_004",
  "status": "success",
  "strategy": "normal",
  "scope": "full",
  "candidate_count": 4,
  "candidate_ids": [
    "cand_0",
    "cand_1",
    "cand_2",
    "cand_3"
  ],
  "condition_fingerprint": "..."
}
```

不把：

```text
SMPL Tensor
151-D Tensor
```

塞入 Planner Context。

---

# 8.35 Planner Skill Card

建议：

```text
planner_skills/generation.md
```

至少包含：

```text
WHEN TO GENERATE

Use GENERATE when the current Motion Specification is ready
and all required conditions have been compiled.

NORMAL GENERATION

Use strategy="normal" for the first attempt whenever possible.

GUIDED GENERATION

Use strategy="guided" only when:
- compatible RewardSpec exists;
- normal sampling repeatedly failed on a measurable objective;
- guided generation capability is available.

FULL VS SEGMENT

Use scope="full" when:
- no previous accepted candidate exists;
- the failure affects the global motion plan;
- several segments must change.

Use scope="segment" when:
- a previous candidate exists;
- only one localized segment failed;
- other segments should be preserved.

IMPORTANT

GENERATION does not select the final motion.

After generation:
Tournament → Verify → Diagnosis
are executed automatically.
```

---

# 8.36 内部代码结构

建议：

```text
generators/
│
├── schemas.py
│
├── request_builder.py
├── validators.py
│
├── model_manager.py
├── gem_worker.py
├── gem_adapter.py
│
├── text_cache.py
├── condition_assembler.py
├── seed_manager.py
│
├── normal_generator.py
├── segment_inpaint.py
│
├── candidate_store.py
├── output_validator.py
│
├── guided/
│   ├── sampler.py
│   ├── reward_adapter.py
│   ├── constraint_guidance.py
│   └── dno_initializer.py
│
└── metrics.py
```

职责：

```text
schemas.py
→ Request / Result / Candidate Schema

request_builder.py
→ Planner State → GenerationRequest

validators.py
→ Preflight

model_manager.py
→ Load / Warmup / Health

gem_worker.py
→ GPU Job Execution

gem_adapter.py
→ MotionAgent Condition → GEM data / batch

text_cache.py
→ T5 embedding cache

condition_assembler.py
→ Constraint + Keyframe Hard Condition merge

seed_manager.py
→ reproducible seed

normal_generator.py
→ GEM standard inference

segment_inpaint.py
→ localized regeneration

candidate_store.py
→ tensor persistence

output_validator.py
→ technical output validation

guided/*
→ Phase II guided generation

metrics.py
→ latency / VRAM / throughput
```

---

# 8.37 Model Worker 伪代码

```python
class GEMWorker:

    def __init__(
        self,
        checkpoint,
        device,
    ):

        self.device = device

        self.model = GEMModelManager.load(
            checkpoint=checkpoint,
            device=device,
            load_text_encoder=True,
        )

        self.model.eval()

        warmup_model(
            self.model
        )

    @torch.inference_mode()
    def generate_candidate(
        self,
        prepared_input,
        seed,
        postprocess_policy,
    ):

        with seeded_rng(
            seed=seed,
            device=self.device,
        ):

            pred = run_gem_inference(
                model=self.model,
                data=prepared_input,
                postprocess_policy=postprocess_policy,
            )

        return pred
```

---

# 8.38 Generation 主流程伪代码

```python
def generate_motion(
    request: GenerationRequest,
) -> GenerationResult:

    preflight = validate_generation_request(
        request
    )

    if preflight.status != "ready":
        return blocked_result(
            request,
            preflight,
        )

    prepared_base = gem_adapter.prepare(
        request
    )

    candidates = []
    failures = []

    for seed in request.seeds:

        fingerprint = build_candidate_fingerprint(
            request=request,
            seed=seed,
        )

        cached = candidate_store.get_by_fingerprint(
            fingerprint
        )

        if cached is not None:
            candidates.append(
                cached.summary
            )
            continue

        try:

            candidate = generate_one_candidate(
                prepared_base=prepared_base,
                request=request,
                seed=seed,
            )

            validate_candidate_output(
                candidate
            )

            saved = candidate_store.save(
                candidate,
                fingerprint=fingerprint,
            )

            candidates.append(
                saved.summary
            )

        except RuntimeCandidateError as e:

            failures.append(
                CandidateFailure.from_exception(
                    seed,
                    e,
                )
            )

    if len(candidates) == request.num_candidates:
        status = "success"

    elif len(candidates) > 0:
        status = "partial"

    else:
        status = "failed"

    return GenerationResult(
        generation_id=request.generation_id,
        status=status,
        candidates=candidates,
        failed_candidates=failures,
        condition_fingerprint=
            request.condition_bundle.condition_fingerprint,
        runtime=collect_runtime_metrics(),
        preflight=preflight,
    )
```

---

# 8.39 Pure Text Adapter 伪代码

```python
def prepare_pure_text_input(
    request,
    adapter_context,
):

    L = request.total_frames

    R_w2c, cam_angvel, cam_tvel, K_fullimg = (
        build_static_camera(
            length=L,
            width=adapter_context.width,
            height=adapter_context.height,
        )
    )

    data = {
        "kp2d": torch.zeros(L, 17, 3),

        "bbx_xys": torch.zeros(L, 3),

        "K_fullimg": K_fullimg,

        "cam_angvel": cam_angvel,

        "cam_tvel": cam_tvel,

        "R_w2c": R_w2c,

        "f_imgseq": torch.zeros(L, 1024),

        "has_text": torch.tensor([True]),

        "caption":
            request.text_condition.captions[0],

        "mask": {
            "has_img_mask":
                torch.zeros(L, dtype=torch.bool),

            "has_2d_mask":
                torch.zeros(L, dtype=torch.bool),

            "has_cam_mask":
                torch.zeros(L, dtype=torch.bool),

            "has_audio_mask":
                torch.zeros(L, dtype=torch.bool),

            "has_music_mask":
                torch.zeros(L, dtype=torch.bool),
        },

        "length":
            torch.tensor(L),

        "meta": [{
            "mode": "default",

            "multi_text_data":
                build_multi_text_data(
                    request.text_condition
                ),
        }],
    }

    attach_hard_motion_condition(
        data,
        request.condition_bundle,
    )

    return data
```

---

# 8.40 Segment Inpainting 伪代码

```python
def build_segment_inpaint_condition(
    request,
    previous_candidate,
):

    previous_motion = candidate_store.load_motion_repr(
        previous_candidate
    )

    L, C = previous_motion.shape

    observed = previous_motion.clone()

    preserve_mask = torch.ones(
        L,
        C,
    )

    for segment_id in request.target_segments:

        s, e = get_segment_frames(
            segment_id
        )

        preserve_mask[s:e] = 0

    # keep body shape stable for the same identity
    beta_slice = feature_mapper.get_group_slice(
        "betas"
    )

    preserve_mask[
        :,
        beta_slice[0]:beta_slice[1]
    ] = 1

    explicit_condition = load_explicit_hard_condition(
        request.condition_bundle
    )

    observed, preserve_mask = merge_hard_conditions(
        observed,
        preserve_mask,
        explicit_condition,
    )

    return HardMotionCondition(
        values=observed,
        mask=preserve_mask,
    )
```

---

# 8.41 Candidate 输出抽取

GEM inference 后至少抽取：

```python
motion_repr = (
    pred["net_outputs"]
        ["model_output"]
        ["pred_x"][0]
)

smpl_global = (
    pred["body_params_global"]
)

smpl_incam = (
    pred.get(
        "body_params_incam"
    )
)
```

具体字段必须在 Adapter 层封装，避免业务代码到处访问：

```text
pred["net_outputs"][...]
```

推荐：

```python
gem_adapter.extract_candidate(
    raw_pred
)
```

---

# 8.42 V1 实现顺序

建议按以下顺序实现。

## Step 1：Model Manager

完成：

```text
load checkpoint once
load T5 once
eval
GPU
warmup
health check
```

---

## Step 2：Pure Text Adapter

完成：

```text
MotionSpecification
+
GEMTextCondition
→ GEM data
→ pure text generation
```

先实现：

```text
单 Candidate
full scope
no hard constraint
```

---

## Step 3：Candidate Store

保存：

```text
pred_x
global SMPL
metadata
```

---

## Step 4：Seed Reproducibility

验证：

```text
same condition + same seed
→ same candidate

same condition + different seed
→ diverse candidate
```

---

## Step 5：K Candidate Loop

实现：

```text
K seeds
→ K candidates
→ GenerationResult
```

---

## Step 6：Hard Condition

接入：

```text
observed_motion_3d
motion_mask_3d
```

---

## Step 7：Segment Inpainting

实现：

```text
previous candidate
+
target segment
→ preserve outside
→ regenerate inside
```

---

## Step 8：Postprocess Policy

区分：

```text
gem_default
none
constraint_safe
```

---

## Step 9：Worker / Queue

把：

```text
model loading
GPU execution
```

和 Planner / Orchestrator 解耦。

---

## Step 10：Guided Generation Skeleton

先实现：

```text
GenerationRequest.strategy
GuidanceContext
RewardSpec routing
feature gate
```

再实现真实：

```text
ReAlign / DNO sampling
```

---

# 8.43 V1 最低实现范围

V1 必须完成：

```text
GenerationRequest

GenerationRequestBuilder

GenerationPreflight

GEMModelManager

PureTextGEMAdapter

multi_text_data

Text Embedding Cache

Hard Motion Condition Injection

Normal Full Generation

K Candidate Sequential Sampling

Seed Reproducibility

MotionCandidate

Candidate Store

Segment Inpainting

Postprocess Policy

Technical Output Validation

GenerationResult

GPU Worker
```

V1 可以暂缓：

```text
B=K batched generation

Distributed Candidate Sampling

Semantic ReAlign Guidance

DNO Noise Initialization

Learned Sampling Scheduler

Constraint-preserving custom postprocessor

RGB Video generation
```

---

# 8.44 单元测试要求

至少覆盖：

```text
Pure Text Single Segment

Pure Text Multi-Segment

300-frame / Long Sequence

Same Seed Reproducibility

Different Seed Diversity

K Candidate Generation

Hard Constraint Injection

Keyframe Hard Condition Injection

Segment Regeneration

Outside-Segment Preservation

Body Shape Preservation

Constraint / Keyframe Conflict

Postprocess Disabled with Hard Constraint

Candidate Cache Hit

Partial Candidate Failure

OOM Handling

NaN Output Detection

Invalid Frame Length

Guided Strategy Feature Guard

Concurrent Request Isolation
```

---

## Case 1：Pure Text

```text
Input:
one text segment

Expected:
no video preprocessing
all visual masks = false
static synthetic camera
global SMPL output exists
```

---

## Case 2：Multi-Text

```text
Input:
3 temporal captions

Expected:
multi_text_data contains 3 captions
window_start / end unchanged
full sequence length correct
```

---

## Case 3：Seed

```text
same request
same seed
same checkpoint

Expected:
candidate fingerprint same
motion output reproducible
```

---

## Case 4：Hard Constraint

```text
Input:
right-arm pose hard mask at frame 90

Expected:
condition injected
constraint-safe postproc used
output technical validation passes
```

---

## Case 5：Segment Regeneration

```text
Previous Candidate:
segments 0,1,2

Target:
regenerate segment 1

Expected:
segment 0 / 2 preserved
betas preserved
segment 1 editable
explicit constraints in segment 1 remain active
```

---

## Case 6：Partial Failure

```text
K = 4
3 success
1 runtime failure

Expected:
GenerationResult.status = partial
3 candidates retained
failure metadata recorded
```

---

# 8.45 Generation Tool 运行指标

系统级至少记录：

```text
Model Load Time

Generation Latency / Candidate

Generation Latency / Job

Peak GPU Memory

GPU Utilization

Candidate Failure Rate

OOM Rate

Cache Hit Rate

Throughput

Queue Wait Time

Seed Reproducibility Rate
```

Motion 级记录：

```text
Candidate Diversity

Hard Mask Preservation Error

Outside-Segment Preservation Error

Segment Boundary Discontinuity

Raw vs Postprocessed Constraint Error
```

这些指标用于：

```text
工程优化
ablation
定位 Generator 本身问题
```

不替代后续正式 Motion Quality Evaluation。

---

# 8.46 当前模块完成标准

GEM Generation Tool 完成后，必须稳定实现：

```text
Planner GENERATE
        ↓
GenerationRequestBuilder
        ↓
GenerationPreflight
        ↓
GEMCondition Assembly
        ↓
GPU Worker
        ↓
K Seeded GEM Sampling
        ↓
MotionCandidate[]
        ↓
Candidate Store
        ↓
GenerationResult
        ↓
State
        ↓
Tournament / Verifier
```

并保证：

```text
Generator 只负责生成。

Motion Semantics
由 Motion Compiler 提供。

Constraint
由 Constraint Compiler 提供。

Keyframe
由 Keyframe Tool 提供。

Reference
必须先经过对应消费模块，
普通 Generation 不直接解释 Retrieval Result。

Tournament
负责候选选择。

Verifier
负责质量与 Constraint 检查。

Retry / Repair
由 Planner 根据 Diagnosis 决定。
```
