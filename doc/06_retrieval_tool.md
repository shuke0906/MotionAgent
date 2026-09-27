# MotionAgent for GEM：模块实现指南

> 本文件从总实现指南中拆分，对应第 6 单元。内容保持模块边界独立，其输入输出接口需与相邻模块的 Schema 保持一致。

# 6. Retrieval Tool

本节定义 Retrieval Tool 的**实现规范**。

Retrieval Tool 的职责是：

> **根据 Planner 指定的目标，从外部 Motion Corpus 中检索与当前 Motion Segment 最相关的 Caption、Motion、Pose 或 Trajectory Reference，并以结构化 Reference Handle 的形式写回 MotionAgentState。**

Retrieval Tool 是一个：

```text
Planner-controlled
+
Cross-modal Retrieval
+
Deterministic Execution Tool
```

它不是 Agent，不拥有全局决策权。

### LangGraph 执行边界

V1 中 Retrieval 由 `retrieval` LangGraph node wrapper 调用；TMR/index/reranker 等 domain logic 不依赖 LangGraph。Node 只负责构造 request、调用 service、持久化 Reference Handle、提交 state delta，然后固定返回 Planner。Retrieval 不决定 graph route。

---

## 6.1 在整体 Planner 中的定位

Retrieval 只能由 Planner 的：

```text
RETRIEVE_REFERENCE
```

Action 调用。

典型链路：

```text
Motion Compiler
      │
      └── retrieval_candidate
                │
                ▼
             Planner
                │
      RETRIEVE_REFERENCE
                │
                ▼
         Retrieval Tool
                │
       RetrievedReference[]
                │
                ▼
          Update State
                │
                ▼
             Planner
```

Retrieval Tool 执行完成后，不自动决定下一步。

Planner 根据 `purpose` 决定 Reference 如何使用。

例如：

```text
purpose = prompt_grounding
→ Planner
→ COMPILE_MOTION(mode="revise", focus=["gem_caption"])
```

```text
purpose = keyframe_source
→ Planner
→ BUILD_KEYFRAME
```

```text
purpose = constraint_source
→ Planner
→ BUILD_CONSTRAINT
```

```text
purpose = motion_prior
→ Planner
→ GENERATE
   或后续 Guided Generation
```

因此：

> Retrieval 负责“找到什么”，其他模块负责“怎么使用”。

---

# 6.2 职责边界

Retrieval Tool 负责：

```text
构造 Retrieval Query
编码 Query
搜索 Motion Corpus
过滤候选
Rerank
返回 Caption / Motion Reference
按需生成 Pose / Trajectory View
记录 Retrieval Metadata
```

Retrieval Tool 不负责：

```text
直接修改 GEM Caption
直接决定是否需要 Retrieval
直接生成 Constraint
直接决定 Hard / Soft Constraint
直接生成最终 Keyframe Condition
直接构造 observed_motion_3d
直接构造 motion_mask_3d
直接运行 GEM
直接执行 DNO / ReAlign
直接 Retry
```

所有后续动作重新交给 Planner。

---

# 6.3 主要参考方法

Retrieval 模块主要参考以下三类工作。

## 6.3.1 TMR：Text-to-Motion Retrieval

TMR 的核心作用是提供：

```text
Text
   ↓
Text Encoder
   ↓
Shared Motion-Text Latent Space
   ↑
Motion Encoder
   ↑
Motion
```

Text 与 Motion 被编码到同一个 latent space，通过 cosine similarity 进行检索。

TMR 的训练同时保留：

```text
motion reconstruction objective
+
text-motion latent alignment
+
contrastive objective
```

因此更适合：

```text
Text Query
→ Motion Retrieval
```

而不是使用普通 Sentence Embedding 去猜一个 Motion 是否匹配。

TMR 官方实现已经支持：

```text
encode_motion()
encode_text()
encode_dataset()
text-motion similarity
```

并且可以预先对整个 HumanML3D 数据集编码 Motion Latent。

本项目 V1 中：

> **TMR 作为 Motion Retrieval 的主要 embedding model。**

---

## 6.3.2 RAPO：Training-Distribution Retrieval

RAPO 在本项目中主要提供：

> 从训练数据分布中寻找与当前用户表达相关的 Caption / Modifier，再帮助 Generator-aware Prompt Rewrite。

在 MotionAgent 中不直接复制 RAPO 的视频 appearance / scene modifier。

只检索：

```text
action
body part
direction
speed
gait
motion style
transition
repetition
```

对应的训练风格 Caption。

Retrieval Tool 只负责返回这些 Caption。

真正的 Caption Rewrite 仍由：

```text
Motion Compiler
```

完成。

---

## 6.3.3 Retrieval-Guided DNO

Retrieval-Guided DNO 的关键启发是：

```text
不是所有 Constraint 都需要 Retrieval。

只有困难的时空 / 数值 Constraint
才需要寻找一个更接近目标的 Motion Prior。
```

其框架使用 LLM 做 relational task parsing，识别 difficult constraint，再检索 Reference Motion，并将 Retrieval 结果用于后续 diffusion optimization。

本项目将这一职责拆开：

```text
Motion Compiler
→ 标记 difficult / rare requirement

Planner
→ 决定是否 RETRIEVE_REFERENCE

Retrieval Tool
→ 找 Reference

Constraint / Keyframe / Guided Generator
→ 使用 Reference
```

因此 Retrieval Tool 内部不再放一个拥有全局决策能力的 LLM Agent。

---

# 6.4 V1 数据源

第一版 Retrieval Corpus 优先使用：

```text
HumanML3D
```

原因：

```text
GEM 本身使用 HumanML3D Motion/Text 数据训练
TMR 有 HumanML3D 预训练模型
HumanML3D 同时有 Caption 与 Motion
GEM repo 中已有处理后的 HumanML3D SMPL 数据
```

GEM 当前 HumanML3D Loader 使用：

```text
inputs/HumanML3D_SMPL/hmr4d_support/
humanml3d_smplhpose_{split}.pth
```

其中 Motion Record 已包含：

```text
pose
beta
trans
gender
text_data
```

即：

```text
SMPL body pose
SMPL shape
root translation
text captions
```

因此 V1 不建议重新构建一套完全独立的 Motion Corpus。

优先使用：

> **与 GEM 当前 HumanML3D 预处理结果相同的 Motion ID / SMPL 数据作为最终 Reference Motion Source。**

---

## 6.4.1 Retrieval Corpus 必须使用 Train Split

用于正式 Evaluation 时：

```text
Retrieval Index
→ 只允许使用训练集 / authorized reference corpus
```

禁止：

```text
从当前 benchmark test set
检索目标 Motion 本身
```

避免 Retrieval Data Leakage。

需要维护：

```python
split: Literal[
    "train",
    "val",
    "test"
]
```

并在正式 Evaluation 配置中强制：

```python
allowed_splits = ["train"]
```

---

# 6.5 Unified Motion Record

Retrieval Database 不应只存 Caption 字符串。

建议建立统一 Record：

```python
class MotionRecord(BaseModel):

    motion_id: str

    split: str

    captions: list[str]

    duration_s: float
    fps: int
    num_frames: int

    source: str

    smpl_handle: str

    tmr_motion_index: int | None

    metadata: dict
```

其中：

```text
smpl_handle
```

不是把 SMPL Tensor 放进 Planner State。

它只是指向：

```text
GEM-processed HumanML3D motion record
```

例如：

```text
humanml3d_train:000123
```

实际 Tensor 只有在：

```text
Keyframe Tool
Constraint Tool
Guided Generator
```

需要时才加载。

---

# 6.6 Caption Record

Caption 单独建立索引：

```python
class CaptionRecord(BaseModel):

    caption_id: str

    motion_id: str

    caption: str

    text_embedding_index: int

    source: str

    segment_start: float | None
    segment_end: float | None
```

HumanML3D 一条 Motion 可以对应多个 Caption。

因此：

```text
MotionRecord
1
│
└── N CaptionRecord
```

---

# 6.7 Offline Index 构建

Index 构建不在 Agent Runtime 中执行。

预处理脚本：

```text
scripts/build_motion_index.py
```

离线流程：

```text
HumanML3D
    │
    ├── Captions
    │
    └── Motions
          │
          ▼
Canonical Motion ID Mapping
          │
          ├─────────────┐
          ▼             ▼
 Caption Encoding   TMR Motion Encoding
          │             │
          ▼             ▼
 Caption Index      Motion Index
          │             │
          └──────┬──────┘
                 ▼
          Retrieval Store
```

---

## 6.7.1 Canonical Motion ID

必须建立统一：

```text
motion_id
```

连接：

```text
TMR / original HumanML3D motion
↔
GEM processed HumanML3D SMPL motion
```

不能假设不同 repo 的内部 index 顺序完全相同。

必须建立显式 Mapping：

```python
class MotionIdMap(BaseModel):
    canonical_motion_id: str
    tmr_keyid: str
    gem_mid: str
```

并在 Index 构建阶段验证：

```text
所有可检索 Motion
均能定位到 GEM-compatible SMPL Record
```

如果某个 TMR Motion 无法映射到 GEM SMPL：

```text
可以用于 Caption Retrieval
不能作为 SMPL Reference Motion
```

---

# 6.8 Motion Embedding：TMR

Motion Retrieval 的主要路径：

```text
Query Text
    ↓
TMR Text Encoder
    ↓
normalized text latent
    │
    ▼
Cosine Similarity
    ▲
    │
precomputed normalized TMR Motion Latents
```

TMR 官方实现的核心计算等价于：

```python
query = normalize(
    tmr_text_encoder(text)
)

scores = motion_unit_embeddings @ query
```

TMR 官方 demo 将 cosine similarity：

```text
[-1, 1]
```

映射到：

```text
[0, 1]
```

用于显示。

本项目内部建议保留原始：

```text
cosine similarity
```

作为基础 score。

---

## 6.8.1 预计算 Motion Embedding

离线：

```python
for motion in corpus:

    tmr_feat = load_tmr_motion_feature(
        motion.motion_id
    )

    emb = tmr_motion_encoder(
        tmr_feat
    )

    emb = normalize(emb)

    save(emb)
```

最终：

```text
motion_embeddings.npy
```

形状：

```text
[N_motion, D]
```

同时保存：

```text
index -> motion_id
motion_id -> index
```

---

# 6.9 Caption Embedding

Caption Retrieval 的用途主要是：

```text
prompt_grounding
```

即寻找：

```text
HumanML3D-style training captions
```

第一版可以使用两种实现。

推荐 V1：

```text
TMR Text Encoder
```

对所有 Caption 预编码。

原因：

```text
Text Embedding 与 Motion Space 已对齐
可以共用同一 Retrieval Infrastructure
```

以后可以额外增加：

```text
SentenceTransformer
BM25
Cross-Encoder
```

做 Hybrid Retrieval。

---

# 6.10 Runtime 输入接口

Retrieval Tool 主接口：

```python
def retrieve_reference(
    state: MotionAgentState,
    request: RetrievalRequest,
) -> RetrievalResult:
    ...
```

Schema：

```python
class RetrievalRequest(BaseModel):

    target_segment: int

    query: str

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

    top_k: int = 5

    filters: RetrievalFilters | None = None
```

---

## 6.10.1 `target_segment`

必须指向：

```text
state.motion_plan.segments
```

中的合法 Segment。

Retrieval 默认只针对局部 Segment。

不要把整个复杂 User Request 直接作为 Retrieval Query。

---

## 6.10.2 `query`

Planner 可以给出：

```text
query_hint
```

但 Runtime 会结合：

```text
MotionSegment DSL
```

再次构造标准 Retrieval Query。

例如：

```text
User Prompt:
一个受伤的人一瘸一拐地向前走，然后坐下
```

只检索 Segment 0：

```text
action = walk
direction = forward
style = limping, asymmetric
```

最终 Query：

```text
a person walks forward with an asymmetric limping gait
```

而不是：

```text
一个受伤的人一瘸一拐地向前走，然后坐下
```

这样不会让后面的：

```text
sit
```

污染当前 Segment 的 Retrieval。

---

# 6.11 Retrieval Query Builder

Query Builder 应采用：

```text
Planner query
+
Motion DSL
+
Retrieval purpose
```

确定性构造。

接口：

```python
def build_retrieval_query(
    segment: MotionSegment,
    planner_query: str,
    purpose: str,
) -> RetrievalQuery:
    ...
```

例如：

```python
class RetrievalQuery(BaseModel):
    text: str
    action: str
    body_parts: list[str]
    direction: str | None
    speed: str | None
    style: list[str]
    repetition: int | None
```

第一版不再增加一个额外 LLM Query Agent。

---

# 6.12 Retrieval 执行流程

Runtime 固定执行：

```text
RetrievalRequest
      │
      ▼
1. Validate Request
      │
      ▼
2. Build Structured Query
      │
      ▼
3. Encode Query
      │
      ▼
4. ANN / Cosine Search
      │
      ▼
5. Metadata Filtering
      │
      ▼
6. Structured Reranking
      │
      ▼
7. Build Requested View
      │
      ▼
8. Return RetrievalResult
```

---

# 6.13 Stage 1：Candidate Search

第一阶段只做高召回搜索。

例如：

```text
retrieve_top_n = 50
```

Motion：

```python
scores = motion_index.search(
    query_embedding,
    top_n=50,
)
```

Caption：

```python
scores = caption_index.search(
    query_embedding,
    top_n=50,
)
```

HumanML3D 规模下，第一版可以直接：

```text
PyTorch / NumPy matrix cosine
```

不强制引入 Vector Database。

如果后续 Corpus 扩大，可以切换到：

```text
FAISS
Qdrant
```

但 Retrieval API 不改变。

---

# 6.14 Stage 2：Metadata Filter

Retrieval 不应只根据一个 cosine score。

候选需要根据 Query 与 Segment 做过滤。

第一版 Filter：

```text
split filter
duration filter
invalid-motion filter
duplicate-motion filter
```

可选：

```text
action metadata filter
body-part filter
```

---

## 6.14.1 Duration Filter

如果目标 Segment：

```text
duration = 2.0s
```

而 Reference Motion：

```text
duration = 15s
```

即使文本类似，也未必适合作为：

```text
keyframe / local motion prior
```

因此可以定义：

```python
duration_ratio = reference_duration / target_duration
```

根据 `purpose` 使用不同容忍度。

例如：

```text
prompt_grounding
→ 不需要严格 duration filter

motion_prior
→ 中等 duration filter

keyframe_source
→ duration 不重要

trajectory_source
→ 较严格 duration filter
```

具体阈值通过 validation set 标定，不在代码中随意写死。

---

# 6.15 Stage 3：Structured Reranking

从 Top-N 中重新排序 Top-K。

建议 Score：

\[
S =
w_1 S_{tmr}
+
w_2 S_{action}
+
w_3 S_{body}
+
w_4 S_{style}
+
w_5 S_{duration}
-
w_6 P_{mismatch}
\]

其中：

```text
S_tmr
→ TMR semantic similarity

S_action
→ main action compatibility

S_body
→ body-part compatibility

S_style
→ style / gait compatibility

S_duration
→ duration suitability

P_mismatch
→ clear semantic conflict
```

第一版：

```text
TMR score
+
Rule-based structured compatibility
```

即可。

暂时不要求额外训练 Learned Reranker。

---

## 6.15.1 Rerank Input

```python
class RetrievalCandidate(BaseModel):
    ref_id: str
    caption: str
    motion_id: str

    tmr_score: float

    duration_s: float

    metadata: dict
```

输出：

```python
class RankedReference(BaseModel):
    ref_id: str

    retrieval_score: float

    rank: int

    compatibility: dict
```

---

# 6.16 四种 Retrieval Type

Retrieval Tool 对外支持：

```text
caption
motion
pose
trajectory
```

但底层不需要维护四个完全独立数据库。

建议：

```text
Caption
→ Caption Index

Motion
→ Motion Index

Pose
→ Retrieved Motion 的派生 View

Trajectory
→ Retrieved Motion 的派生 View
```

---

## 6.16.1 Caption Retrieval

用途：

```text
prompt_grounding
```

输出重点：

```text
HumanML3D-style captions
```

示例：

```json
{
  "ref_id": "cap_0034",
  "motion_id": "000341",
  "caption": "a person walks forward with a limp",
  "score": 0.86
}
```

Planner 后续：

```text
COMPILE_MOTION
mode="revise"
focus=["gem_caption"]
```

---

## 6.16.2 Motion Retrieval

用途：

```text
motion_prior
constraint_source
keyframe_source
```

返回：

```text
Caption
+
Motion Handle
+
Motion Metadata
```

不把完整 Tensor 放入 Planner State。

示例：

```json
{
  "ref_id": "motion_ref_012",
  "motion_id": "000341",
  "caption": "a person walks forward with a limp",
  "score": 0.89,
  "duration_s": 3.4,
  "motion_handle": "humanml3d_train:000341"
}
```

---

## 6.16.3 Pose Retrieval

Pose Retrieval 不单独训练 Pose Retriever。

第一版流程：

```text
Text Query
→ retrieve Motion Top-K
→ identify relevant phase / frame
→ return Pose Candidate Handle
```

例如：

```text
stable seated pose
```

从 retrieved sitting motion 中返回：

```text
end / stable frames
```

但：

> Retrieval Tool 只返回 Pose Candidate，不把它转换成 GEM Keyframe Condition。

真正的 Keyframe 编译仍由：

```text
Keyframe Tool
```

负责。

---

## 6.16.4 Trajectory Retrieval

第一版：

```text
Text Query
→ retrieve Motion
→ extract root translation trajectory
→ normalize / canonicalize
→ return Trajectory Reference
```

例如：

```text
curved walking trajectory
```

返回：

```text
trajectory_handle
```

真正生成：

```text
root_trajectory constraint
local_transl_vel
```

仍由：

```text
Constraint Compiler
```

负责。

---

# 6.17 GEM-compatible Motion Source

一个重要实现原则：

> Retrieval 返回的 Motion Reference 最好直接来自 GEM 使用的 HumanML3D SMPL 预处理数据，而不是运行时再把任意外部 Motion 强行转 GEM Representation。

GEM HumanML3D 数据已经包含：

```text
body_pose
betas
global_orient
transl
```

因此下游可以复用 GEM 本身的：

```text
coordinate transform
SMPL processing
EnDecoder
```

将 Reference Motion 转换到 GEM Representation。

链路：

```text
Retrieved Motion Handle
      ↓
load HumanML3D SMPL Record
      ↓
GEM-compatible preprocessing
      ↓
SMPL Params
      ↓
GEM EnDecoder
      ↓
151-D Representation
```

但是：

> 这个转换由消费该 Reference 的模块执行，不在 Retrieval Tool 内部执行。

例如：

```text
Keyframe Tool
→ load ref
→ extract pose
→ encode keyframe
```

或者：

```text
Constraint Tool
→ load ref
→ extract root / joint trajectory
→ compile constraint
```

---

# 6.18 DNO-style Motion Prior 接口

如果：

```text
purpose = motion_prior
```

Retrieval Tool 返回：

```python
RetrievedMotionReference
```

后续可以由 Guided Generator 使用。

V1：

```text
Reference 主要用于
Prompt / Keyframe / Constraint
```

Phase II：

```text
Retrieved Motion
→ reference latent / motion initialization
→ DNO-style reward-guided initialization
→ guided diffusion optimization
```

Retrieval Tool 本身不实现：

```text
noise optimization
reward gradient
diffusion update
```

这些属于：

```text
Guided Generation
```

模块。

---

# 6.19 Retrieval 输出接口

主输出：

```python
class RetrievalResult(BaseModel):

    request_id: str

    target_segment: int

    retrieval_type: str

    purpose: str

    status: Literal[
        "success",
        "low_confidence",
        "empty",
        "error",
    ]

    references: list[RetrievedReference]

    query_used: RetrievalQuery

    retrieval_meta: RetrievalMeta
```

---

## 6.19.1 `RetrievedReference`

```python
class RetrievedReference(BaseModel):

    ref_id: str

    source: str

    motion_id: str | None
    caption_id: str | None

    caption: str | None

    retrieval_score: float

    duration_s: float | None

    motion_handle: str | None
    pose_handle: str | None
    trajectory_handle: str | None

    metadata: dict
```

---

## 6.19.2 Planner State 只存 Summary

Planner 不读取完整 Motion Tensor。

State 保存：

```json
{
  "ref_id": "motion_ref_012",
  "segment": 0,
  "type": "motion",
  "purpose": "motion_prior",
  "caption": "a person walks forward with a limp",
  "score": 0.89,
  "status": "active"
}
```

完整：

```text
SMPL
Pose
Trajectory
```

通过：

```text
ref_id / handle
```

由下游 Tool 加载。

---

# 6.20 Low-confidence Handling

如果 Top-K 结果全部质量较低：

```text
Retrieval Tool 不应该硬返回“看起来最像”的 Reference 并假装可靠。
```

输出：

```json
{
  "status": "low_confidence",
  "references": []
}
```

或保留候选但标记：

```text
low_confidence = true
```

具体 threshold 不能直接使用任意固定常数。

需要在：

```text
HumanML3D validation queries
```

上校准。

Planner 收到：

```text
low_confidence
```

之后可以选择：

```text
COMPILE_MOTION
GENERATE without retrieval
BUILD_CONSTRAINT
STOP_FAILED
```

Retrieval Tool 不自行决定 fallback。

---

# 6.21 Duplicate Retrieval Guard

与 Planner 第 3 部分一致。

必须防止：

```text
same query
+
same segment
+
same corpus state
```

反复检索。

定义：

```python
retrieval_signature = hash(
    target_segment,
    normalized_query,
    retrieval_type,
    purpose,
    corpus_version,
)
```

如果：

```text
signature 已成功执行
且相关 State 未变化
```

Planner Guard 禁止重复调用。

如果：

```text
Motion Plan 改变
Constraint 改变
Verifier 提供新的 failure evidence
```

可生成新的 Retrieval Query。

---

# 6.22 Retrieval Tool 不直接“自动相信”Top-1

返回：

```text
Top-K References
```

而不是只返回 Top-1。

原因：

```text
Text-Motion Similarity
不等价于
当前 downstream control suitability
```

例如：

```text
limping walk
```

Top-1 可能语义最接近，

但 Top-3 中某个 Motion：

```text
步态更明显
时长更接近
关节轨迹更适合 Keyframe / Constraint
```

因此下游模块可以根据 `purpose` 再选择。

第一版推荐：

```text
top_k = 5
```

作为默认配置项，而不是硬编码。

---

# 6.23 Tool Skill Card

建议增加：

```text
planner_skills/retrieval.md
```

核心内容：

```text
WHEN TO USE

Use retrieval when external motion knowledge is likely to
improve the current segment, including:

- rare or unfamiliar motion;
- specialized gait or style;
- repeated semantic failure;
- reference pose needed;
- reference trajectory needed;
- difficult constraint for which a feasible motion prior is useful.

DO NOT USE

Do not retrieve for a common motion when the current Motion
Specification is already sufficient.

Do not repeat an equivalent query without new evidence.

PURPOSE

prompt_grounding
→ retrieve captions; normally return to COMPILE_MOTION.

motion_prior
→ retrieve full motion reference.

keyframe_source
→ retrieve motion / pose candidate; normally followed by BUILD_KEYFRAME.

constraint_source
→ retrieve motion / trajectory candidate; normally followed by BUILD_CONSTRAINT.

IMPORTANT

Retrieval does not itself modify the Motion Plan, create
constraints, create GEM tensors, or generate motion.
```

---

# 6.24 内部代码结构

建议：

```text
retrieval/
│
├── retriever.py
├── schemas.py
├── query_builder.py
├── index_builder.py
├── motion_store.py
├── caption_store.py
├── tmr_encoder.py
├── search_backend.py
├── filters.py
├── reranker.py
├── view_extractor.py
└── id_mapping.py
```

职责：

```text
retriever.py
→ Runtime 主入口

schemas.py
→ RetrievalRequest / Result / Reference

query_builder.py
→ Motion DSL → Retrieval Query

index_builder.py
→ Offline Index 构建

motion_store.py
→ Motion metadata / handles

caption_store.py
→ Caption metadata

tmr_encoder.py
→ TMR Text / Motion Encoding

search_backend.py
→ cosine / FAISS backend

filters.py
→ split / duration / invalid candidate filter

reranker.py
→ TMR + structured compatibility

view_extractor.py
→ pose / trajectory derived view

id_mapping.py
→ TMR ID ↔ GEM HumanML3D ID
```

---

# 6.25 主执行伪代码

```python
def retrieve_reference(
    state: MotionAgentState,
    request: RetrievalRequest,
) -> RetrievalResult:

    validate_request(
        state=state,
        request=request,
    )

    segment = state.motion_plan.get_segment(
        request.target_segment
    )

    query = build_retrieval_query(
        segment=segment,
        planner_query=request.query,
        purpose=request.purpose,
    )

    query_embedding = tmr_encoder.encode_text(
        query.text
    )

    if request.retrieval_type == "caption":
        candidates = caption_index.search(
            query_embedding,
            top_n=config.retrieve_top_n,
        )

    else:
        candidates = motion_index.search(
            query_embedding,
            top_n=config.retrieve_top_n,
        )

    candidates = apply_filters(
        candidates=candidates,
        request=request,
        segment=segment,
        allowed_splits=config.allowed_splits,
    )

    ranked = rerank(
        query=query,
        candidates=candidates,
        purpose=request.purpose,
    )

    selected = ranked[: request.top_k]

    references = []

    for candidate in selected:

        ref = build_reference(
            candidate=candidate,
            retrieval_type=request.retrieval_type,
        )

        if request.retrieval_type in [
            "pose",
            "trajectory",
        ]:
            ref = view_extractor.attach_view(
                reference=ref,
                retrieval_type=request.retrieval_type,
                segment=segment,
            )

        references.append(ref)

    status = calibrate_result_status(
        references
    )

    return RetrievalResult(
        request_id=new_request_id(),
        target_segment=request.target_segment,
        retrieval_type=request.retrieval_type,
        purpose=request.purpose,
        status=status,
        references=references,
        query_used=query,
        retrieval_meta=build_meta(...),
    )
```

---

# 6.26 与 Motion Compiler 的连接

### Prompt Grounding

```text
Motion Compiler
→ retrieval_candidate
→ Planner
→ RETRIEVE_REFERENCE(
      type="caption",
      purpose="prompt_grounding"
   )
→ Retrieved Captions
→ State
→ Planner
→ COMPILE_MOTION(
      mode="revise",
      focus=["gem_caption"]
   )
→ RAPO-style Caption Refinement
```

Retrieval Tool 不直接改：

```text
gem_caption
```

---

# 6.27 与 Constraint Compiler 的连接

例如：

```text
用户：
沿一个不稳定的跛行轨迹向前走
```

或者困难的 Contact / trajectory requirement。

链路：

```text
Compiler
→ difficult constraint hint

Planner
→ RETRIEVE_REFERENCE(
      type="motion",
      purpose="constraint_source"
   )

Retrieval Tool
→ Reference Motion Handle

Planner
→ BUILD_CONSTRAINT

Constraint Tool
→ load Reference
→ extract useful joint / root information
→ construct executable constraint
```

---

# 6.28 与 Keyframe Tool 的连接

```text
Planner
→ RETRIEVE_REFERENCE(
      type="pose",
      purpose="keyframe_source"
   )

Retrieval
→ Pose Candidate / Motion Handle

Planner
→ BUILD_KEYFRAME

Keyframe Tool
→ select / refine / encode
→ GEM-compatible Keyframe Condition
```

因此：

```text
retrieved pose
```

不等价于：

```text
final GEM keyframe tensor
```

---

# 6.29 与 GEM / Guided Generation 的连接

普通 V1：

```text
Retrieval
→ 不直接进入 GEM
```

而是通过：

```text
Caption
Keyframe
Constraint
```

间接进入 GEM。

后续 Phase II：

```text
Retrieved Motion
→ Guided Generator
→ DNO-style reference initialization / reward guidance
→ GEM sampling
```

因此 RetrievalResult 必须保留：

```text
motion_handle
```

以支持后续扩展。

---

# 6.30 V1 实现范围

第一版必须完成：

```text
HumanML3D Corpus

Caption Index

TMR Motion Index

Text → Caption Retrieval

Text → Motion Retrieval

Top-N Search

Metadata Filter

Structured Rerank

Reference Handle

Planner State Update
```

Pose / Trajectory 第一版可以先做简单 Derived View：

```text
Pose
→ candidate frame / phase pointer

Trajectory
→ root translation path
```

暂时不做：

```text
Learned Pose Retriever

Learned Reranker

多数据库 Federation

Online Continual Index

DNO Noise Initialization

Vector DB Distributed Service
```

---

# 6.31 单元测试要求

至少覆盖：

```text
Common Motion Retrieval

Rare Style Retrieval

Body-Part Specific Query

Direction-sensitive Query

Duration-sensitive Query

Caption Retrieval

Motion Retrieval

Pose View

Trajectory View

Low-confidence Retrieval

Duplicate Query Guard

Train/Test Leakage Guard

ID Mapping Failure
```

---

## Case 1：Rare Style

```text
Input Segment:
walk forward with a limping asymmetric gait

Request:
type = motion
purpose = motion_prior

Expected:
Top-K motions semantically related to limping / uneven gait
No unrelated sitting / jumping motion
```

---

## Case 2：Prompt Grounding

```text
Request:
type = caption
purpose = prompt_grounding

Expected:
HumanML3D-style concise action captions

Next Planner Action:
normally COMPILE_MOTION(revise gem_caption)
```

---

## Case 3：Keyframe Source

```text
Input:
final state should be seated

Request:
type = pose
purpose = keyframe_source

Expected:
Reference from sitting / seated motions

No GEM Keyframe Tensor created inside Retrieval Tool
```

---

## Case 4：Leakage

```text
Evaluation sample belongs to test split

Expected:
test sample cannot appear in Retrieval Index
```

---

# 6.32 Retrieval 评估指标

除最终 Motion Generation 指标外，Retrieval 模块单独记录：

```text
Recall@K
Median Rank
Top-K semantic relevance
Structured compatibility rate
Low-confidence rate
Retrieval latency
Duplicate retrieval rate
Downstream utilization rate
```

对于 Agent，还要记录：

```text
Retrieval Call Success Rate

Retrieval → Compiler improvement

Retrieval → Constraint success

Retrieval → Keyframe success

Retrieval → Guided Generation success
```

以判断：

> Planner 调用了 Retrieval 之后，Reference 是否真的帮助了后续任务。

---

# 6.33 当前模块完成标准

Retrieval Tool 完成后，必须稳定实现：

```text
Planner RETRIEVE_REFERENCE
        ↓
Structured Segment Query
        ↓
TMR / Caption Search
        ↓
Filter + Rerank
        ↓
Top-K RetrievedReference
        ↓
State
        ↓
Planner
```

并保证：

```text
Retrieval 只获取外部 Reference。

Retrieval 不修改 Motion Plan。

Retrieval 不自动生成 Constraint。

Retrieval 不自动生成 Keyframe。

Retrieval 不直接修改 GEM Input。

Reference 如何被使用，
由 Planner 的下一步 Action 决定。
```
