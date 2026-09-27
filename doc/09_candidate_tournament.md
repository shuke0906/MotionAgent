# MotionAgent for GEM：模块实现指南

> 本文件对应第 9 单元 Candidate Tournament。  
> 本模块必须与 `03_motion_planner.md` 中定义的 Automatic Post-Generation Subgraph，以及 `08_gem_generation_tool.md` 的 `GenerationResult / MotionCandidateSummary / Candidate Store` 保持接口一致。
>
> 本模块只负责：
>
> **从已经生成的候选 Motion 中进行相对选择，返回一个 Champion。**
>
> 它不负责最终验收，也不负责 Repair。

# 9. Candidate Tournament

本节定义 Candidate Tournament 的**实现规范**。

Candidate Tournament 的职责是：

> **接收 GEM Generation Tool 返回的多个合法 Motion Candidate，通过可复现、抗位置偏差、可审计的 Pairwise Tournament，从当前候选集合中选出一个 Champion，交给 Multi-Verifier 做后续绝对验证。**

Candidate Tournament 是：

```text
Automatic Post-Generation Subgraph
+
Relative Candidate Selection
+
Pairwise Judge Service
```

它不是 Planner Action，也不拥有全局控制权。

### LangGraph 执行边界

V1 中本模块是 `post_generation` LangGraph subgraph 的固定 `tournament` node。它由 graph edge 自动进入，并固定流向 `verifier`；这条 edge 不由 Planner/LLM 每轮重新选择。模块内部算法和 typed service interface 保持 framework-agnostic。

---

## 9.1 在整体 Planner 中的定位

根据第 3 单元的规定，Planner 不直接调用：

```text
TOURNAMENT
VERIFY
DIAGNOSE
```

Planner 只输出：

```text
GENERATE
```

之后 Orchestrator 自动执行：

```text
Planner
   │
   └── GENERATE
          │
          ▼
   GEM Generation Tool
          │
          ▼
    GenerationResult
          │
          ▼
 ┌──────────────────────┐
 │ Candidate Tournament │
 └──────────────────────┘
          │
          ▼
      Champion
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

因此 Candidate Tournament：

```text
不出现在 Planner Action Space 中
```

也不会返回：

```text
ACCEPT / REPAIR / GENERATE
```

这种全局动作。

它只返回：

```text
Champion Candidate
+
Tournament Evidence
```

---

# 9.2 与第 8 单元 Generation Tool 的接口

Generation Tool 输出：

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

Tournament 只读取：

```text
GenerationResult.candidates
```

中的成功 Candidate。

真实 Motion Tensor 通过：

```text
candidate_handle
```

从 Candidate Store 加载。

禁止把：

```text
151-D Tensor
SMPL Tensor
```

直接塞进 TournamentRequest JSON。

---

# 9.3 与第 10 单元 Multi-Verifier 的职责边界

必须严格区分：

```text
Tournament
= relative selection

Verifier
= absolute verification
```

Tournament 回答：

> **A 和 B 哪一个更适合继续进入验证？**

Verifier 回答：

> **Champion 是否真的满足任务？如果不满足，具体哪里失败？**

因此 Tournament 不负责：

```text
最终 overall_pass

最终 semantic pass/fail

最终 constraint pass/fail

failure localization

repair recommendation

Planner routing
```

即使 Tournament 选出了：

```text
当前 K 个候选中最好的 Motion
```

它仍然可能：

```text
全部不合格
```

例如：

```text
Candidate A contact error = 0.20m
Candidate B contact error = 0.15m
Candidate C contact error = 0.30m

Tournament
→ B wins

Verifier
→ 0.15m > 0.05m threshold
→ FAIL
```

这是合法行为。

---

# 9.4 主要参考方法

## 9.4.1 VISTA：Pairwise Tournament Selection

本模块的核心选择框架参考 VISTA。

VISTA 不采用：

```text
每个 Candidate 独立打一个 0~10 分
→ 直接取最高分
```

而采用：

```text
Candidate Pair
→ Relative Comparison
→ Winner advances
→ Binary Tournament
```

并使用：

```text
Forward:
A vs B

Swapped:
B vs A
```

双向比较降低：

```text
position / token order bias
```

VISTA 还先对每个 Candidate 生成：

```text
probing critique
```

再进行两两比较。

这意味着 Judge 不需要同时完成：

```text
第一次理解 Candidate
+
发现缺陷
+
比较两个 Candidate
```

三个任务。

本项目保留这一核心设计。

---

## 9.4.2 VISTA 原始 Criteria 的 Motion Adaptation

VISTA 原始视频选择维度包括：

```text
Visual Fidelity
Physical Commonsense
Text-Video Alignment
Audio-Video Alignment
Engagement
```

不能原样搬到 GEM Motion。

MotionAgent 改成：

```text
Semantic Alignment
Event / Temporal Alignment
Motion Naturalness
Constraint Compliance
Continuity / Preservation
```

其中根据当前任务动态启用。

---

## 9.4.3 AToM：Event-Level Comparison

AToM 专门研究：

```text
Integrity
Temporal Relationship
Frequency
```

三类 event-level Text-to-Motion alignment。

因此 Tournament 的：

```text
Event / Temporal Alignment
```

可以采用 AToM-style motion visualization + VLM comparison。

例如：

```text
required:
wave right hand 3 times

Candidate A:
3 times

Candidate B:
2 times

→ Candidate A wins event criterion
```

但这里只做：

```text
relative selection
```

最终 Frequency 是否严格通过：

```text
由 Multi-Verifier 决定
```

---

## 9.4.4 MotionCritic：Naturalness Evidence

MotionCritic 学习 Human Motion 的：

```text
naturalness
smoothness
plausibility
```

并与 Human Perception 对齐。

因此 V1 可以将：

```text
MotionCritic score
```

作为：

```text
Motion Naturalness
```

的辅助选择证据。

但：

```text
MotionCritic score
```

不能单独决定 Tournament Winner。

---

## 9.4.5 TMR：Semantic Evidence

TMR 提供：

```text
Text
↔
Motion
```

共同 Latent Space。

Tournament 可以使用：

```text
TMR similarity
```

作为 Semantic Alignment 的 cheap evidence。

同样：

```text
TMR score
```

是辅助 evidence，不是最终 Judge。

---

# 9.5 Tournament 的完整执行链路

固定流程：

```text
GenerationResult
      │
      ▼
1. TournamentRequestBuilder
      │
      ▼
2. Tournament Preflight
      │
      ▼
3. Candidate Preparation
      │
      ├── Motion Load
      ├── Lightweight Render
      └── Cheap Metrics
      │
      ▼
4. Candidate Probing
      │
      ▼
5. Build Deterministic Bracket
      │
      ▼
6. Pairwise Round
      │
      ├── A vs B
      └── B vs A
      │
      ▼
7. Resolve Match
      │
      ├── consistent → winner
      └── inconsistent → tiebreak
      │
      ▼
8. Repeat Until One Candidate
      │
      ▼
9. TournamentResult
      │
      ▼
Update State.champion
      │
      ▼
Multi-Verifier
```

---

# 9.6 输入接口：`TournamentRequest`

主接口：

```python
def run_tournament(
    request: TournamentRequest,
) -> TournamentResult:
    ...
```

推荐 Schema：

```python
class TournamentRequest(BaseModel):

    tournament_id: str

    generation_id: str

    candidate_ids: list[str]

    incumbent_candidate_id: str | None

    original_request: str

    motion_spec_id: str

    condition_fingerprint: str

    generation_scope: Literal[
        "full",
        "segment",
    ]

    target_segments: list[int] | None

    selection_policy: TournamentSelectionPolicy

    judge_config: JudgeConfig

    render_profile: TournamentRenderProfile

    bracket_seed: int

    timeout_s: float | None
```

---

## 9.6.1 `candidate_ids`

来自：

```text
GenerationResult.candidates
```

必须全部：

```text
status == success
```

Candidate Tensor 从：

```text
Candidate Store
```

加载。

---

## 9.6.2 `incumbent_candidate_id`

首次 Generation：

```text
None
```

Repair / Retry 后：

```text
可以加入上一轮 Champion
```

作为 incumbent。

这样实现 VISTA-style：

> **新 Candidate 必须真正击败旧 Champion，才能替换当前最佳结果。**

典型：

```text
previous champion
      +
new K candidates
      ↓
Tournament
      ↓
new champion or old champion
```

有利于：

```text
monotonic improvement
```

---

## 9.6.3 Incumbent 可比较条件

不是所有旧 Champion 都可以直接加入。

必须满足：

```text
same original task

same motion skeleton / representation

same fps

same total_frames
或存在明确 temporal alignment

candidate file still available
```

如果：

```text
Motion Plan 发生结构性变化
总时长发生变化
任务被重新定义
```

则：

```text
incumbent_candidate_id = None
```

不能强行比较。

---

# 9.7 `TournamentSelectionPolicy`

```python
class TournamentSelectionPolicy(BaseModel):

    criteria: list[SelectionCriterion]

    use_probe: bool = True

    use_bidirectional_compare: bool = True

    use_quick_metrics: bool = True

    allow_metric_prefilter: bool = False

    max_candidates_after_prefilter: int | None

    include_incumbent: bool = True

    tie_break_policy: Literal[
        "meta_judge_then_metrics",
        "metrics_only",
    ] = "meta_judge_then_metrics"
```

---

# 9.8 Selection Criteria

统一 Schema：

```python
class SelectionCriterion(BaseModel):

    name: Literal[
        "semantic_alignment",
        "event_alignment",
        "motion_naturalness",
        "constraint_compliance",
        "continuity_preservation",
    ]

    enabled: bool

    weight: float

    critical: bool

    evidence_sources: list[str]
```

---

## 9.8.1 默认 Criteria

所有任务：

```text
semantic_alignment

motion_naturalness
```

如果有：

```text
multiple events
explicit ordering
repetition
```

增加：

```text
event_alignment
```

如果：

```text
CompiledConstraint[]
不为空
```

增加：

```text
constraint_compliance
```

如果：

```text
scope="segment"
```

增加：

```text
continuity_preservation
```

---

## 9.8.2 Cross-Segment Heading Continuity

`continuity_preservation` 不再只在 `scope="segment"` 时启用。

如果当前 `MotionSpecification` 中存在：

```text
HeadingContinuitySpec.mode in {
  inherit_previous,
  explicit,
  follow_trajectory
}
```

则 Full Generation 也启用：

```text
continuity_preservation
```

这一 Criterion 只比较 Candidate 是否遵守已经存在的 continuity expectation，不自行推断新的 facing requirement。

# 9.9 Criteria 不能由 Judge 随意创造

Selection Criteria 必须在：

```text
TournamentRequestBuilder
```

阶段根据 State 确定。

Judge 可以：

```text
评价 Criteria
```

但不能自己决定：

```text
忽然增加 aesthetics
忽然增加 character attractiveness
忽然忽略 explicit constraint
```

避免 Judge drift。

---

# 9.10 `TournamentRequestBuilder`

接口：

```python
def build_tournament_request(
    state: MotionAgentState,
    generation_result: GenerationResult,
) -> TournamentRequest:
    ...
```

流程：

```text
1. 读取 GenerationResult
2. 收集成功 Candidate
3. 检查是否存在可比较 Incumbent
4. 根据 MotionSpecification 构造 Criteria
5. 根据 ConstraintBundle 加入 Constraint Criterion
6. 根据 generation scope 加入 Continuity Criterion
7. 固定 Judge Config
8. 固定 Render Profile
9. 生成 deterministic bracket_seed
10. 输出 TournamentRequest
```

---

# 9.11 Tournament Preflight

GPU / MLLM Judge 调用前必须验证：

```python
def validate_tournament_request(
    request: TournamentRequest,
) -> TournamentPreflightReport:
    ...
```

---

## 9.11.1 必须阻止的情况

```text
candidate_ids == []

Candidate Store 找不到 Candidate

Candidate technical validation 未通过

Motion Representation 不兼容

没有 Original Request

没有 Motion Specification

Judge Config 不可用

Render Profile 不合法
```

---

## 9.11.2 只有一个 Candidate

如果：

```text
len(candidate_ids) == 1
```

不需要伪造 Tournament。

直接：

```text
Champion = candidate
status = single_candidate
```

然后继续：

```text
Multi-Verifier
```

不能因为只有一个 Candidate 就：

```text
自动 ACCEPT
```

---

## 9.11.3 Generation Partial Success

例如：

```text
Planner 请求 K = 4

Generation:
3 success
1 failed
```

Tournament：

```text
只比较 3 个成功 Candidate
```

失败 Candidate 不进入 bracket。

---

# 9.12 Candidate Preparation

Tournament 不直接让 MLLM 读取：

```text
151-D Motion Tensor
```

需要先构造：

```python
class CandidateSelectionPacket(BaseModel):

    candidate_id: str

    candidate_handle: str

    render_bundle_handle: str | None

    quick_metrics: dict

    probe_id: str | None

    metadata: dict
```

---

# 9.13 Lightweight Render

第 8 单元明确规定：

```text
Generation Tool 默认不 Render
```

因此 Tournament 如果使用 MLLM Judge：

```text
Tournament
→ Rendering Tool
```

按需生成轻量 Visual Representation。

---

## 9.13.1 为什么不能给每个 Candidate 用不同 Camera

如果 Candidate A：

```text
camera zoom in
```

Candidate B：

```text
camera zoom out
```

MLLM 可能因为可见尺度不同产生评价偏差。

因此同一 Tournament 必须：

> **所有 Candidate 使用同一 Render Profile 与统一 Framing。**

---

## 9.13.2 推荐 `TournamentRenderProfile`

```python
class TournamentRenderProfile(BaseModel):

    mode: Literal[
        "dual_view",
        "global_view",
        "skeleton_only",
    ]

    width: int

    height: int

    fps: int

    background: str

    show_ground: bool

    show_joint_markers: bool

    show_segment_boundaries: bool
```

V1 推荐：

```text
dual_view
```

包括：

```text
Global View
+
Root-Centered Local View
```

---

## 9.13.3 Global View

用于观察：

```text
root trajectory
overall displacement
contact / spatial behavior
global body motion
```

所有 Candidate 使用：

```text
同一 world bounds
同一 camera
同一 scale
```

World Bounds 建议根据：

```text
所有 Candidate 的 union bounding box
```

统一计算。

---

## 9.13.4 Local View

用于观察：

```text
body articulation
gait
arm / leg motion
pose naturalness
```

使用：

```text
root-centered camera
```

但不能改变 Motion 时间轴。

---

# 9.14 Render Cache

Render Key：

```text
candidate_id
+
render_profile_hash
```

同一个 Candidate 在：

```text
Tournament
Verifier
```

需要相同 Render 时可复用。

避免重复渲染。

---

# 9.15 Cheap Metric Evidence

Candidate Probing 前，可以计算便宜的结构化 Evidence。

建议 V1：

```text
TMR similarity

MotionCritic score

Constraint quick metrics

Segment preservation error
```

---

## 9.15.1 Semantic Evidence

```text
TMR
```

输入：

```text
Motion Specification canonical caption
+
Candidate Motion
```

输出：

```text
semantic_similarity
```

---

## 9.15.2 Naturalness Evidence

```text
MotionCritic
```

输入：

```text
Candidate SMPL
```

输出：

```text
perceptual motion score
```

---

## 9.15.3 Constraint Evidence

直接复用第 7 单元：

```text
VerificationSpec
```

做 cheap numeric evaluation。

例如：

```text
joint error
trajectory RMSE
contact distance
fixed joint drift
```

这里只用于：

```text
Candidate relative comparison
```

最终 threshold pass/fail 仍由第 10 单元执行。

---

## 9.15.4 Continuity Evidence

仅：

```text
scope="segment"
```

时启用。

计算：

```text
left boundary pose jump

right boundary pose jump

root velocity discontinuity

outside-target preservation error
```

因为 Segment Regeneration 的目标之一是：

> **修复局部问题，同时保留已经通过的 Motion。**

---

## 9.15.5 Heading Continuity Evidence

对 `inherit_previous` / `explicit` heading expectation，Quick Metric Service 可以计算：

```text
heading_drift_mean_deg
heading_drift_p95_deg
heading_drift_max_deg
uncommanded_turn_duration_ratio
```

计算规则：

```text
1. 从 `smpl_global` root orientation 得到 body forward vector；
2. 使用 GEM / SMPL canonical forward axis，不在 Tournament 中硬编码另一套坐标系；
3. 投影到 ground plane；
4. anchor heading 来自指定 anchor segment 结束附近的稳定窗口；
5. 显式 turn / spin / transition window 不计入“无指令转向”惩罚；
6. 比较 Candidate 时只作为 relative evidence，最终 PASS/FAIL 仍由 Multi-Verifier 决定。
```

Pairwise Judge 应优先选择：

```text
在满足动作语义的前提下，heading drift 更小、跨段朝向更自然连续的 Candidate。
```

不能因为一个 Candidate 完全僵硬不转身，就自动认为它更好；Naturalness 与 Semantic Alignment 仍然同时生效。

# 9.16 Candidate Probe

参考 VISTA 的：

```text
probing critiques
```

在真正 Pairwise Compare 之前，每个 Candidate 独立分析一次。

接口：

```python
def probe_candidate(
    packet: CandidateSelectionPacket,
    context: TournamentJudgeContext,
) -> CandidateProbe:
    ...
```

---

## 9.16.1 `CandidateProbe`

```python
class CandidateProbe(BaseModel):

    probe_id: str

    candidate_id: str

    criterion_findings: list[CriterionFinding]

    strengths: list[str]

    weaknesses: list[str]

    critical_flags: list[str]

    confidence: float

    judge_version: str
```

---

## 9.16.2 `CriterionFinding`

```python
class CriterionFinding(BaseModel):

    criterion: str

    assessment: Literal[
        "strong",
        "acceptable",
        "weak",
        "uncertain",
    ]

    evidence: list[str]

    metric_evidence: dict
```

---

# 9.17 Probe 的目标

Probe 不决定 Winner。

只回答：

```text
这个 Candidate 在每个 Selection Criterion 上表现如何？

有什么明确优点？

有什么明确缺陷？

有哪些地方无法可靠判断？
```

这样 Pairwise Judge 可以直接比较：

```text
Candidate A Probe
vs
Candidate B Probe
```

而不是从零分析两个视频。

---

# 9.18 Probe Prompt

第一版可使用：

```text
You are the Motion Candidate Probe for MotionAgent.

Your task is to inspect ONE generated human motion candidate
against the current motion specification.

Do not choose a winner.
Do not decide whether the motion should be accepted.
Do not propose repairs.

Evaluate only the provided selection criteria.

Use:
- the original user request;
- the structured Motion Specification;
- the candidate motion visualization;
- the provided deterministic metric evidence.

For each criterion:
1. identify concrete strengths;
2. identify concrete weaknesses;
3. identify uncertainty;
4. preserve event order, repetition, body-part identity,
   direction, and explicit constraints from the specification.

Do not reward:
- visual attractiveness unrelated to motion;
- camera framing;
- clothing;
- background;
- rendering aesthetics.

Return only CandidateProbe.
```

---

# 9.19 Pairwise Judge 输入

```python
class PairwiseJudgeRequest(BaseModel):

    match_id: str

    original_request: str

    motion_spec_summary: dict

    criteria: list[SelectionCriterion]

    candidate_a: CandidateSelectionPacket

    candidate_b: CandidateSelectionPacket

    probe_a: CandidateProbe

    probe_b: CandidateProbe

    presentation_order: Literal[
        "AB",
        "BA",
    ]
```

---

# 9.20 Pairwise Judge 输出

```python
class PairwiseVerdict(BaseModel):

    match_id: str

    winner_candidate_id: str | None

    loser_candidate_id: str | None

    outcome: Literal[
        "A",
        "B",
        "tie",
        "uncertain",
    ]

    criteria_results: list[PairwiseCriterionResult]

    critical_penalties: dict

    confidence: float

    summary: str

    judge_version: str
```

---

## 9.20.1 Criteria Result

```python
class PairwiseCriterionResult(BaseModel):

    criterion: str

    winner: Literal[
        "A",
        "B",
        "tie",
    ]

    confidence: float

    evidence: list[str]
```

---

# 9.21 Pairwise Judge Prompt

```text
You are the Pairwise Motion Judge for MotionAgent.

You are comparing exactly two candidate motions generated
for the same user task.

Your goal is NOT to decide whether either motion is absolutely
good enough.

Your goal is to decide which candidate should advance to the
next tournament round.

Use only the provided selection criteria.

Important rules:

1. Preserve the original user intent.

2. Prefer the candidate that better satisfies the structured
   Motion Specification.

3. Do not reward rendering aesthetics, camera quality,
   clothing, background, or visual attractiveness.

4. Use deterministic metric evidence when it directly measures
   a requirement.

5. For event-level requirements, consider:
   - event integrity;
   - temporal order;
   - repetition / frequency.

6. For motion quality, consider:
   - naturalness;
   - smoothness;
   - physical plausibility.

7. For explicit constraints, prefer the candidate with lower
   relevant constraint error, all else being comparable.

8. For segment regeneration, penalize unnecessary changes to
   already preserved regions.

9. A candidate may win overall even if both candidates contain
   flaws. Absolute acceptance is handled later by Multi-Verifier.

10. Return criterion-level comparisons before the overall winner.

Return only PairwiseVerdict.
```

---

# 9.22 Bidirectional Pairwise Comparison

每一对：

```text
Candidate X
Candidate Y
```

必须执行：

```text
Forward:
X as A
Y as B

Swapped:
Y as A
X as B
```

得到：

```text
verdict_forward

verdict_swapped
```

然后把：

```text
A / B
```

统一映射回：

```text
candidate_id
```

---

## 9.22.1 Consistent

例如：

```text
Forward:
X wins

Swapped:
X wins
```

则：

```text
winner = X
resolution = bidirectional_consensus
```

---

## 9.22.2 Inconsistent

例如：

```text
Forward:
X wins

Swapped:
Y wins
```

说明：

```text
Judge 对位置敏感
或证据不足
```

不能直接随机决胜。

---

# 9.23 与 VISTA 原方法的差异

VISTA 在：

```text
forward
和
swapped
```

结果不一致时允许：

```text
random assignment
```

MotionAgent 工业实现不采用该策略。

原因：

```text
不可复现
难调试
不利于 ablation
会污染 Planner trajectory
```

本项目使用：

```text
Inconsistent
      ↓
TieBreak Meta Judge
      ↓
Metric Fallback
      ↓
Deterministic Final Fallback
```

---

# 9.24 Tie-Break Meta Judge

输入：

```text
Candidate Probes
Forward Verdict
Swapped Verdict
Quick Metrics
Selection Criteria
```

不直接复用之前：

```text
A / B label
```

候选在 Tie-Break 中使用重新匿名化 Label。

接口：

```python
def resolve_inconsistent_pair(
    match: PairwiseMatch,
) -> MatchResolution:
    ...
```

---

## 9.24.1 TieBreak 输出

```python
class TieBreakVerdict(BaseModel):

    winner_candidate_id: str | None

    outcome: Literal[
        "winner",
        "tie",
        "uncertain",
    ]

    confidence: float

    decisive_criteria: list[str]

    summary: str
```

---

# 9.25 Metric Fallback

如果：

```text
TieBreak Meta Judge
```

仍无法稳定选择，

使用：

```text
normalized quick metric evidence
```

做 deterministic fallback。

---

## 9.25.1 不允许直接平均原始 Metric

例如：

```text
TMR = 0.72

MotionCritic = 4.1

Contact Error = 0.03 m
```

这些量纲不同。

必须先：

```text
normalize
+
direction correction
```

得到：

```text
higher-is-better [0,1]
```

再计算：

\[
S_i =
\frac{
\sum_c w_c s_{i,c}
}{
\sum_c w_c
}
\]

---

## 9.25.2 Metric Availability

如果某个 Metric：

```text
不可用
```

不填：

```text
0
```

而是从：

```text
denominator
```

中移除其 weight。

---

# 9.26 Final Deterministic Fallback

如果：

```text
Judge tie
+
Metric tie
```

仍无法区分，

使用：

```text
stable hash
```

而不是 RNG。

例如：

```python
winner = min(
    candidate_ids,
    key=lambda x: stable_hash(
        tournament_id,
        x,
    )
)
```

必须记录：

```text
resolution = deterministic_fallback
```

这种 Match 应在 Evaluation 中单独统计。

---

# 9.27 Critical Penalty

参考 VISTA 的：

```text
customizable penalty mechanism
```

MotionAgent 支持：

```text
critical criterion penalty
```

例如：

```text
explicit constraint severe violation

missing required event

gross outside-segment corruption
```

但 Tournament 不自行定义 Critical Requirement。

它来自：

```text
Motion Specification
Compiled Constraint
SelectionPolicy
```

---

## 9.27.1 Penalty 的作用

Penalty 只影响：

```text
Pairwise Selection
```

不等价于：

```text
Verifier fail
```

例如：

```text
Candidate A:
semantic excellent
contact error 0.30 m

Candidate B:
semantic good
contact error 0.04 m

constraint criterion = critical

→ B should strongly win
```

---

# 9.28 Bracket 构建

不能直接按：

```text
candidate_0
candidate_1
candidate_2
candidate_3
```

固定相邻配对。

否则 Seed / Generation Order 可能影响 bracket。

---

## 9.28.1 Deterministic Shuffle

使用：

```text
bracket_seed
+
candidate_id
```

构造稳定顺序。

例如：

```python
ordered = sorted(
    candidate_ids,
    key=lambda cid: stable_hash(
        request.bracket_seed,
        cid,
    )
)
```

这样：

```text
相同 Request
→ 相同 Bracket

不同 Experiment Seed
→ 可测试 Bracket Stability
```

---

# 9.29 Odd Candidate Count

例如：

```text
K = 3
```

一轮中：

```text
1 pair
+
1 bye
```

Bye Candidate 自动进入下一轮。

Bye 的分配由：

```text
deterministic bracket
```

决定，

不能人工偏向：

```text
incumbent
candidate_0
```

---

# 9.30 Incumbent Champion

Repair 之后，为避免：

```text
新一轮生成反而退化
```

可以把上一轮：

```text
state.champion
```

加入 Tournament。

对应 VISTA Self-Improvement 中：

```text
previous winner
+
new candidates
→ re-selection
```

---

## 9.30.1 Incumbent 的意义

例如：

```text
Round 1 champion:
semantic 0.9
contact bad

Repair:
add contact constraint

Round 2 new candidates:
contact better
但 naturalness 变差
```

Tournament 可以判断：

```text
新 Candidate 是否真的整体优于旧 Champion
```

而不是无条件覆盖。

---

# 9.31 Candidate Probe Cache

Probe 成本较高。

Cache Key：

```text
candidate_id
+
criteria_fingerprint
+
render_profile_hash
+
probe_model_version
```

如果同一 Candidate 被：

```text
新的 Tournament
```

再次使用且 Criteria 没变：

```text
直接复用 CandidateProbe
```

---

# 9.32 Pairwise Verdict Cache

Cache Key：

```text
unordered(candidate_a, candidate_b)
+
criteria_fingerprint
+
judge_version
+
probe_version
```

必须同时记录：

```text
forward
swapped
```

结果。

如果 Judge Version / Prompt Version 改变：

```text
Cache invalid
```

---

# 9.33 `criteria_fingerprint`

由：

```text
Selection Criteria

Motion Specification Version

Constraint VerificationSpec Version

Target Segment

Generation Scope
```

计算。

如果：

```text
Prompt / Constraint / Timeline
```

发生变化，

不能复用旧 Pairwise Verdict。

---

# 9.34 服务实现

建议：

```text
Orchestrator
     │
     ▼
Tournament Service
     │
     ├── Candidate Store
     ├── Rendering Service
     ├── Quick Metric Service
     ├── Probe Judge Provider
     ├── Pairwise Judge Provider
     └── Tournament Store
```

---

## 9.34.1 Tournament Service

对外接口：

```python
class TournamentService:

    async def run(
        self,
        request: TournamentRequest,
    ) -> TournamentResult:
        ...
```

---

## 9.34.2 Rendering Service

```python
async def render_for_tournament(
    candidate_ids,
    render_profile,
) -> dict[str, RenderBundleHandle]:
    ...
```

---

## 9.34.3 Quick Metric Service

```python
async def compute_selection_evidence(
    candidate_id,
    criteria,
) -> QuickMetricEvidence:
    ...
```

---

## 9.34.4 Judge Provider

```python
class TournamentJudgeProvider:

    async def probe(
        self,
        request,
    ) -> CandidateProbe:
        ...

    async def compare(
        self,
        request,
    ) -> PairwiseVerdict:
        ...

    async def tiebreak(
        self,
        request,
    ) -> TieBreakVerdict:
        ...
```

可以替换具体：

```text
MLLM Provider
```

但上层 Tournament API 不变。

---

# 9.35 并行执行

Candidate Probe 之间互相独立：

```text
Candidate 0 Probe
Candidate 1 Probe
Candidate 2 Probe
Candidate 3 Probe
```

可以并行。

同一 Tournament Round 内的 Match：

```text
A vs B
C vs D
```

也可以并行。

但 Round 之间：

```text
Round 2
```

依赖 Round 1 Winner，

必须顺序执行。

---

# 9.36 Cost Control

VISTA 的分析显示 Pairwise Tournament 是明显的推理成本来源之一。

因此工业实现需要：

```text
Cache

Limited K

Parallel Probe

Round Parallelism

Lightweight Rendering

Optional Metric Prefilter
```

---

## 9.36.1 V1 推荐

如果：

```text
K <= 4~6
```

不做 metric prefilter。

直接 Tournament。

原因：

```text
过早 prefilter
可能把 Judge 会喜欢的 Candidate 错误删掉
```

---

## 9.36.2 K 较大

如果后续：

```text
K >> 8
```

可以：

```text
Quick Metrics
→ Top-M
→ Pairwise Tournament
```

但该功能默认关闭。

需要单独做：

```text
prefilter recall ablation
```

---

# 9.37 Tournament Complexity

Binary Tournament 有：

```text
N - 1
```

个 Match。

每个 Match：

```text
Forward
+
Swapped
```

所以基础 Pairwise Judge Call：

\[
2(N-1)
\]

另外还有：

```text
N Candidate Probe
```

和少量：

```text
TieBreak Calls
```

例如：

```text
N = 4
```

基础：

```text
4 probes
+
6 pairwise judge calls
```

---

# 9.38 Failure Handling

Tournament 必须区分：

```text
candidate failure

render failure

metric failure

judge failure

schema failure

timeout
```

---

## 9.38.1 Judge Schema Failure

如果 Judge 返回：

```text
非法 JSON
非法 candidate_id
非法 criterion
```

允许：

```text
same request
same presentation
```

做有限次数：

```text
schema retry
```

不能改变：

```text
candidate
criteria
```

---

## 9.38.2 Judge Timeout

优先：

```text
retry same call
```

如果仍失败：

```text
fallback to metrics
```

并记录：

```text
judge_degraded = true
```

---

## 9.38.3 Render Failure

如果：

```text
VLM render unavailable
```

但：

```text
Quick Metrics
```

足够，

允许：

```text
metric-only degraded mode
```

否则：

```text
TournamentResult.status = failed
```

不能虚构 Judge Evidence。

---

# 9.39 Idempotency

Tournament Job 必须可重复执行。

Fingerprint：

```text
generation_id

candidate set

incumbent candidate

criteria fingerprint

judge version

prompt version

render profile

bracket seed
```

相同 Fingerprint：

```text
可直接返回已有 TournamentResult
```

避免重复花费 MLLM Cost。

---

# 9.40 输出接口：`TournamentResult`

```python
class TournamentResult(BaseModel):

    tournament_id: str

    generation_id: str

    status: Literal[
        "success",
        "single_candidate",
        "degraded",
        "failed",
    ]

    champion_candidate_id: str | None

    runner_up_candidate_id: str | None

    candidate_ids: list[str]

    incumbent_candidate_id: str | None

    bracket: list[TournamentRound]

    match_records: list[PairwiseMatchRecord]

    champion_evidence: ChampionSelectionEvidence | None

    selection_confidence: float | None

    resolution_stats: TournamentResolutionStats

    criteria_fingerprint: str

    judge_version: str

    runtime: TournamentRuntime
```

---

# 9.41 Match Record

```python
class PairwiseMatchRecord(BaseModel):

    match_id: str

    round_index: int

    candidate_x: str

    candidate_y: str

    forward_verdict: PairwiseVerdict | None

    swapped_verdict: PairwiseVerdict | None

    tiebreak_verdict: TieBreakVerdict | None

    winner_candidate_id: str

    resolution: Literal[
        "bidirectional_consensus",
        "meta_judge",
        "metric_fallback",
        "deterministic_fallback",
        "bye",
    ]
```

---

# 9.42 Champion Evidence

```python
class ChampionSelectionEvidence(BaseModel):

    champion_candidate_id: str

    wins: int

    criteria_wins: dict

    critical_penalties: dict

    probe_summary: dict

    unresolved_uncertainty: list[str]
```

注意：

```text
Champion Evidence
```

不等于：

```text
VerifierReport
```

它只解释：

> 为什么当前 Candidate 在 Tournament 中赢了。

---

# 9.43 Planner State 写回

Tournament 后：

```python
state.champion = result.champion_candidate_id

state.latest_tournament = {
    "tournament_id":
        result.tournament_id,

    "champion_candidate_id":
        result.champion_candidate_id,

    "selection_confidence":
        result.selection_confidence,

    "status":
        result.status,
}
```

然后：

```text
立即进入 Multi-Verifier
```

不返回 Planner。

这与：

```text
03 Motion Planner
```

的 Automatic Post-Generation Subgraph 保持一致。

---

# 9.44 Multi-Verifier 的输入

Tournament 最终只需要向下一模块提供：

```text
Champion Candidate Handle

Original Request

Motion Specification

Constraint VerificationSpec

Keyframe / Event Requirements

Tournament Summary（可选辅助信息）
```

Multi-Verifier 不应直接相信：

```text
Tournament says semantic winner
```

而跳过自己的绝对检查。

---

# 9.45 主执行伪代码

```python
async def run_tournament(
    request: TournamentRequest,
) -> TournamentResult:

    preflight = validate_tournament_request(
        request
    )

    if not preflight.ready:
        return failed_tournament(
            request,
            preflight,
        )

    candidates = load_successful_candidates(
        request.candidate_ids
    )

    if request.incumbent_candidate_id is not None:

        incumbent = try_load_comparable_incumbent(
            request.incumbent_candidate_id,
            request,
        )

        if incumbent is not None:
            candidates.append(
                incumbent
            )

    candidates = deduplicate_candidates(
        candidates
    )

    if len(candidates) == 1:
        return single_candidate_result(
            request,
            candidates[0],
        )

    packets = await prepare_candidate_packets(
        candidates=candidates,
        request=request,
    )

    probes = await probe_candidates_parallel(
        packets=packets,
        request=request,
    )

    bracket = build_deterministic_bracket(
        candidate_ids=[
            x.candidate_id
            for x in candidates
        ],
        seed=request.bracket_seed,
    )

    match_records = []

    current_round = bracket

    round_index = 0

    while len(current_round) > 1:

        pairs, byes = build_pairs(
            current_round
        )

        round_results = await compare_pairs_parallel(
            pairs=pairs,
            packets=packets,
            probes=probes,
            request=request,
            round_index=round_index,
        )

        winners = []

        for result in round_results:

            resolution = resolve_pair(
                result=result,
                packets=packets,
                request=request,
            )

            match_records.append(
                resolution.match_record
            )

            winners.append(
                resolution.winner_candidate_id
            )

        winners.extend(
            byes
        )

        current_round = winners

        round_index += 1

    champion = current_round[0]

    runner_up = infer_runner_up(
        match_records,
        champion,
    )

    return build_tournament_result(
        request=request,
        champion=champion,
        runner_up=runner_up,
        match_records=match_records,
        probes=probes,
    )
```

---

# 9.46 Pair Resolution 伪代码

```python
def resolve_pair(
    result,
    packets,
    request,
):

    forward_winner = normalize_winner_id(
        result.forward
    )

    swapped_winner = normalize_winner_id(
        result.swapped
    )

    if (
        forward_winner is not None
        and forward_winner == swapped_winner
    ):

        return MatchResolution(
            winner_candidate_id=
                forward_winner,

            resolution=
                "bidirectional_consensus",
        )

    tiebreak = run_meta_tiebreak(
        ...
    )

    if tiebreak.winner_candidate_id:
        return MatchResolution(
            winner_candidate_id=
                tiebreak.winner_candidate_id,

            resolution=
                "meta_judge",
        )

    metric_winner = resolve_with_metrics(
        ...
    )

    if metric_winner is not None:
        return MatchResolution(
            winner_candidate_id=
                metric_winner,

            resolution=
                "metric_fallback",
        )

    return MatchResolution(
        winner_candidate_id=
            deterministic_fallback(...),

        resolution=
            "deterministic_fallback",
    )
```

---

# 9.47 Candidate Preparation 伪代码

```python
async def prepare_candidate_packet(
    candidate_id,
    request,
):

    candidate = candidate_store.load(
        candidate_id
    )

    render_handle = await render_cache.get_or_create(
        candidate=candidate,
        profile=request.render_profile,
    )

    quick_metrics = await compute_quick_metrics(
        candidate=candidate,
        criteria=request.selection_policy.criteria,
    )

    return CandidateSelectionPacket(
        candidate_id=candidate_id,
        candidate_handle=candidate.candidate_handle,
        render_bundle_handle=render_handle,
        quick_metrics=quick_metrics,
        probe_id=None,
        metadata={
            "seed": candidate.metadata.seed,
            "scope": candidate.metadata.scope,
        },
    )
```

---

# 9.48 Pairwise Judge Service 伪代码

```python
async def compare_pair(
    candidate_x,
    candidate_y,
    probes,
    request,
):

    forward = await judge.compare(
        PairwiseJudgeRequest(
            candidate_a=candidate_x,
            candidate_b=candidate_y,
            probe_a=probes[
                candidate_x.candidate_id
            ],
            probe_b=probes[
                candidate_y.candidate_id
            ],
            presentation_order="AB",
            ...
        )
    )

    swapped = await judge.compare(
        PairwiseJudgeRequest(
            candidate_a=candidate_y,
            candidate_b=candidate_x,
            probe_a=probes[
                candidate_y.candidate_id
            ],
            probe_b=probes[
                candidate_x.candidate_id
            ],
            presentation_order="BA",
            ...
        )
    )

    return RawPairwiseComparison(
        forward=forward,
        swapped=swapped,
    )
```

---

# 9.49 与 Segment Regeneration 的连接

第 8 单元：

```text
scope="segment"
```

会尽量固定未失败区间。

Tournament 此时必须额外启用：

```text
continuity_preservation
```

并比较：

```text
目标 Segment 是否改善

非目标 Segment 是否被保留

Segment Boundary 是否连续
```

如果一个 Candidate：

```text
修好了 Segment 1
但破坏 Segment 0 / 2
```

不应轻易获胜。

---

# 9.50 与 Constraint 的连接

Constraint Compiler 已经生成：

```text
VerificationSpec
```

Tournament 只做：

```text
Quick Constraint Evidence
```

例如：

```text
contact distance

joint position error

trajectory RMSE
```

这些 Evidence 直接来自同一个：

```text
VerificationSpec
```

禁止 Tournament 自己重新解释：

```text
"right hand touches table"
```

避免：

```text
Generator / Tournament / Verifier
```

三套定义不一致。

---

# 9.51 与 Motion Compiler 的连接

Tournament 读取：

```text
MotionSpecification
```

而不是只读：

```text
最终 GEM Caption
```

因为 Motion Specification 中保留了：

```text
body part

event order

repetition

direction

style

segment timing
```

这些信息对于候选比较比单一 Caption 更可靠。

---

# 9.52 V1 Judge Backend

第一版推荐：

```text
Hybrid Judge
```

即：

```text
MLLM Visual Pairwise Judge
+
TMR
+
MotionCritic
+
Constraint Numeric Evidence
```

不是：

```text
纯 MLLM

也不是

纯 scalar metric
```

---

## 9.52.1 为什么不只用 MLLM

MLLM 可能：

```text
受 Camera / Render 影响
忽略细微 Joint Error
出现 Position Bias
评价不稳定
```

---

## 9.52.2 为什么不只用 Metric

单一 Metric 很难同时处理：

```text
复杂语义

事件顺序

风格

自然度

局部约束
```

所以使用 Hybrid Evidence。

---

# 9.53 V1 实现顺序

## Step 1：Tournament Request / Result Schema

先固定：

```text
TournamentRequest

CandidateSelectionPacket

CandidateProbe

PairwiseVerdict

TournamentResult
```

---

## Step 2：Candidate Store 连接

能够：

```text
GenerationResult.candidate_id
→ load SMPL / motion_repr
```

---

## Step 3：Lightweight Renderer

实现统一：

```text
global / local view
```

确保同一 Tournament Camera 一致。

---

## Step 4：Quick Metrics

先实现：

```text
TMR

MotionCritic

Constraint numeric metrics
```

---

## Step 5：Candidate Probe

实现：

```text
one candidate
→ structured criterion findings
```

---

## Step 6：Pairwise Judge

实现：

```text
A/B comparison
```

---

## Step 7：Swapped Comparison

实现：

```text
A/B
+
B/A
```

---

## Step 8：TieBreak

实现：

```text
Meta Judge
→ Metric Fallback
→ Stable Fallback
```

---

## Step 9：Binary Bracket

支持：

```text
K even

K odd

bye

incumbent
```

---

## Step 10：Cache / Service

实现：

```text
Probe Cache

Pairwise Cache

Tournament Idempotency

Parallel Round Execution
```

---

# 9.54 V1 最低实现范围

V1 必须完成：

```text
TournamentRequestBuilder

Tournament Preflight

Candidate Store Load

Canonical Tournament Rendering

Quick Metric Evidence

Candidate Probe

Pairwise Judge

Forward / Swapped Comparison

Inconsistency Resolution

Deterministic Bracket

Odd-K Bye

Incumbent Candidate

TournamentResult

State Champion Update

Multi-Verifier Handoff
```

V1 可以暂缓：

```text
Learned Pairwise Motion Ranker

Bradley-Terry Global Ranking

Multi-Bracket Ensemble

Round-Robin Tournament

Adaptive Tournament Budget

Fine-tuned Motion MLLM Judge
```

---

# 9.55 单元测试要求

至少覆盖：

```text
Single Candidate

Two Candidate Pair

K=4 Binary Tournament

Odd K

Partial Generation Result

Incumbent Candidate

Incumbent Incompatible

Forward / Swapped Consensus

Forward / Swapped Conflict

Meta-Judge TieBreak

Metric Fallback

Deterministic Final Fallback

Constraint-Critical Criterion

Segment Continuity Criterion

Probe Cache

Pairwise Cache

Render Camera Consistency

Judge Schema Error

Judge Timeout

Metric-only Degraded Mode

Tournament Idempotency
```

---

## Case 1：Single Candidate

```text
Generation:
1 successful candidate

Expected:
status = single_candidate

Champion = candidate

No Pairwise Judge call

Continue to Verifier
```

---

## Case 2：Position Bias

```text
Forward:
A wins

Swapped:
A wins

Expected:
resolution =
bidirectional_consensus
```

---

## Case 3：Inconsistent Judge

```text
Forward:
A wins

Swapped:
B wins

Expected:
do not random select

Run:
Meta TieBreak
→ Metrics
→ Stable Fallback
```

---

## Case 4：Constraint Critical

```text
A:
better style
contact error = 0.30m

B:
slightly less expressive
contact error = 0.04m

Constraint:
critical

Expected:
B strongly preferred
```

---

## Case 5：Segment Repair

```text
A:
target segment fixed
outside segments corrupted

B:
target segment fixed
outside segments preserved

Expected:
B wins continuity criterion
```

---

## Case 6：Incumbent

```text
Previous Champion:
candidate_old

New Candidates:
candidate_1
candidate_2
candidate_3

Expected:
candidate_old joins tournament

Only replace incumbent if
a new candidate wins
```

---

# 9.56 Tournament 评估指标

系统级：

```text
Tournament Latency

Probe Latency

Pairwise Judge Latency

Render Latency

Judge Token / Compute Cost

Cache Hit Rate

TieBreak Rate

Degraded Mode Rate
```

Judge 稳定性：

```text
Forward / Swapped Consistency Rate

Bracket Stability

Repeat-run Consistency

Judge / Human Agreement

Judge Confidence Calibration
```

Selection Quality：

```text
Champion Verifier Pass Rate

Champion vs Random Candidate Pass Rate

Champion vs Metric-Only Selection

Champion vs Absolute-Score Selection

Incumbent Replacement Success Rate
```

---

## 9.56.1 Bracket Stability

Offline Evaluation 中：

```text
同一 Candidate Set
```

用多个：

```text
bracket_seed
```

重复 Tournament。

观察：

```text
Champion 是否稳定
```

如果 Champion 对 bracket 非常敏感：

```text
Pairwise Judge 不稳定
或
Candidate 质量非常接近
```

需要进一步分析。

---

# 9.57 Ablation

建议至少比较：

```text
Random Candidate

Absolute MLLM Score

Metric-only Top-1

Pairwise Tournament

Pairwise + Swap

Pairwise + Swap + Probe

Hybrid Pairwise
```

验证：

> Pairwise Tournament 是否真的比简单打分更稳定。

---

# 9.58 参考工作与本模块对应关系

| 工作 | 借鉴内容 | 本模块位置 |
|---|---|---|
| VISTA | Binary Pairwise Tournament | Tournament 主结构 |
| VISTA | Probing Critique before comparison | Candidate Probe |
| VISTA | Forward / Swapped comparison | Position-bias control |
| VISTA | Criteria-based comparison | Selection Criteria |
| VISTA | Previous winner retained in iterative improvement | Incumbent Champion |
| AToM | Integrity / Temporal / Frequency | Event Alignment |
| MotionCritic | Human-perception motion quality | Naturalness Evidence |
| TMR | Text-Motion semantic matching | Semantic Evidence |
| Video-T1 | Test-time candidate selection / multi-verifier scaling | Selection scaling reference |

---

# 9.59 当前模块完成标准

Candidate Tournament 完成后，必须稳定实现：

```text
GenerationResult
       ↓
TournamentRequest
       ↓
Candidate Preparation
       ↓
Probe
       ↓
Deterministic Binary Bracket
       ↓
A/B + B/A Pairwise Judge
       ↓
Tie Resolution
       ↓
Champion
       ↓
TournamentResult
       ↓
State.champion
       ↓
Multi-Verifier
```

并保证：

```text
Tournament 不进入 Planner Action Space。

Tournament 不直接修改 Motion Plan。

Tournament 不重新生成 Motion。

Tournament 不执行 Repair。

Tournament 不决定 ACCEPT。

Tournament 只做相对 Candidate Selection。

Multi-Verifier 才做绝对 pass/fail。

Forward / Swapped 不一致时禁止随机决胜。

所有选择过程必须可复现、可缓存、可审计。

Repair 后允许保留 Incumbent Champion，
避免新一轮生成无条件覆盖旧结果。
```

# 9.60 Heading Continuity Completion Addendum

Candidate Tournament 还必须保证：

```text
有 HeadingContinuitySpec 时，full generation 也启用 continuity evidence；
无 continuity expectation 时，不自行创造 facing 要求；
explicit turn / spin window 不被错误惩罚；
heading metric 只做 relative evidence，不替代 Multi-Verifier 的 absolute gate。
```
