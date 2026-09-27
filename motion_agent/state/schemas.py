"""Canonical Phase 1 MotionAgent schemas.

This file is the single source of truth for the state substrate. Module-specific
schemas should import or extend these models instead of redefining them.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ArtifactHandle(StrictModel):
    artifact_id: str
    artifact_type: str
    uri: str
    fingerprint: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str


class RunState(StrictModel):
    run_id: str
    status: Literal["created", "running", "accepted", "failed", "cancelled"] = "created"
    created_at: str
    updated_at: str
    trace_id: str
    model_profile: str = "local_phase1"
    config_profile: str = "phase1"


class TaskState(StrictModel):
    original_request: str
    target_duration_s: float = 6.0
    fps: int = 30
    total_frames: int = 180
    user_constraints_summary: list[str] = Field(default_factory=list)
    scene_context_id: str | None = None
    task_version: int = 1


class GEMAdapterContext(StrictModel):
    fps: int = 30
    total_frames: int = 180
    width: int = 1280
    height: int = 720
    static_camera: bool = True
    coordinate_system: str = "gem_canonical_v1"
    model_version: str = "gem_smpl"
    checkpoint_version: str = "phase0_frozen"
    camera_handle: str | None = None


class HeadingContinuitySummary(StrictModel):
    mode: Literal["free", "inherit_previous", "explicit", "follow_trajectory"]
    anchor_segment_id: int | None = None
    preserve_facing: bool = False
    explicit_facing: str | None = None
    source: Literal["user_explicit", "continuity_default", "trajectory", "unknown"] = "unknown"


class MotionSegmentSummary(StrictModel):
    segment_id: int
    start_s: float
    end_s: float
    action: str
    body_parts: list[str] = Field(default_factory=list)
    style: list[str] = Field(default_factory=list)
    repetition: int | None = None
    heading_continuity: HeadingContinuitySummary | None = None
    status: str = "active"


class RoutingHintSummary(StrictModel):
    intent_type: Literal["retrieval_candidate", "constraint_candidate", "keyframe_candidate"]
    segment_id: int
    reason_code: str
    details: dict[str, Any] = Field(default_factory=dict)


class PlanState(StrictModel):
    motion_spec_id: str | None = None
    gem_text_condition_id: str | None = None
    status: Literal["missing", "draft", "ready", "needs_revision"] = "missing"
    segment_summaries: list[MotionSegmentSummary] = Field(default_factory=list)
    routing_hints: list[RoutingHintSummary] = Field(default_factory=list)
    plan_version: int = 0


class TextConditionSummary(StrictModel):
    ready: bool = False
    gem_text_condition_id: str | None = None
    segment_count: int = 0
    version: int = 0


class RetrievalSummary(StrictModel):
    ref_id: str
    segment_id: int
    retrieval_type: Literal["caption", "motion", "pose", "trajectory"]
    purpose: Literal["prompt_grounding", "motion_prior", "keyframe_source", "constraint_source"]
    score: float | None = None
    status: Literal["active", "superseded", "low_confidence", "invalid"] = "active"


class ConstraintSummary(StrictModel):
    constraint_id: str
    segment_id: int
    constraint_type: str
    requested_strength: str = "soft"
    effective_mode: str = "selection_reward"
    status: Literal["active", "superseded", "removed", "conflict"] = "active"
    verification_spec_id: str | None = None


class KeyframeSummary(StrictModel):
    keyframe_id: str
    segment_id: int
    target_time_s: float
    source_type: str
    control_root_translation: bool = True
    status: Literal["active", "superseded", "removed", "conflict"] = "active"
    verification_spec_id: str | None = None


class ConditionState(StrictModel):
    text: TextConditionSummary = Field(default_factory=TextConditionSummary)
    retrievals: list[RetrievalSummary] = Field(default_factory=list)
    constraints: list[ConstraintSummary] = Field(default_factory=list)
    keyframes: list[KeyframeSummary] = Field(default_factory=list)
    condition_fingerprint: str | None = None


class GenerationState(StrictModel):
    latest_generation_id: str | None = None
    strategy: Literal["normal", "guided"] | None = None
    scope: Literal["full", "segment"] | None = None
    target_segments: list[int] | None = None
    candidate_ids: list[str] = Field(default_factory=list)
    champion_candidate_id: str | None = None
    latest_tournament_id: str | None = None
    previous_champion_candidate_id: str | None = None
    generation_round: int = 0


class VerificationSummary(StrictModel):
    status: Literal["complete", "incomplete", "invalid", "failed_service"]
    overall_pass: bool
    passed_check_ids: list[str] = Field(default_factory=list)
    failed_check_ids: list[str] = Field(default_factory=list)
    warning_check_ids: list[str] = Field(default_factory=list)
    uncertain_check_ids: list[str] = Field(default_factory=list)
    critical_failures: list[str] = Field(default_factory=list)
    affected_segments: list[int] = Field(default_factory=list)
    diagnostic_codes: list[str] = Field(default_factory=list)


class RepairProposalSummary(StrictModel):
    proposal_id: str
    action: str
    reason_code: str
    target_segments: list[int] | None = None
    confidence: float


class DiagnosisSummary(StrictModel):
    status: str
    primary_failures: list[str] = Field(default_factory=list)
    root_causes: list[str] = Field(default_factory=list)
    target_segments: list[int] = Field(default_factory=list)
    proposal_summaries: list[RepairProposalSummary] = Field(default_factory=list)
    terminal_hint: str | None = None


class EvaluationState(StrictModel):
    latest_verification_id: str | None = None
    latest_diagnosis_id: str | None = None
    verification_summary: VerificationSummary | None = None
    diagnosis_summary: DiagnosisSummary | None = None


class RepairHistorySummary(StrictModel):
    failure_signature: str
    repair_family: str
    planner_action: str
    target_segments: list[int] = Field(default_factory=list)
    outcome: Literal["pending", "resolved", "improved", "unchanged", "worse", "failed_execution"] = "pending"


class BlockedRepairFamily(StrictModel):
    failure_signature: str
    repair_family: str
    reason: str


class RepairState(StrictModel):
    active_proposal_id: str | None = None
    recent_repairs: list[RepairHistorySummary] = Field(default_factory=list)
    blocked_repair_families: list[BlockedRepairFamily] = Field(default_factory=list)


class BudgetState(StrictModel):
    iterations_left: int = 8
    generations_left: int = 4
    retrieval_calls_left: int = 4
    guided_generations_left: int = 1
    llm_judge_calls_left: int | None = None
    max_total_candidates_left: int | None = None


class CapabilitySummary(StrictModel):
    retrieval_available: bool = True
    keyframe_available: bool = True
    hard_constraint_available: bool = True
    guided_generation_available: bool = False
    semantic_mllm_available: bool = False
    motioncritic_available: bool = False
    scene_geometry_available: bool = False


class ControlState(StrictModel):
    iteration: int = 0
    budgets: BudgetState = Field(default_factory=BudgetState)
    capabilities: CapabilitySummary = Field(default_factory=CapabilitySummary)
    accepted: bool = False
    terminal_reason: str | None = None


class ActionHistorySummary(StrictModel):
    step: int
    action: str
    target_segments: list[int] | None = None
    result_status: str
    artifact_ids: list[str] = Field(default_factory=list)


class HistoryState(StrictModel):
    recent_actions: list[ActionHistorySummary] = Field(default_factory=list)
    repeated_action_counts: dict[str, int] = Field(default_factory=dict)
    last_failure_signatures: list[str] = Field(default_factory=list)


class MotionAgentState(StrictModel):
    run: RunState
    task: TaskState
    gem_adapter_context: GEMAdapterContext
    plan: PlanState = Field(default_factory=PlanState)
    conditions: ConditionState = Field(default_factory=ConditionState)
    generation: GenerationState = Field(default_factory=GenerationState)
    evaluation: EvaluationState = Field(default_factory=EvaluationState)
    repair: RepairState = Field(default_factory=RepairState)
    control: ControlState = Field(default_factory=ControlState)
    history: HistoryState = Field(default_factory=HistoryState)
    state_version: int = 0
    schema_version: str = "phase1.v1"


class TaskSummary(StrictModel):
    original_request: str
    target_duration_s: float
    fps: int
    total_frames: int


class MotionPlanSummary(StrictModel):
    status: str
    segments: list[MotionSegmentSummary] = Field(default_factory=list)
    unresolved: list[str] = Field(default_factory=list)
    plan_version: int


class ConditionSummaryForPlanner(StrictModel):
    text: TextConditionSummary
    retrievals: list[RetrievalSummary] = Field(default_factory=list)
    constraints: list[ConstraintSummary] = Field(default_factory=list)
    keyframes: list[KeyframeSummary] = Field(default_factory=list)
    condition_fingerprint: str | None = None


class GenerationSummary(StrictModel):
    generation_id: str | None
    strategy: str | None
    scope: str | None
    candidate_count: int
    champion_candidate_id: str | None
    generation_round: int


class HistorySummary(StrictModel):
    recent_actions: list[ActionHistorySummary] = Field(default_factory=list)
    repeated_action_counts: dict[str, int] = Field(default_factory=dict)
    last_failure_signatures: list[str] = Field(default_factory=list)


class PlannerContext(StrictModel):
    task: TaskSummary
    plan: MotionPlanSummary | None
    conditions: ConditionSummaryForPlanner
    latest_generation: GenerationSummary | None
    latest_verification: VerificationSummary | None
    diagnosis: DiagnosisSummary | None
    history: HistorySummary
    budget: BudgetState
    capabilities: CapabilitySummary
