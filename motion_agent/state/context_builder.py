"""Build bounded Planner-facing context from canonical state."""

from __future__ import annotations

from motion_agent.state.schemas import (
    ConditionSummaryForPlanner,
    GenerationSummary,
    HistorySummary,
    MotionAgentState,
    MotionPlanSummary,
    PlannerContext,
    TaskSummary,
)


class PlannerContextBuilder:
    def __init__(self, max_recent_actions: int = 6, max_repair_proposals: int = 3, max_items_per_kind: int = 6) -> None:
        self.max_recent_actions = max_recent_actions
        self.max_repair_proposals = max_repair_proposals
        self.max_items_per_kind = max_items_per_kind

    def build(self, state: MotionAgentState) -> PlannerContext:
        plan = None
        if state.plan.status != "missing":
            plan = MotionPlanSummary(
                status=state.plan.status,
                segments=state.plan.segment_summaries[: self.max_items_per_kind],
                unresolved=[],
                plan_version=state.plan.plan_version,
            )

        generation = None
        if state.generation.latest_generation_id or state.generation.candidate_ids:
            generation = GenerationSummary(
                generation_id=state.generation.latest_generation_id,
                strategy=state.generation.strategy,
                scope=state.generation.scope,
                candidate_count=len(state.generation.candidate_ids),
                champion_candidate_id=state.generation.champion_candidate_id,
                generation_round=state.generation.generation_round,
            )

        diagnosis = state.evaluation.diagnosis_summary
        if diagnosis is not None:
            diagnosis = diagnosis.model_copy(
                update={"proposal_summaries": diagnosis.proposal_summaries[: self.max_repair_proposals]}
            )

        return PlannerContext(
            task=TaskSummary(
                original_request=state.task.original_request,
                target_duration_s=state.task.target_duration_s,
                fps=state.task.fps,
                total_frames=state.task.total_frames,
            ),
            plan=plan,
            conditions=ConditionSummaryForPlanner(
                text=state.conditions.text,
                retrievals=state.conditions.retrievals[: self.max_items_per_kind],
                constraints=state.conditions.constraints[: self.max_items_per_kind],
                keyframes=state.conditions.keyframes[: self.max_items_per_kind],
                condition_fingerprint=state.conditions.condition_fingerprint,
            ),
            latest_generation=generation,
            latest_verification=state.evaluation.verification_summary,
            diagnosis=diagnosis,
            history=HistorySummary(
                recent_actions=state.history.recent_actions[-self.max_recent_actions :],
                repeated_action_counts=dict(list(state.history.repeated_action_counts.items())[-self.max_recent_actions :]),
                last_failure_signatures=state.history.last_failure_signatures[-self.max_recent_actions :],
            ),
            budget=state.control.budgets,
            capabilities=state.control.capabilities,
        )

