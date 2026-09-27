"""State reducer: the only legal domain-state mutation path for Phase 1."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable

from motion_agent.common.fingerprints import condition_fingerprint
from motion_agent.common.ids import new_id, new_run_id
from motion_agent.state.schemas import (
    ActionHistorySummary,
    ConditionState,
    ConstraintSummary,
    DiagnosisSummary,
    EvaluationState,
    GEMAdapterContext,
    GenerationState,
    HistoryState,
    KeyframeSummary,
    MotionAgentState,
    MotionSegmentSummary,
    PlanState,
    RetrievalSummary,
    RoutingHintSummary,
    RunState,
    TaskState,
    TextConditionSummary,
    VerificationSummary,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class StateReducer:
    def create_run(self, original_request: str, *, run_id: str | None = None, fps: int = 30, total_frames: int = 180) -> MotionAgentState:
        now = utc_now()
        run_id = run_id or new_run_id()
        return MotionAgentState(
            run=RunState(
                run_id=run_id,
                status="running",
                created_at=now,
                updated_at=now,
                trace_id=new_id("trace"),
            ),
            task=TaskState(
                original_request=original_request,
                fps=fps,
                total_frames=total_frames,
                target_duration_s=total_frames / fps,
            ),
            gem_adapter_context=GEMAdapterContext(fps=fps, total_frames=total_frames),
        )

    def _advance(
        self,
        state: MotionAgentState,
        *,
        action: str,
        result_status: str,
        artifact_ids: Iterable[str] = (),
        target_segments: list[int] | None = None,
        updates: dict,
    ) -> MotionAgentState:
        step = state.control.iteration + 1
        history = state.history.model_copy(deep=True)
        history.recent_actions.append(
            ActionHistorySummary(
                step=step,
                action=action,
                target_segments=target_segments,
                result_status=result_status,
                artifact_ids=list(artifact_ids),
            )
        )
        history.recent_actions = history.recent_actions[-10:]
        history.repeated_action_counts[action] = history.repeated_action_counts.get(action, 0) + 1
        base_control = updates.get("control", state.control)
        control = base_control.model_copy(deep=True)
        control.iteration = step
        control.budgets.iterations_left -= 1
        base_run = updates.get("run", state.run)
        run = base_run.model_copy(update={"updated_at": utc_now()})
        return state.model_copy(
            deep=True,
            update={
                **updates,
                "run": run,
                "control": control,
                "history": history,
                "state_version": state.state_version + 1,
            },
        )

    def _refresh_condition_fingerprint(self, conditions: ConditionState) -> ConditionState:
        active = {
            "retrievals": [r.model_dump(mode="json") for r in conditions.retrievals if r.status == "active"],
            "constraints": [c.model_dump(mode="json") for c in conditions.constraints if c.status == "active"],
            "keyframes": [k.model_dump(mode="json") for k in conditions.keyframes if k.status == "active"],
            "text": conditions.text.model_dump(mode="json"),
        }
        return conditions.model_copy(update={"condition_fingerprint": condition_fingerprint(active)})

    def commit_motion_plan(
        self,
        state: MotionAgentState,
        *,
        motion_spec_id: str,
        gem_text_condition_id: str,
        segments: list[MotionSegmentSummary],
        routing_hints: list[RoutingHintSummary] | None = None,
    ) -> MotionAgentState:
        plan = PlanState(
            motion_spec_id=motion_spec_id,
            gem_text_condition_id=gem_text_condition_id,
            status="ready",
            segment_summaries=segments,
            routing_hints=routing_hints or [],
            plan_version=state.plan.plan_version + 1,
        )
        conditions = state.conditions.model_copy(
            update={
                "text": TextConditionSummary(
                    ready=True,
                    gem_text_condition_id=gem_text_condition_id,
                    segment_count=len(segments),
                    version=state.conditions.text.version + 1,
                )
            }
        )
        conditions = self._refresh_condition_fingerprint(conditions)
        return self._advance(
            state,
            action="COMPILE_MOTION",
            result_status="success",
            artifact_ids=[motion_spec_id, gem_text_condition_id],
            updates={"plan": plan, "conditions": conditions},
        )

    def add_or_replace_constraint(self, state: MotionAgentState, summary: ConstraintSummary) -> MotionAgentState:
        constraints = [
            c.model_copy(update={"status": "superseded"})
            if c.constraint_id == summary.constraint_id or (c.segment_id == summary.segment_id and c.constraint_type == summary.constraint_type and c.status == "active")
            else c
            for c in state.conditions.constraints
        ]
        constraints.append(summary)
        conditions = self._refresh_condition_fingerprint(state.conditions.model_copy(update={"constraints": constraints}))
        artifact_ids = [summary.constraint_id] + ([summary.verification_spec_id] if summary.verification_spec_id else [])
        return self._advance(
            state,
            action="BUILD_CONSTRAINT",
            result_status="success",
            artifact_ids=artifact_ids,
            target_segments=[summary.segment_id],
            updates={"conditions": conditions},
        )

    def add_retrieval(self, state: MotionAgentState, summary: RetrievalSummary) -> MotionAgentState:
        retrievals = [
            r.model_copy(update={"status": "superseded"})
            if (
                r.status == "active"
                and r.segment_id == summary.segment_id
                and r.retrieval_type == summary.retrieval_type
                and r.purpose == summary.purpose
            )
            else r
            for r in state.conditions.retrievals
        ]
        retrievals.append(summary)
        conditions = self._refresh_condition_fingerprint(state.conditions.model_copy(update={"retrievals": retrievals}))
        control = state.control.model_copy(deep=True)
        control.budgets.retrieval_calls_left -= 1
        return self._advance(
            state.model_copy(update={"control": control}),
            action="RETRIEVE_REFERENCE",
            result_status=summary.status,
            artifact_ids=[summary.ref_id],
            target_segments=[summary.segment_id],
            updates={"conditions": conditions},
        )

    def remove_constraint(self, state: MotionAgentState, constraint_id: str) -> MotionAgentState:
        constraints = [
            c.model_copy(update={"status": "removed"}) if c.constraint_id == constraint_id else c
            for c in state.conditions.constraints
        ]
        conditions = self._refresh_condition_fingerprint(state.conditions.model_copy(update={"constraints": constraints}))
        return self._advance(
            state,
            action="BUILD_CONSTRAINT",
            result_status="removed",
            artifact_ids=[constraint_id],
            updates={"conditions": conditions},
        )

    def add_or_replace_keyframe(self, state: MotionAgentState, summary: KeyframeSummary) -> MotionAgentState:
        keyframes = [
            k.model_copy(update={"status": "superseded"})
            if k.keyframe_id == summary.keyframe_id or (k.segment_id == summary.segment_id and k.status == "active")
            else k
            for k in state.conditions.keyframes
        ]
        keyframes.append(summary)
        conditions = self._refresh_condition_fingerprint(state.conditions.model_copy(update={"keyframes": keyframes}))
        artifact_ids = [summary.keyframe_id] + ([summary.verification_spec_id] if summary.verification_spec_id else [])
        return self._advance(
            state,
            action="BUILD_KEYFRAME",
            result_status="success",
            artifact_ids=artifact_ids,
            target_segments=[summary.segment_id],
            updates={"conditions": conditions},
        )

    def remove_keyframe(self, state: MotionAgentState, keyframe_id: str) -> MotionAgentState:
        keyframes = [
            k.model_copy(update={"status": "removed"}) if k.keyframe_id == keyframe_id else k
            for k in state.conditions.keyframes
        ]
        conditions = self._refresh_condition_fingerprint(state.conditions.model_copy(update={"keyframes": keyframes}))
        return self._advance(
            state,
            action="BUILD_KEYFRAME",
            result_status="removed",
            artifact_ids=[keyframe_id],
            updates={"conditions": conditions},
        )

    def commit_generation_result(
        self,
        state: MotionAgentState,
        *,
        generation_id: str,
        candidate_ids: list[str],
        strategy: str = "normal",
        scope: str = "full",
        target_segments: list[int] | None = None,
    ) -> MotionAgentState:
        control = state.control.model_copy(deep=True)
        control.budgets.generations_left -= 1
        generation = GenerationState(
            latest_generation_id=generation_id,
            strategy=strategy,
            scope=scope,
            target_segments=target_segments,
            candidate_ids=candidate_ids,
            previous_champion_candidate_id=state.generation.champion_candidate_id,
            generation_round=state.generation.generation_round + 1,
        )
        advanced = self._advance(
            state.model_copy(update={"control": control}),
            action="GENERATE",
            result_status="success",
            artifact_ids=[generation_id, *candidate_ids],
            target_segments=target_segments,
            updates={"generation": generation, "evaluation": EvaluationState()},
        )
        return advanced

    def set_champion(self, state: MotionAgentState, candidate_id: str, *, tournament_id: str | None = None) -> MotionAgentState:
        generation = state.generation.model_copy(
            update={"champion_candidate_id": candidate_id, "latest_tournament_id": tournament_id}
        )
        artifact_ids = [candidate_id] + ([tournament_id] if tournament_id else [])
        return self._advance(
            state,
            action="TOURNAMENT",
            result_status="success",
            artifact_ids=artifact_ids,
            updates={"generation": generation},
        )

    def commit_verification_report(
        self,
        state: MotionAgentState,
        *,
        verification_id: str,
        summary: VerificationSummary,
    ) -> MotionAgentState:
        evaluation = state.evaluation.model_copy(
            update={"latest_verification_id": verification_id, "verification_summary": summary}
        )
        return self._advance(
            state,
            action="VERIFY",
            result_status="success" if summary.status == "complete" else summary.status,
            artifact_ids=[verification_id],
            target_segments=summary.affected_segments or None,
            updates={"evaluation": evaluation},
        )

    def commit_diagnosis_result(
        self,
        state: MotionAgentState,
        *,
        diagnosis_id: str,
        summary: DiagnosisSummary,
    ) -> MotionAgentState:
        evaluation = state.evaluation.model_copy(
            update={"latest_diagnosis_id": diagnosis_id, "diagnosis_summary": summary}
        )
        return self._advance(
            state,
            action="DIAGNOSE",
            result_status=summary.status,
            artifact_ids=[diagnosis_id],
            target_segments=summary.target_segments or None,
            updates={"evaluation": evaluation},
        )

    def accept(self, state: MotionAgentState) -> MotionAgentState:
        control = state.control.model_copy(update={"accepted": True, "terminal_reason": "accepted"})
        run = state.run.model_copy(update={"status": "accepted"})
        return self._advance(
            state,
            action="ACCEPT",
            result_status="success",
            updates={"control": control, "run": run},
        )

    def stop_failed(self, state: MotionAgentState, *, reason: str = "failed") -> MotionAgentState:
        control = state.control.model_copy(update={"accepted": False, "terminal_reason": reason})
        run = state.run.model_copy(update={"status": "failed"})
        return self._advance(
            state,
            action="STOP_FAILED",
            result_status=reason,
            updates={"control": control, "run": run},
        )
