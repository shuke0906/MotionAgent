"""Planner interface, deterministic test backend, and real LLM backend."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Protocol

from motion_agent.agent.actions import PlannerDecision, parse_planner_decision
from motion_agent.agent.llm_schema import LLMPlannerDecision
from motion_agent.agent.budgets import get_legal_actions
from motion_agent.llm import LLMClient, LLMConfig, PromptEnvelope
from motion_agent.state.schemas import MotionAgentState, PlannerContext


class Planner(Protocol):
    def step(self, context: PlannerContext, state: MotionAgentState) -> PlannerDecision:
        ...


class ScriptedPlanner:
    """Deterministic planner used by Phase 2 tests and validation."""

    def __init__(self, decisions: Sequence[PlannerDecision | dict]) -> None:
        self._decisions = [parse_planner_decision(decision) for decision in decisions]
        self._index = 0

    @property
    def decisions_made(self) -> int:
        return self._index

    def step(self, context: PlannerContext, state: MotionAgentState) -> PlannerDecision:
        if self._index >= len(self._decisions):
            raise RuntimeError("ScriptedPlanner has no remaining decisions")
        decision = self._decisions[self._index]
        self._index += 1
        return decision


FakePlanner = ScriptedPlanner


PLANNER_SYSTEM_PROMPT = """You are MotionPlanner, the top-level controller of MotionAgent.

Select exactly ONE next action from this fixed action space:
COMPILE_MOTION, RETRIEVE_REFERENCE, BUILD_CONSTRAINT, BUILD_KEYFRAME, GENERATE, ACCEPT, STOP_FAILED.

Do not invent tool calls, node names, schemas, or extra actions. GENERATE is followed automatically by generation,
K=1 selection, verification, diagnosis, and then a later planner step.

Use COMPILE_MOTION when no ready plan exists or semantic/timeline/frequency/caption repair is needed.
Use RETRIEVE_REFERENCE only for missing priors or rare/unfamiliar motion when retrieval budget remains.
Use BUILD_CONSTRAINT only for measurable spatial, trajectory, contact, or body-part requirements.
Use BUILD_KEYFRAME only for a required whole-body state at a time or boundary.
Use GENERATE only when the plan and text condition are ready.
Use ACCEPT only when complete verification reports overall_pass=true and no critical failure remains.
Use STOP_FAILED when budgets are exhausted, a terminal hint exists, or no legal recovery remains.

Current runtime uses K=1: GENERATE num_candidates must be 1. Do not invent reward targets.
Follow the supplied legal_actions and validated repair proposals. Do not repeat blocked repair families.
For a repair choose one supplied proposal's action, target_segments and parameters exactly.
If a required failure has no valid proposal, or a terminal hint exists, choose STOP_FAILED.
Set payload fields unrelated to the selected action to null. COMPILE_MOTION needs mode and focus;
GENERATE needs strategy, scope, num_candidates=1, reward_targets=[]; ACCEPT and STOP_FAILED need all payload fields null.
For structured constraint targets, use target_json as a JSON object string, otherwise target.
Return only a valid structured decision. Do not include chain-of-thought or prose."""


class RealLLMPlannerBackend:
    """Real Phase 11 planner backend using the shared LLM provider layer."""

    def __init__(self, client: LLMClient | None = None, *, prompt_version: str | None = None) -> None:
        self.client = client or LLMClient(LLMConfig.from_env())
        self.prompt_version = prompt_version or self.client.config.prompt_version
        self.last_metadata = None

    def step(self, context: PlannerContext, state: MotionAgentState) -> PlannerDecision:
        envelope = PromptEnvelope(
            prompt_version=self.prompt_version,
            system=PLANNER_SYSTEM_PROMPT,
            user=json.dumps(
                {
                    "planner_context": context.model_dump(mode="json"),
                    "legal_actions": [action.value for action in get_legal_actions(state)],
                    "blocked_repair_families": [item.model_dump(mode="json") for item in state.repair.blocked_repair_families],
                    "state_guard_notes": {
                        "accept_requires_complete_overall_pass": True,
                        "generate_requires_ready_plan_and_text": True,
                        "planner_must_choose_one_action": True,
                    },
                },
                sort_keys=True,
            ),
            metadata={"run_id": state.run.run_id, "state_version": state.state_version},
        )
        decision, metadata = self.client.complete_structured(envelope, LLMPlannerDecision)
        self.last_metadata = metadata
        return parse_planner_decision(decision.canonical())
