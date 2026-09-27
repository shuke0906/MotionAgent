"""Planner interface and deterministic Phase 2 implementation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from motion_agent.agent.actions import PlannerDecision, parse_planner_decision
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

