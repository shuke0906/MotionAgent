"""Strict visual observations. Expected requirements come only from the DSL."""

from typing import Literal

from pydantic import Field

from motion_agent.state.schemas import StrictModel


class VisualEvidence(StrictModel):
    timestamp_s: float = Field(ge=0)
    observation: str


class SemanticObservation(StrictModel):
    status: Literal["pass", "fail", "uncertain"]
    observed_actions: list[str]
    missing_actions: list[str]
    body_part_match: bool | None
    direction_match: bool | None
    confidence: float = Field(ge=0, le=1)
    evidence: list[VisualEvidence]


class IntegrityObservation(SemanticObservation):
    pass


class FrequencyObservation(StrictModel):
    status: Literal["pass", "fail", "uncertain"]
    event: str
    body_part: str | None
    expected_count: int = Field(gt=0)
    observed_count: int | None = Field(ge=0)
    confidence: float = Field(ge=0, le=1)
    evidence_timestamps: list[float]
    body_part_match: bool | None


class ObservedInterval(StrictModel):
    event_id: str
    start_s: float = Field(ge=0)
    end_s: float = Field(ge=0)


class TemporalObservation(StrictModel):
    status: Literal["pass", "fail", "uncertain"]
    expected_order: list[str]
    observed_order: list[str]
    relation_match: bool | None
    observed_intervals: list[ObservedInterval]
    confidence: float = Field(ge=0, le=1)
    evidence: list[VisualEvidence]


OBSERVATION_SCHEMAS = {"semantic": SemanticObservation, "event_integrity": IntegrityObservation,
                       "event_frequency": FrequencyObservation, "event_temporal": TemporalObservation}
