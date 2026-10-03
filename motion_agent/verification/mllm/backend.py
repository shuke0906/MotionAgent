"""Normalize structured visual observations to the existing VerifierFinding."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from pydantic import ValidationError

from motion_agent.common.fingerprints import stable_fingerprint
from motion_agent.verification.backends import BackendUnavailable, MLLMVerifierBackend, finding
from motion_agent.verification.mllm.prompts import PROMPTS, PROMPT_VERSIONS, expected_requirements
from motion_agent.verification.mllm.provider import OpenAIVisualProvider, ProviderError, api_base_url
from motion_agent.verification.mllm.schemas import OBSERVATION_SCHEMAS
from motion_agent.verification.mllm.storyboard import StoryboardBuilder


def normalize_observation(request, check, observation, requirements):
    status, code = observation.status, None
    data = observation.model_dump(mode="json")
    if check.direction == "event_frequency":
        expected = requirements["expected_count"]
        if observation.expected_count != expected or observation.event != requirements["event"]:
            raise ValueError("response changed target event or expected count")
        observed = observation.observed_count
        if observation.body_part_match is False or (observation.body_part is not None and
                                                   observation.body_part not in requirements["body_parts"]):
            status, code = "fail", "BODY_PART_MISMATCH"
        elif status != "uncertain" and observed is not None and observed != expected:
            status, code = "fail", "FREQUENCY_MISMATCH"
        elif status == "pass" and (observed is None or observation.body_part_match is None):
            status = "uncertain"
        if observed is not None and len(observation.evidence_timestamps) != observed:
            raise ValueError("each observed cycle requires a timestamp")
        times = observation.evidence_timestamps
        if any(right <= left for left, right in zip(times, times[1:])):
            raise ValueError("cycle timestamps must be distinct and chronological")
    elif check.direction == "event_temporal":
        if observation.expected_order != requirements["expected_order"]:
            raise ValueError("response changed expected event IDs")
        intervals = {interval.event_id: interval for interval in observation.observed_intervals}
        if len(intervals) != len(observation.observed_intervals):
            raise ValueError("duplicate observed interval ID")
        for interval in intervals.values():
            if interval.start_s >= interval.end_s:
                raise ValueError("invalid event interval")
        for relation in requirements["relations"]:
            left, right = intervals.get(relation["left"]), intervals.get(relation["right"])
            if left is None or right is None:
                if status == "pass":
                    status = "uncertain"
                continue
            matches = (left.end_s <= right.start_s if relation["type"] == "before" else
                       max(left.start_s, right.start_s) < min(left.end_s, right.end_s))
            if relation["type"] == "alternating":
                # One interval per action cannot prove alternation.
                if status == "pass":
                    status = "uncertain"
            elif not matches:
                status, code = "fail", "TEMPORAL_RELATION_MISMATCH"
        if observation.relation_match is False:
            status, code = "fail", "TEMPORAL_RELATION_MISMATCH"
        elif observation.relation_match is None and status == "pass":
            status = "uncertain"
        times = [e.timestamp_s for e in observation.evidence] + [v for i in intervals.values() for v in (i.start_s, i.end_s)]
    else:
        if observation.missing_actions and status != "uncertain":
            status, code = "fail", "EVENT_MISSING" if check.direction == "event_integrity" else "SEMANTIC_MISMATCH"
        if observation.body_part_match is False:
            status, code = "fail", "BODY_PART_MISMATCH"
        if observation.direction_match is False:
            status, code = "fail", "DIRECTION_MISMATCH"
        expects_direction = any(event.get("direction") not in (None, "none") for event in requirements["events"])
        expects_body = any(any(part != "full_body" for part in event["body_parts"]) for event in requirements["events"])
        if status == "pass" and ((expects_direction and observation.direction_match is None) or
                                 (expects_body and observation.body_part_match is None)):
            status = "uncertain"
        required_actions = {event["action"] for event in requirements["events"]}
        if status == "pass" and not required_actions.issubset(observation.observed_actions):
            status = "uncertain"
        times = [e.timestamp_s for e in observation.evidence]
    if any(not 0 <= t <= request.motion_spec.duration_s for t in times):
        raise ValueError("evidence timestamp outside candidate duration")
    if status == "pass" and not times:
        status = "uncertain"
    if observation.confidence < 0.5 and status == "pass":
        status = "uncertain"
    if status == "fail" and code is None:
        code = "VISUAL_REQUIREMENT_MISMATCH"
    if status == "uncertain":
        code = "MLLM_UNCERTAIN"
    return status, code, data


class UnavailableMLLMBackend(MLLMVerifierBackend):
    def __init__(self, reason="MLLM_BACKEND_UNAVAILABLE"):
        self.reason = reason

    def verify(self, request, check):
        return finding(check, "mllm_unavailable", "uncertain", diagnostic_code="MLLM_BACKEND_UNAVAILABLE", message=self.reason)


class MockMLLMBackend(MLLMVerifierBackend):
    def __init__(self, status="uncertain"):
        self.status = status

    def verify(self, request, check):
        return finding(check, "fixture_mllm_v1", self.status, observed={"fixture": True})

    def cache_identity(self):
        return {"backend": "fixture_mllm_v1", "status": self.status}


class RealMLLMBackend(MLLMVerifierBackend):
    evaluator_version = "mllm_visual_adapter_v2"

    def __init__(self, config, *, provider=None, storyboard_builder=None):
        self.config = config
        self.provider = provider or OpenAIVisualProvider(config)
        self.storyboards = storyboard_builder or StoryboardBuilder(config.output_root + "/mllm/storyboards", config.storyboard_max_frames,
                                                                  config.mllm_contact_sheet)
        self.api_calls = 0
        self.latency_ms = 0.0
        self.executions = []
        self.raw_response_count = 0
        self.valid_observation_count = 0

    def cache_identity(self):
        return {"code": self.evaluator_version, "provider": self.config.provider, "model": self.config.mllm_model,
                "prompts": PROMPT_VERSIONS, "schema": "visual_observations_v1",
                "temperature": None if self.config.mllm_model.startswith(("gpt-5", "gpt-6", "o1", "o3", "o4")) else 0,
                "endpoint_identity": stable_fingerprint(api_base_url()),
                "image_detail": self.config.mllm_image_detail,
                "contact_sheet": self.config.mllm_contact_sheet,
                "contact_sheet_version": "2x2_max512_v2",
                "storyboard_max_frames": self.config.storyboard_max_frames, "sampling": "scenario8hz_dense16hz_v1"}

    def verify(self, request, check):
        started = time.perf_counter()
        refs = []
        requirements = expected_requirements(request, check)
        observation_schema = OBSERVATION_SCHEMAS[check.direction]
        try:
            if hasattr(self.provider, "check_ready"):
                self.provider.check_ready()
            for attempt in range(2):
                storyboard = self.storyboards.build(request, check, dense=attempt == 1)
                if self.api_calls >= self.config.mllm_max_calls:
                    raise BackendUnavailable("MLLM_API_CALL_BUDGET_EXHAUSTED")
                self.api_calls += 1
                raw = self.provider.observe(PROMPTS[check.direction], json.dumps(requirements), storyboard, observation_schema)
                self.raw_response_count += 1
                identity = stable_fingerprint({"candidate": request.candidate_id, "check": check.check_id,
                                               "requirements": requirements, "storyboard": storyboard,
                                               "backend": self.cache_identity(), "attempt": attempt,
                                               "response_id": raw.get("response_id")})
                folder = Path(self.config.output_root) / "mllm/structured_responses" / identity
                folder.mkdir(parents=True, exist_ok=True)
                raw_path = folder / "response.json"
                raw_path.write_text(json.dumps(raw, indent=2), encoding="utf-8")
                manifest_path = folder / "evidence_manifest.json"
                manifest_path.write_text(json.dumps(storyboard, indent=2), encoding="utf-8")
                refs.extend([str(raw_path.resolve()), str(manifest_path.resolve())])
                try:
                    observation = observation_schema.model_validate_json(raw["output_text"], strict=True)
                    status, code, data = normalize_observation(request, check, observation, requirements)
                    self.valid_observation_count += 1
                except (ValidationError, ValueError, KeyError):
                    if attempt == 0:
                        continue
                    raise ProviderError("MLLM_INVALID_RESPONSE_SCHEMA") from None
                if status == "uncertain" and attempt == 0:
                    continue
                result = finding(check, self.evaluator_version, status, diagnostic_code=code, expected=requirements,
                                 observed={**data, "source": "real_mllm", "provider": self.config.provider,
                                           "model": raw.get("model", self.config.mllm_model), "attempts": attempt + 1,
                                           "prompt_version": PROMPT_VERSIONS[check.direction]}, evidence_refs=refs)
                (folder / "finding.json").write_text(result.model_dump_json(indent=2), encoding="utf-8")
                self.executions.append(result.model_dump(mode="json"))
                return result
        except (BackendUnavailable, FileNotFoundError, ModuleNotFoundError) as exc:
            return finding(check, self.evaluator_version, "uncertain", diagnostic_code="MLLM_BACKEND_UNAVAILABLE",
                           message=str(exc), evidence_refs=refs)
        except ProviderError as exc:
            return finding(check, self.evaluator_version, "error", diagnostic_code=exc.code, evidence_refs=refs)
        except Exception as exc:
            return finding(check, self.evaluator_version, "error", diagnostic_code="MLLM_EVIDENCE_ERROR",
                           message=type(exc).__name__, evidence_refs=refs)
        finally:
            self.latency_ms += (time.perf_counter() - started) * 1000
