from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from motion_agent.compiler.motion_compiler import CompilerRequest, MotionCompiler
from motion_agent.constraints.schemas import VerificationSpec
from motion_agent.verification.artifacts import candidate_identity
from motion_agent.verification.backends import BackendUnavailable, FixtureNaturalnessBackend
from motion_agent.verification.cache import verifier_cache_key
from motion_agent.verification.calibration import provisional_calibration
from motion_agent.verification.config import LearnedVerifierConfig, create_verification_service
from motion_agent.verification.mllm.backend import MockMLLMBackend, RealMLLMBackend, normalize_observation
from motion_agent.verification.mllm.prompts import expected_requirements
from motion_agent.verification.mllm.provider import OpenAIVisualProvider, ProviderError, api_base_url
from motion_agent.verification.mllm.schemas import FrequencyObservation, SemanticObservation, TemporalObservation
from motion_agent.verification.naturalness.motioncritic import MotionCriticBackend
from motion_agent.verification.plan_builder import VerificationPlanBuilder
from motion_agent.verification.representations import smpl_to_motioncritic, resample_critic, smplx_joints
from motion_agent.verification.schemas import VerificationRequest, VerifierCheck, VerifierFinding
from motion_agent.verification.semantic.real_tmr import TMRBackend
from motion_agent.verification.semantic.text_policy import semantic_text
from motion_agent.verification.semantic.tmr import FixtureSemanticBackend
from motion_agent.verification.service import VerificationService


PROMPTS = ["A person walks forward.", "A person waves their right hand three times.",
           "A person walks forward while waving their right hand three times.",
           "A person walks forward, then turns left, then sits down."]


@pytest.mark.parametrize("endpoint", ["", "/", "/embeddings", "/responses", "/chat/completions"])
def test_provider_normalizes_api_endpoint_without_changing_host(monkeypatch, endpoint):
    monkeypatch.delenv("MLLM_BASE_URL", raising=False)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1" + endpoint)
    assert api_base_url() == "https://api.openai.com/v1"


def test_provider_custom_root_and_override(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1/embeddings")
    monkeypatch.setenv("MLLM_BASE_URL", "https://example.org/compatible/v1/responses")
    assert api_base_url() == "https://example.org/compatible/v1"


def test_provider_rejects_url_embedded_credentials(monkeypatch):
    monkeypatch.setenv("MLLM_BASE_URL", "https://user:secret@example.org/v1")
    with pytest.raises(BackendUnavailable, match="INVALID_API_BASE_URL"):
        api_base_url()


@pytest.mark.parametrize("budget,quota,oversized", [(2, False, False), (1, False, False), (2, True, False), (2, False, True)])
def test_provider_rate_retry_respects_network_budget(monkeypatch, tmp_path, budget, quota, oversized):
    from types import SimpleNamespace
    import openai
    monkeypatch.setenv("MLLM_API_KEY", "fixture-not-a-real-key")
    monkeypatch.delenv("MLLM_BASE_URL", raising=False)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1/embeddings")
    calls, waits, settings = [], [], []

    class RateLimitError(Exception):
        code = "insufficient_quota" if quota else "rate_limit_exceeded"
        body = {"message": "Limit: 30000, Requested: 43605"} if oversized else None

    def create(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise RateLimitError("sensitive provider text must not be propagated")
        return SimpleNamespace(status="completed", output_text="{}", id="fixture", model="fixture", usage=None)

    def client(**kwargs):
        settings.append(kwargs)
        return SimpleNamespace(responses=SimpleNamespace(create=create))

    monkeypatch.setattr(openai, "OpenAI", client)
    monkeypatch.setattr("motion_agent.verification.mllm.provider.time.sleep", waits.append)
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"fixture")
    provider = OpenAIVisualProvider(LearnedVerifierConfig(mllm_max_calls=budget, mllm_rate_limit_retry_s=1))
    board = {"frames": [{"path": str(frame), "frame_index": 0, "timestamp_s": 0.0}]}
    if oversized:
        with pytest.raises(ProviderError, match="REQUEST_EXCEEDS_TOKEN_LIMIT"):
            provider.observe("fixture", "fixture", board, FrequencyObservation)
        assert waits == [] and len(calls) == 1
        assert provider.transport_errors[0]["requested_tokens"] == 43605
    elif quota:
        with pytest.raises(BackendUnavailable, match="INSUFFICIENT_QUOTA"):
            provider.observe("fixture", "fixture", board, FrequencyObservation)
        assert waits == [] and len(calls) == 1
        with pytest.raises(BackendUnavailable):
            provider.check_ready()
    elif budget == 1:
        with pytest.raises(ProviderError, match="RATE_LIMIT_EXCEEDED"):
            provider.observe("fixture", "fixture", board, FrequencyObservation)
        assert waits == [] and len(calls) == 1
    else:
        assert provider.observe("fixture", "fixture", board, FrequencyObservation)["response_id"] == "fixture"
        assert len(calls) == 2 and waits == [1]
        assert calls[-1]["input"][0]["content"][-1]["detail"] == "low"
    assert provider.network_calls == len(calls)
    assert settings[0]["base_url"] == "https://api.openai.com/v1"


def request_for(prompt=PROMPTS[0]):
    spec = MotionCompiler().compile(CompilerRequest(original_request=prompt)).motion_spec
    return VerificationRequest(verification_id="verify_test", candidate_id="candidate", original_request=prompt,
        motion_spec=spec, candidate={"motion_repr": torch.zeros(180, 151), "body_params_global": {
            "body_pose": torch.zeros(180, 63), "global_orient": torch.zeros(180, 3),
            "transl": torch.zeros(180, 3), "betas": torch.zeros(180, 10)}})


def test_fk_chunks_tail_with_fixed_model_batch(monkeypatch, tmp_path):
    import sys
    from types import SimpleNamespace
    batches = []

    class Model:
        def __init__(self, *args, batch_size, **kwargs):
            self.batch_size = batch_size

        def __call__(self, **kwargs):
            batches.append(len(kwargs["body_pose"]))
            assert all(value.shape[0] == self.batch_size for value in kwargs.values() if isinstance(value, torch.Tensor))
            return SimpleNamespace(joints=torch.ones(self.batch_size, 55, 3))

    monkeypatch.setitem(sys.modules, "smplx", SimpleNamespace(SMPLX=Model))
    asset = tmp_path / "SMPLX_NEUTRAL.npz"
    asset.touch()
    params = {key: value[:36] for key, value in request_for().candidate["body_params_global"].items()}
    assert smplx_joints(params, str(asset)).shape == (36, 22, 3)
    assert batches == [32, 32]


def test_fk_evidence_reused_across_backends(monkeypatch, tmp_path):
    from motion_agent.verification.evidence import SMPLXEvidencePreparer
    asset = tmp_path / "model.npz"
    asset.touch()
    calls = []
    monkeypatch.setattr("motion_agent.verification.evidence.smplx_joints",
                        lambda params, path: calls.append(path) or torch.zeros(180, 22, 3))
    prepare = SMPLXEvidencePreparer(str(asset))
    request = request_for()
    first, second = prepare(request), prepare(request.model_copy())
    assert len(calls) == 1
    assert torch.equal(first.candidate["joints_yup"], second.candidate["joints_yup"])
    assert "model_sha256" in first.candidate["fk_evidence"]


def learned_builder():
    return VerificationPlanBuilder(learned_config=LearnedVerifierConfig())


@pytest.mark.parametrize("prompt,index", [(p, i) for i, p in enumerate(PROMPTS)])
def test_scenario_routing(prompt, index):
    plan = learned_builder().build(request_for(prompt))
    directions = {c.direction for c in plan.checks}
    assert {"technical", "semantic", "naturalness", "physical"} <= directions
    if index == 0:
        assert "event_frequency" not in directions
        assert all(c.evaluator != "mllm_backend" for c in plan.checks)
    if index in (1, 2):
        frequency = next(c for c in plan.checks if c.direction == "event_frequency")
        assert frequency.metadata == {"action": "wave", "expected_count": 3}
        assert frequency.required and frequency.evaluator == "mllm_backend"
        assert "event_integrity" in directions
    if index in (2, 3):
        assert "event_temporal" in directions


def test_count_trace_and_semantic_text_policy():
    request = request_for(PROMPTS[2])
    wave = next(s for s in request.motion_spec.segments if s.temporal_constraint and s.temporal_constraint.count)
    assert wave.temporal_constraint.count == 3
    check = next(c for c in learned_builder().build(request).checks if c.direction == "event_frequency")
    assert check.metadata["expected_count"] == wave.temporal_constraint.count
    text = semantic_text(request.motion_spec)
    assert "right hand" in text and "forward" in text and "wave" in text
    assert "3" not in text and "three" not in text and "then" not in text


@pytest.mark.parametrize("count,status", [(3, "pass"), (2, "fail"), (0, "fail")])
def test_frequency_observed_evidence_controls_result(count, status):
    request = request_for(PROMPTS[1])
    check = next(c for c in learned_builder().build(request).checks if c.direction == "event_frequency")
    response = FrequencyObservation(status="pass", event="wave", body_part="right_hand", expected_count=3,
        observed_count=count, confidence=0.9, evidence_timestamps=list(range(1, count + 1)), body_part_match=True)
    actual, code, data = normalize_observation(request, check, response, expected_requirements(request, check))
    assert actual == status
    if count != 3:
        assert code == "FREQUENCY_MISMATCH"
    assert data["observed_count"] == count


def test_wrong_hand_fails_even_if_count_matches():
    request = request_for(PROMPTS[1])
    check = next(c for c in learned_builder().build(request).checks if c.direction == "event_frequency")
    obs = FrequencyObservation(status="pass", event="wave", body_part="left_hand", expected_count=3,
        observed_count=3, confidence=1, evidence_timestamps=[1, 2, 3], body_part_match=False)
    assert normalize_observation(request, check, obs, expected_requirements(request, check))[:2] == ("fail", "BODY_PART_MISMATCH")


@pytest.mark.parametrize("timestamps", [[1, 1, 2], [3, 2, 1]])
def test_frequency_rejects_duplicate_or_reversed_cycle_evidence(timestamps):
    request = request_for(PROMPTS[1])
    check = next(c for c in learned_builder().build(request).checks if c.direction == "event_frequency")
    obs = FrequencyObservation(status="pass", event="wave", body_part="right_hand", expected_count=3,
        observed_count=3, confidence=1, evidence_timestamps=timestamps, body_part_match=True)
    with pytest.raises(ValueError, match="distinct and chronological"):
        normalize_observation(request, check, obs, expected_requirements(request, check))


@pytest.mark.parametrize("prompt", [PROMPTS[2], PROMPTS[3]])
def test_sequential_overlap_and_reversed_events_fail(prompt):
    request = request_for(prompt)
    check = next(c for c in learned_builder().build(request).checks if c.direction == "event_temporal")
    requirements = expected_requirements(request, check)
    events = requirements["expected_order"]
    intervals = [{"event_id": event, "start_s": float(i + 1), "end_s": float(i + 1.8)}
                 for i, event in enumerate(events if prompt == PROMPTS[2] else reversed(events))]
    obs = TemporalObservation(status="pass", expected_order=events, observed_order=[i["event_id"] for i in intervals],
        relation_match=True, observed_intervals=intervals, confidence=1,
        evidence=[{"timestamp_s": 1.0, "observation": "fixture event"}])
    assert normalize_observation(request, check, obs, requirements)[0] == "fail"


@pytest.mark.parametrize("missing,body,direction,code", [(["wave"], True, True, "EVENT_MISSING"),
                                                        ([], False, True, "BODY_PART_MISMATCH"),
                                                        ([], True, False, "DIRECTION_MISMATCH")])
def test_integrity_semantic_observations(missing, body, direction, code):
    request = request_for(PROMPTS[3])
    check = next(c for c in learned_builder().build(request).checks if c.direction == "event_integrity")
    obs = SemanticObservation(status="pass", observed_actions=["walk", "turn", "sit_down"], missing_actions=missing,
        body_part_match=body, direction_match=direction, confidence=1,
        evidence=[{"timestamp_s": 1.0, "observation": "controlled fixture"}])
    assert normalize_observation(request, check, obs, expected_requirements(request, check))[:2] == ("fail", code)


def test_motioncritic_conversion_and_strict_hand_boundary():
    params = request_for().candidate["body_params_global"]
    with pytest.raises(BackendUnavailable):
        smpl_to_motioncritic(params)
    motion, conversion = smpl_to_motioncritic(params, "neutral_terminal_hands")
    assert motion.shape == (1, 180, 25, 3)
    assert conversion["approximation"]
    motion[0, :, 24, 0] = torch.arange(180) / 30
    converted = resample_critic(motion, 30)
    assert converted.shape[1] == 120
    assert converted[0, 1, 24, 0].item() == pytest.approx(0.05)


def test_real_and_fixture_backends_return_identical_contract(tmp_path):
    config = LearnedVerifierConfig(tmr_run_dir=str(tmp_path), motioncritic_checkpoint=str(tmp_path / "missing.pth"))
    request = request_for()
    plan = learned_builder().build(request)
    semantic = next(c for c in plan.checks if c.check_id == "semantic")
    natural = next(c for c in plan.checks if c.check_id == "motioncritic")
    for backend, check in [(FixtureSemanticBackend(), semantic), (TMRBackend(config), semantic),
                           (FixtureNaturalnessBackend(), natural), (MotionCriticBackend(config), natural),
                           (MockMLLMBackend(), semantic)]:
        result = backend.verify(request, check)
        assert isinstance(result, VerifierFinding)
        assert result.check_id == check.check_id and result.required == check.required


class FakeStoryboards:
    def __init__(self):
        self.dense = []

    def build(self, request, check, *, dense):
        self.dense.append(dense)
        return {"frames": [{"frame_index": 0, "timestamp_s": 1.0, "path": "fixture.jpg"}], "fixture": True}


class FixtureProvider:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = 0

    def observe(self, *args):
        self.calls += 1
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return {"output_text": json.dumps(response), "model": "fixture", "response_id": f"fixture_{self.calls}"}


def semantic_response(status="pass"):
    return {"status": status, "observed_actions": ["walk"], "missing_actions": [],
            "body_part_match": True, "direction_match": True, "confidence": 0.8,
            "evidence": [{"timestamp_s": 1.0, "observation": "fixture only"}]}


@pytest.mark.parametrize("responses,expected", [([semantic_response("uncertain"), semantic_response("uncertain")], "uncertain"),
    ([{"prose": "looks fine"}, semantic_response()], "pass"), ([{"prose": "bad"}, {"prose": "bad"}], "error"),
    ([ProviderError("MLLM_TIMEOUT")], "error"), ([BackendUnavailable("NO_CREDENTIALS")], "uncertain")])
def test_real_adapter_retry_schema_timeout_uncertainty(tmp_path, responses, expected):
    provider, storyboard = FixtureProvider(responses), FakeStoryboards()
    backend = RealMLLMBackend(LearnedVerifierConfig(output_root=str(tmp_path)), provider=provider, storyboard_builder=storyboard)
    check = VerifierCheck(check_id="visual", direction="semantic", evaluator="mllm_backend", required=True)
    result = backend.verify(request_for(), check)
    assert result.status == expected
    if len(responses) == 2:
        assert storyboard.dense == [False, True]
    assert backend.api_calls == len(responses)
    assert backend.latency_ms > 0


def test_tmr_and_motioncritic_disagreement_never_averaged():
    request = request_for()
    request.physical_evidence = {"foot_skating": 0, "ground_penetration": 0, "smoothness": 0}
    service = VerificationService(plan_builder=learned_builder(), semantic_backend=FixtureSemanticBackend("pass"),
                                  naturalness_backend=FixtureNaturalnessBackend("fail"))
    report = service.verify(request)
    assert next(f for f in report.findings if f.check_id == "semantic").status == "pass"
    assert next(f for f in report.findings if f.check_id == "motioncritic").status == "fail"
    assert not report.overall_pass and report.status == "complete"


def test_numeric_failure_cannot_be_overridden_by_visual_pass():
    request = request_for(PROMPTS[1])
    request.physical_evidence = {"foot_skating": 0, "ground_penetration": 0, "smoothness": 0}
    request.constraint_verification_specs = [VerificationSpec(verification_spec_id="v", constraint_id="c",
        metric_type="joint_position_error", target_segments=[0], pass_threshold=0.05, units="m")]
    request.observed_measurements = {"v": 0.08}
    service = VerificationService(plan_builder=learned_builder(), semantic_backend=FixtureSemanticBackend(),
                                  naturalness_backend=FixtureNaturalnessBackend(), mllm_backend=MockMLLMBackend("pass"))
    report = service.verify(request)
    assert not report.overall_pass and "constraint_c" in report.failed_required_checks


def test_cache_covers_actual_motion_spec_and_backend_identity():
    request = request_for(PROMPTS[1])
    check = next(c for c in learned_builder().build(request).checks if c.direction == "event_frequency")
    first = verifier_cache_key(request, check, backend_identity="model1")
    assert first != verifier_cache_key(request, check, backend_identity="model2")
    request.motion_spec.segments[0].temporal_constraint.count = 2
    assert first != verifier_cache_key(request, check, backend_identity="model1")
    identity = candidate_identity(request)
    request.candidate["motion_repr"][0, 0] = 1
    assert identity != candidate_identity(request)


def test_physical_missing_is_incomplete_in_real_profile():
    service = VerificationService(plan_builder=learned_builder(), semantic_backend=FixtureSemanticBackend(),
                                  naturalness_backend=FixtureNaturalnessBackend())
    report = service.verify(request_for())
    assert report.status == "incomplete" and not report.overall_pass
    assert any(f.diagnostic_code == "PHYSICAL_EVIDENCE_MISSING" for f in report.findings)


def test_calibration_no_threshold_when_distributions_overlap():
    assert provisional_calibration([0.8], [0.3])["provisional_threshold"] == pytest.approx(0.55)
    assert provisional_calibration([0.3], [0.8])["provisional_threshold"] is None
    assert provisional_calibration([], [])["production_calibrated"] is False


def test_storyboard_real_video_sampling_and_binding(tmp_path):
    cv2 = pytest.importorskip("cv2")
    import numpy as np
    from motion_agent.verification.mllm.storyboard import StoryboardBuilder
    request = request_for(PROMPTS[1])
    video_path = tmp_path / "fixture.avi"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"MJPG"), 30, (64, 64))
    for i in range(180):
        writer.write(np.full((64, 64, 3), i % 255, dtype=np.uint8))
    writer.release()
    request.visual_evidence_uri, request.visual_evidence_candidate_id = str(video_path), request.candidate_id
    check = next(c for c in learned_builder().build(request).checks if c.direction == "event_frequency")
    builder = StoryboardBuilder(str(tmp_path))
    base, dense = builder.build(request, check), builder.build(request, check, dense=True)
    assert len(base["frames"]) >= 48 and len(dense["frames"]) > len(base["frames"])
    assert base["frames"][0]["timestamp_s"] == 0
    assert base["frames"][-1]["frame_index"] == 179
    sheets = StoryboardBuilder(str(tmp_path), contact_sheet=True).build(request, check)
    assert len(sheets["images"]) == (len(base["frames"]) + 3) // 4
    assert [tile["frame_index"] for page in sheets["images"] for tile in page["tiles"]] == [f["frame_index"] for f in base["frames"]]
    sheet = cv2.imread(sheets["images"][0]["path"])
    original = cv2.imread(base["frames"][0]["path"])
    assert np.array_equal(sheet[24:88, :64], original)
    assert sheets["images"][-1]["blank_tiles"] == (-len(base["frames"])) % 4
    request.visual_evidence_candidate_id = "other"
    with pytest.raises(BackendUnavailable):
        builder.build(request, check)


def test_contact_sheet_upload_size_preserves_all_source_frames(tmp_path):
    cv2 = pytest.importorskip("cv2")
    import numpy as np
    from motion_agent.verification.mllm.storyboard import StoryboardBuilder
    frames = []
    for index in range(4):
        path = tmp_path / f"source_{index}.png"
        cv2.imwrite(str(path), np.full((384, 512, 3), 30 + 30 * index, dtype=np.uint8))
        frames.append({"path": str(path), "frame_index": index * 4, "timestamp_s": index / 8})
    sheets = StoryboardBuilder(str(tmp_path), contact_sheet=True)._sheets(frames, tmp_path)
    assert len(sheets) == 1 and sheets[0]["width"] == 512 and sheets[0]["height"] == 408
    assert [tile["frame_index"] for tile in sheets[0]["tiles"]] == [0, 4, 8, 12]
    assert "sheets_v2_512" in sheets[0]["path"]
    assert cv2.imread(sheets[0]["path"]).std() > 0


def test_cache_survives_tensor_reload_and_mllm_provider_not_called_again(tmp_path):
    request = request_for()
    request.candidate["motion_repr"] = request.candidate["motion_repr"].clone()
    first_identity = candidate_identity(request)
    request.candidate["motion_repr"] = request.candidate["motion_repr"].clone()
    assert first_identity == candidate_identity(request)
    provider = FixtureProvider([semantic_response()])
    backend = RealMLLMBackend(LearnedVerifierConfig(output_root=str(tmp_path)), provider=provider,
                              storyboard_builder=FakeStoryboards())
    service = VerificationService(mllm_backend=backend)
    check = VerifierCheck(check_id="visual", direction="semantic", evaluator="mllm_backend", required=True)
    first = service._run_cached(request, check)
    assert first.status == "pass"
    request.candidate["motion_repr"] = request.candidate["motion_repr"].clone()
    assert service._run_cached(request, check).status == "pass"
    assert provider.calls == 1 and service.cache.hits == 1


def test_deterministic_direction_preferred_and_cannot_be_visually_overridden():
    request = request_for(PROMPTS[3])
    turn = next(s for s in request.motion_spec.segments if s.action == "turn")
    request.observed_measurements = {f"signed_heading_change_{turn.segment_id}": -90}
    plan = learned_builder().build(request)
    check = next(c for c in plan.checks if c.evaluator == "heading_geometry_v1")
    assert VerificationService()._run_check(request, check).status == "fail"
    visual = next(c for c in plan.checks if c.check_id == "semantic_compositional")
    expected = expected_requirements(request, visual)
    assert next(e for e in expected["events"] if e["action"] == "turn")["direction"] is None


def test_numeric_only_and_keyframe_routes_without_mllm_requirement():
    request = request_for()
    request.motion_spec.segments[0].action = "hold"
    request.constraint_verification_specs = [VerificationSpec(verification_spec_id="v", constraint_id="c",
        metric_type="joint_position_error", target_segments=[0], pass_threshold=0.05, units="m")]
    plan = learned_builder().build(request)
    assert any(c.direction == "constraint" and c.required for c in plan.checks)
    assert all(c.evaluator != "mllm_backend" for c in plan.checks)
    from motion_agent.keyframes.schemas import KeyframeVerificationSpec
    request.keyframe_verification_specs = [KeyframeVerificationSpec(keyframe_id="k", target_frame=179,
        temporal_tolerance_frames=3, pose_handle="pose", metrics=["joint_rotation_error"])]
    assert any(c.direction == "keyframe" and c.required for c in learned_builder().build(request).checks)


def test_real_adapter_success_paths_with_explicit_stub_inference(tmp_path, monkeypatch):
    config = LearnedVerifierConfig(tmr_threshold=0.5, motioncritic_threshold=0.5,
                                   motioncritic_hand_policy="neutral_terminal_hands")
    request = request_for()
    request.candidate["joints_yup"] = torch.zeros(180, 22, 3)
    tmr = TMRBackend(config)
    monkeypatch.setattr(tmr.encoder, "load", lambda: None)
    monkeypatch.setattr(tmr.encoder, "similarity", lambda text, motion: 0.8)
    monkeypatch.setattr("motion_agent.verification.semantic.real_tmr.joints_to_tmr", lambda *args: torch.zeros(119, 263))
    semantic = next(c for c in learned_builder().build(request).checks if c.check_id == "semantic")
    result = tmr.verify(request, semantic)
    assert result.status == "pass" and result.measured_value == 0.8
    critic = MotionCriticBackend(config)
    monkeypatch.setattr(critic, "load", lambda: None)
    monkeypatch.setattr(critic, "score", lambda motion: [0.2, 0.9])
    check = next(c for c in learned_builder().build(request).checks if c.check_id == "motioncritic")
    result = critic.verify(request, check)
    assert result.status == "fail" and result.measured_value == 0.2
    assert result.observed["window_scores"] == [0.2, 0.9]


def test_frequency_fail_with_coarse_tmr_pass_is_not_averaged(tmp_path):
    request = request_for(PROMPTS[1])
    request.physical_evidence = {"foot_skating": 0, "ground_penetration": 0, "smoothness": 0}
    frequency = next(c for c in learned_builder().build(request).checks if c.direction == "event_frequency")
    response = {"status": "pass", "event": "wave", "body_part": "right_hand", "expected_count": 3,
                "observed_count": 2, "confidence": 1.0, "evidence_timestamps": [1.0, 2.0], "body_part_match": True}
    backend = RealMLLMBackend(LearnedVerifierConfig(output_root=str(tmp_path)), provider=FixtureProvider([response]),
                              storyboard_builder=FakeStoryboards())
    service = VerificationService(semantic_backend=FixtureSemanticBackend(), mllm_backend=backend)
    finding = service._run_cached(request, frequency)
    from motion_agent.verification.aggregator import aggregate_verification
    plan = VerificationPlanBuilder().build(request)
    all_findings = [VerifierFinding(check_id=c.check_id, direction=c.direction, required=c.required, status="pass")
                    for c in plan.checks if c.check_id != frequency.check_id] + [finding]
    report = aggregate_verification(request, plan, all_findings)
    assert not report.overall_pass and finding.diagnostic_code == "FREQUENCY_MISMATCH"


def test_required_not_applicable_cannot_certify_all_required_pass():
    service = VerificationService(semantic_backend=FixtureSemanticBackend("not_applicable"))
    report = service.verify(request_for())
    assert not report.overall_pass and report.status == "incomplete"


def test_nonfinite_smpl_is_technical_failure_even_when_gem_tensor_is_finite():
    request = request_for()
    request.candidate["body_params_global"]["body_pose"][0, 0] = float("nan")
    report = VerificationService(semantic_backend=FixtureSemanticBackend()).verify(request)
    assert report.status == "invalid" and not report.overall_pass
    assert report.findings[0].diagnostic_code == "SMPL_NONFINITE_OR_INVALID"
