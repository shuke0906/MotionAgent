"""Security and baseline contracts for the small real-run drivers."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tarfile


ROOT = Path(__file__).resolve().parents[2]


def driver(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_transfer_uses_dotenv_over_stale_environment_and_only_api_settings(tmp_path, monkeypatch, capsys):
    module = driver("sync_runpod_api_env")
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setenv("MOTIONAGENT_SSH_HOST", "test-user@example.invalid")
    monkeypatch.setenv("OPENAI_API_KEY", "stale-inherited-test-key")
    monkeypatch.setenv("MLLM_MODEL", "stale-model")
    (tmp_path / ".env").write_text("OPENAI_API_KEY=fake-test-key\nOPENAI_MODEL=test-model\nRUNPOD_API_KEY=do-not-transfer\n")
    calls = []

    def receive(command, **kwargs):
        calls.append((command, json.loads(kwargs["input"])))
        return SimpleNamespace(returncode=0, stdout="REMOTE_API_SETTINGS_INSTALLED_MODE_600\n")

    monkeypatch.setattr(module.subprocess, "run", receive)
    assert module.main() == 0
    command, payload = calls[0]
    assert payload["OPENAI_API_KEY"] == "fake-test-key"
    assert payload["MLLM_API_KEY"] == "fake-test-key"
    assert payload["MLLM_MODEL"] == "test-model"
    assert "RUNPOD_API_KEY" not in payload
    assert "fake-test-key" not in " ".join(command)
    assert "fake-test-key" not in capsys.readouterr().out


def test_runtime_archive_excludes_dotenv_and_caches(tmp_path, monkeypatch):
    module = driver("package_phase11_runtime")
    folder = tmp_path / "motion_agent"
    folder.mkdir()
    (folder / "current.py").write_text("pass\n")
    (folder / ".env").write_text("FAKE_SECRET=excluded\n")
    cache = folder / "__pycache__"
    cache.mkdir()
    (cache / "old.pyc").write_bytes(b"cache")
    destination = tmp_path / "bundle.tar.gz"
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "DESTINATION", destination)
    monkeypatch.setattr(module, "TREES", ["motion_agent"])
    monkeypatch.setattr(module, "FILES", [])
    module.main()
    with tarfile.open(destination) as archive:
        assert archive.getnames() == ["motion_agent/current.py"]


def test_baseline_has_one_raw_condition_without_motion_specification():
    module = driver("run_phase11_minimal")
    request = module.raw_request(module.PROMPT, 420, "test_raw_baseline")
    assert request.text_condition.captions == [module.PROMPT]
    assert request.motion_spec is None
    assert request.num_candidates == 1
    assert len(request.seeds) == 1
    assert request.total_frames == 420
    assert not request.condition_bundle.active_constraint_ids
    assert not request.condition_bundle.active_keyframe_ids


def test_split_simultaneous_events_reference_existing_event_ids():
    from motion_agent.compiler.schemas import MotionSegment, MotionSpecification, TemporalRelation
    from motion_agent.verification.mllm.prompts import expected_requirements
    from motion_agent.verification.schemas import VerificationRequest, VerifierCheck

    spec = MotionSpecification(original_request="Walk while waving.", duration_s=6, fps=30, total_frames=180,
        segments=[MotionSegment(segment_id=0, action="walk", start_frame=0, end_frame=180),
                  MotionSegment(segment_id=1, action="wave", body_parts=["right_hand"],
                    start_frame=0, end_frame=180, parent_segment_id=0, simultaneous_with=0,
                    temporal_relation=TemporalRelation(type="simultaneous", related_action="walk"))])
    request = VerificationRequest(verification_id="test", candidate_id="test", original_request=spec.original_request,
                                  motion_spec=spec)
    check = VerifierCheck(check_id="temporal", direction="event_temporal", evaluator="mllm_backend", required=True)
    requirements = expected_requirements(request, check)
    ids = {event["event_id"] for event in requirements["events"]}
    assert len(requirements["relations"]) == 1
    for relation in requirements["relations"]:
        assert relation["left"] in ids
        assert relation["right"] in ids
        assert {relation["left"], relation["right"]} == {"0:walk", "1:wave"}


def test_uncertain_visibility_is_not_converted_to_missing_action_failure():
    from motion_agent.compiler.motion_compiler import CompilerRequest, MotionCompiler
    from motion_agent.verification.mllm.backend import normalize_observation
    from motion_agent.verification.mllm.prompts import expected_requirements
    from motion_agent.verification.mllm.schemas import SemanticObservation
    from motion_agent.verification.schemas import VerificationRequest, VerifierCheck

    spec = MotionCompiler().compile(CompilerRequest(original_request="A person walks forward.")).motion_spec
    request = VerificationRequest(verification_id="test", candidate_id="test", original_request=spec.original_request,
                                  motion_spec=spec)
    check = VerifierCheck(check_id="integrity", direction="event_integrity", evaluator="mllm_backend", required=True)
    observation = SemanticObservation(status="uncertain", observed_actions=[],
        missing_actions=["Walking cannot be resolved at this scale"], body_part_match=None,
        direction_match=None, confidence=.8, evidence=[])
    assert normalize_observation(request, check, observation, expected_requirements(request, check))[:2] == (
        "uncertain", "MLLM_UNCERTAIN")
