"""Small real backend probes; no GEM generation and no invented verifier passes."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "outputs/phase11_setup"


def write(name, data):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(data, allow_nan=False), flush=True)


def visual_probe(config):
    from PIL import Image, ImageDraw
    from motion_agent.verification.mllm.provider import OpenAIVisualProvider
    from motion_agent.verification.mllm.schemas import SemanticObservation

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "synthetic_visual_smoke.png"
    image = Image.new("RGB", (256, 256), "white")
    drawing = ImageDraw.Draw(image)
    drawing.ellipse((112, 24, 144, 56), outline="black", width=3)
    drawing.line([(128, 56), (128, 160)], fill="black", width=3)
    for points in ([(70, 120), (128, 80), (186, 120)], [(85, 225), (128, 160), (171, 225)]):
        drawing.line(points, fill="black", width=3)
    image.save(path)
    provider = OpenAIVisualProvider(config)
    started = time.perf_counter()
    result = {"probe": "real_mllm_transport", "status": "FAIL", "api_calls": 0,
              "uses_synthetic_image": True, "certifies_candidate": False}
    try:
        response = provider.observe(
            "Describe observed actions using the schema. A single static image cannot establish motion; return uncertain for that.",
            "Expected action: a person walks forward. Only one synthetic frame is provided.",
            {"frames": [{"path": str(path), "frame_index": 0, "timestamp_s": 0.0}]},
            SemanticObservation,
        )
        parsed = SemanticObservation.model_validate_json(response["output_text"])
        result.update(status="PASS", model=response["model"], observation=parsed.model_dump(mode="json"),
                      usage=response.get("usage"), response_id=response.get("response_id"))
    except Exception as exc:
        result.update(error_type=type(exc).__name__, code=getattr(exc, "code", None),
                      unavailable_reason=provider.unavailable_reason)
    result.update(api_calls=provider.network_calls, latency_ms=round((time.perf_counter() - started) * 1000, 2))
    write("mllm_probe.json", result)
    return result["status"] == "PASS"


def learned_probe(config):
    import torch
    from motion_agent.generation.schemas import GenerationRequest
    from motion_agent.verification.config import create_verification_service
    from motion_agent.verification.schemas import VerificationRequest
    from motion_agent.verification.representations import smpl_to_motioncritic, resample_critic

    torch.set_num_threads(4)
    source = ROOT / "outputs/temporal_compiler_validation_final/p00_s7/fixed"
    generation = GenerationRequest.model_validate_json((source / "generation_request.json").read_text())
    folder = source / "candidates/cand_bddf0f0eb4785474"
    params = torch.load(folder / "smpl_global.pt", map_location="cpu", weights_only=True)
    request = VerificationRequest(verification_id="phase11_setup_learned_probe",
        candidate_id="cand_bddf0f0eb4785474", original_request=generation.motion_spec.original_request,
        motion_spec=generation.motion_spec,
        candidate={"motion_repr": torch.load(folder / "motion_repr.pt", map_location="cpu", weights_only=True),
                   "body_params_global": params})
    service = create_verification_service(config)
    request = service.evidence_preparer(request)
    checks = service.plan_builder.build(request).checks
    tmr_check = next(check for check in checks if check.check_id == "semantic")
    critic_check = next(check for check in checks if check.check_id == "motioncritic")
    tmr = service.semantic_backend.verify(request, tmr_check)
    critic = service.naturalness_backend.verify(request, critic_check)
    result = {"probe": "real_learned_backends", "source": "saved_actual_candidate_not_regenerated",
              "candidate_id": request.candidate_id, "body_pose_shape": list(params["body_pose"].shape),
              "fk_available": request.candidate.get("joints_yup") is not None,
              "tmr": tmr.model_dump(mode="json"), "motioncritic_strict": critic.model_dump(mode="json"),
              "tmr_inference_calls": service.semantic_backend.inference_calls,
              "motioncritic_inference_calls": service.naturalness_backend.inference_calls}
    try:
        approximate, conversion = smpl_to_motioncritic(params, "neutral_terminal_hands")
        scores = service.naturalness_backend.score(resample_critic(approximate, generation.fps))
        result["motioncritic_exploratory"] = {"scores": scores, "conversion": conversion,
                                            "certifies_exact_candidate": False}
    except Exception as exc:
        result["motioncritic_exploratory"] = {"error_type": type(exc).__name__}
    write("learned_probe.json", result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--visual", action="store_true")
    parser.add_argument("--learned", action="store_true")
    args = parser.parse_args()
    os.chdir(ROOT)
    from motion_agent.llm.config import load_dotenv
    for name in ("OPENAI_API_KEY", "OPENAI_MODEL", "OPENAI_BASE_URL", "MLLM_API_KEY", "MLLM_MODEL", "MLLM_BASE_URL"):
        os.environ.pop(name, None)
    load_dotenv(ROOT / ".env")
    os.environ.setdefault("MLLM_MODEL", os.environ.get("OPENAI_MODEL", "gpt-4o-2024-08-06"))
    from motion_agent.verification.config import LearnedVerifierConfig
    config = LearnedVerifierConfig.load()
    model = ROOT / "vendor/GENMO/inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz"
    if model.is_file():
        config.smplx_model_path = str(model)
    config.output_root = str(OUT)
    if args.visual and not visual_probe(config):
        return 1
    if args.learned:
        learned_probe(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
