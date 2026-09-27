from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from motion_agent.compiler import CompilerRequest, MotionCompiler
from motion_agent.generation.candidate_store import CandidateStore
from motion_agent.generation.model_manager import GEMModelManager
from motion_agent.generation.normal_generator import generate_motion
from motion_agent.generation.request_builder import build_generation_request_from_compiler_result


def tensor_diff(left_uri: str, right_uri: str) -> dict[str, float]:
    left = torch.load(left_uri, map_location="cpu", weights_only=False)
    right = torch.load(right_uri, map_location="cpu", weights_only=False)
    diff = (left - right).abs()
    return {"max_abs_diff": float(diff.max()), "mean_abs_diff": float(diff.mean())}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run real Phase 4 CUDA GEM validation.")
    parser.add_argument("--checkpoint", default="vendor/GENMO/inputs/pretrained/gem_smpl.ckpt")
    parser.add_argument("--output-root", default="outputs/phase4_validation")
    parser.add_argument("--artifact-root", default="artifacts/phase4_validation")
    parser.add_argument("--frames", type=int, default=180)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    output_root = Path(args.output_root)
    artifact_root = Path(args.artifact_root)
    output_root.mkdir(parents=True, exist_ok=True)
    artifact_root.mkdir(parents=True, exist_ok=True)

    manager = GEMModelManager(Path(args.checkpoint).resolve())
    health = manager.health()
    if not health["cuda_available"]:
        raise RuntimeError("CUDA PyTorch is not available")
    model = manager.load()
    store = CandidateStore(artifact_root)

    metrics: dict[str, object] = {"environment": health}

    single_compiled = MotionCompiler().compile(
        CompilerRequest(original_request="walk forward", duration_s=args.frames / args.fps, fps=args.fps, total_frames=args.frames)
    )
    single_request = build_generation_request_from_compiler_result(
        single_compiled,
        generation_id="gen_phase4_single",
        seed=7,
    )
    single_start = time.perf_counter()
    single_result = generate_motion(
        single_request,
        model=model,
        checkpoint_version=manager.checkpoint_version,
        candidate_store=store,
    )
    metrics["single_text"] = single_result.model_dump(mode="json")
    metrics["single_text_latency_s"] = time.perf_counter() - single_start

    multi_compiled = MotionCompiler().compile(
        CompilerRequest(
            original_request="walk forward then wave the right hand three times then sit down",
            duration_s=args.frames / args.fps,
            fps=args.fps,
            total_frames=args.frames,
        )
    )
    multi_request = build_generation_request_from_compiler_result(
        multi_compiled,
        generation_id="gen_phase4_multi",
        seed=11,
    )
    multi_result = generate_motion(
        multi_request,
        model=model,
        checkpoint_version=manager.checkpoint_version,
        candidate_store=store,
    )
    metrics["multi_text"] = multi_result.model_dump(mode="json")
    metrics["multi_text_compiler_condition"] = multi_compiled.gem_text_condition.model_dump(mode="json")

    same_a = build_generation_request_from_compiler_result(
        single_compiled,
        generation_id="gen_phase4_seed",
        seed=123,
    )
    same_b = same_a.model_copy(deep=True)
    diff_seed = build_generation_request_from_compiler_result(
        single_compiled,
        generation_id="gen_phase4_seed",
        seed=124,
    )
    repro_run = datetime.now(timezone.utc).strftime("repro_%Y%m%dT%H%M%SZ")
    result_a = generate_motion(
        same_a,
        model=model,
        checkpoint_version=manager.checkpoint_version,
        candidate_store=CandidateStore(artifact_root / repro_run / "same_seed_a"),
    )
    result_b = generate_motion(
        same_b,
        model=model,
        checkpoint_version=manager.checkpoint_version,
        candidate_store=CandidateStore(artifact_root / repro_run / "same_seed_b"),
    )
    result_c = generate_motion(diff_seed, model=model, checkpoint_version=manager.checkpoint_version, candidate_store=store)
    metrics["same_seed"] = {
        "actual_inference_runs": 2,
        "separate_candidate_stores": True,
        "fingerprint_a": result_a.candidates[0].fingerprint,
        "fingerprint_b": result_b.candidates[0].fingerprint,
        **tensor_diff(result_a.candidates[0].motion_repr_uri, result_b.candidates[0].motion_repr_uri),
    }
    metrics["different_seed"] = {
        "fingerprint_a": result_a.candidates[0].fingerprint,
        "fingerprint_c": result_c.candidates[0].fingerprint,
        **tensor_diff(result_a.candidates[0].motion_repr_uri, result_c.candidates[0].motion_repr_uri),
    }
    loaded = store.load(single_result.candidates[0].candidate_id)
    metrics["candidate_store"] = {
        "loaded_candidate_id": loaded["candidate"].candidate_id,
        "generation_id": loaded["candidate"].metadata.generation_id,
        "seed": loaded["candidate"].metadata.seed,
        "condition_fingerprint": loaded["candidate"].metadata.condition_fingerprint,
        "checkpoint": loaded["candidate"].metadata.checkpoint_version,
    }

    metrics_path = output_root / "phase4_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
