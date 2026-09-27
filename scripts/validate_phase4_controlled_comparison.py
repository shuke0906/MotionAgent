from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from motion_agent.compiler import CompilerRequest, GEMTextCondition, MotionCompiler
from motion_agent.generation.candidate_store import CandidateStore
from motion_agent.generation.model_manager import GEMModelManager
from motion_agent.generation.normal_generator import generate_motion
from motion_agent.generation.output_validator import validate_candidate_output
from motion_agent.generation.request_builder import (
    build_generation_request,
    build_generation_request_from_compiler_result,
)


PROMPT = "walk forward, wave the right hand, then sit down"


def _with_explicit_seed(request: Any, seed: int) -> Any:
    return request.model_copy(update={"seeds": [seed]})


def _tensor_diff(left_uri: str, right_uri: str) -> dict[str, float]:
    left = torch.load(left_uri, map_location="cpu", weights_only=False)
    right = torch.load(right_uri, map_location="cpu", weights_only=False)
    difference = (left - right).abs()
    return {
        "max_abs_diff": float(difference.max()),
        "mean_abs_diff": float(difference.mean()),
        "exact_equal": bool(torch.equal(left, right)),
    }


def _validate_stored_candidate(store: CandidateStore, candidate_id: str, frames: int) -> dict[str, Any]:
    loaded = store.load(candidate_id)
    validate_candidate_output(loaded, expected_frames=frames)
    motion = loaded["motion_repr"]
    smpl = loaded["body_params_global"]
    return {
        "candidate_id": loaded["candidate"].candidate_id,
        "generation_id": loaded["candidate"].metadata.generation_id,
        "seed": loaded["candidate"].metadata.seed,
        "condition_fingerprint": loaded["candidate"].metadata.condition_fingerprint,
        "checkpoint": loaded["candidate"].metadata.checkpoint_version,
        "motion_shape": list(motion.shape),
        "motion_finite": bool(torch.isfinite(motion).all()),
        "smpl_shapes": {key: list(value.shape) for key, value in smpl.items()},
        "smpl_finite": all(bool(torch.isfinite(value).all()) for value in smpl.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the controlled Phase 4 300-frame A/B validation.")
    parser.add_argument("--checkpoint", default="vendor/GENMO/inputs/pretrained/gem_smpl.ckpt")
    parser.add_argument("--output-root", default="outputs/render_comparison_fixed")
    parser.add_argument("--artifact-root", default="artifacts/phase4_controlled_comparison")
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    if args.frames != 300 or args.fps != 30:
        raise ValueError("controlled comparison requires exactly 300 frames at 30 fps")

    output_root = Path(args.output_root)
    artifact_root = Path(args.artifact_root)
    output_root.mkdir(parents=True, exist_ok=True)
    artifact_root.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ")
    run_root = artifact_root / run_id

    manager = GEMModelManager(Path(args.checkpoint).resolve())
    health = manager.health()
    if not health["cuda_available"]:
        raise RuntimeError("CUDA PyTorch is not available")
    model = manager.load()

    baseline_condition = GEMTextCondition(
        captions=[PROMPT],
        window_start=[0.0],
        window_end=[1.0],
        total_frames=args.frames,
    )
    baseline_request = _with_explicit_seed(
        build_generation_request(
            condition=baseline_condition,
            fps=args.fps,
            generation_id="gen_controlled_baseline_compound",
            seed=args.seed,
        ),
        args.seed,
    )

    compiled = MotionCompiler().compile(
        CompilerRequest(
            original_request=PROMPT,
            duration_s=args.frames / args.fps,
            fps=args.fps,
            total_frames=args.frames,
        )
    )
    phase4_request = _with_explicit_seed(
        build_generation_request_from_compiler_result(
            compiled,
            generation_id="gen_controlled_phase4_compound",
            seed=args.seed,
        ),
        args.seed,
    )
    phase4_repeat_request = phase4_request.model_copy(deep=True)

    baseline_store = CandidateStore(run_root / "baseline")
    phase4_store = CandidateStore(run_root / "phase4")
    repeat_store = CandidateStore(run_root / "phase4_repeat")

    baseline_result = generate_motion(
        baseline_request,
        model=model,
        checkpoint_version=manager.checkpoint_version,
        candidate_store=baseline_store,
    )
    phase4_result = generate_motion(
        phase4_request,
        model=model,
        checkpoint_version=manager.checkpoint_version,
        candidate_store=phase4_store,
    )
    repeat_result = generate_motion(
        phase4_repeat_request,
        model=model,
        checkpoint_version=manager.checkpoint_version,
        candidate_store=repeat_store,
    )

    for label, result in (
        ("baseline", baseline_result),
        ("phase4", phase4_result),
        ("phase4_repeat", repeat_result),
    ):
        if result.status != "success" or len(result.candidates) != 1:
            raise RuntimeError(f"{label} generation failed: {result.model_dump(mode='json')}")

    baseline_candidate = baseline_result.candidates[0]
    phase4_candidate = phase4_result.candidates[0]
    repeat_candidate = repeat_result.candidates[0]
    same_seed = _tensor_diff(phase4_candidate.motion_repr_uri, repeat_candidate.motion_repr_uri)

    metrics = {
        "run_id": run_id,
        "environment": health,
        "comparison": {
            "prompt": PROMPT,
            "frames": args.frames,
            "fps": args.fps,
            "duration_s": args.frames / args.fps,
            "sampler_seed": args.seed,
            "checkpoint": manager.checkpoint_version,
            "sampler": "gem_checkpoint_default",
            "frozen_gem": bool(health["all_weights_frozen"]),
        },
        "baseline_mode": "single_caption",
        "baseline_condition": baseline_condition.model_dump(mode="json"),
        "baseline_result": baseline_result.model_dump(mode="json"),
        "phase4_mode": "motion_compiler_multi_text",
        "phase4_compiler_condition": compiled.gem_text_condition.model_dump(mode="json"),
        "phase4_result": phase4_result.model_dump(mode="json"),
        "same_seed": {
            "actual_inference_runs": 2,
            "separate_candidate_stores": True,
            "seed": args.seed,
            "fingerprint_primary": phase4_candidate.fingerprint,
            "fingerprint_repeat": repeat_candidate.fingerprint,
            **same_seed,
        },
        "candidate_store": {
            "baseline": _validate_stored_candidate(
                baseline_store, baseline_candidate.candidate_id, args.frames
            ),
            "phase4": _validate_stored_candidate(
                phase4_store, phase4_candidate.candidate_id, args.frames
            ),
            "phase4_repeat": _validate_stored_candidate(
                repeat_store, repeat_candidate.candidate_id, args.frames
            ),
        },
    }

    metrics_path = output_root / "comparison_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
