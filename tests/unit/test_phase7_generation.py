from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from motion_agent.compiler import CompilerRequest, MotionCompiler
from motion_agent.constraints.schemas import HardMotionCondition
from motion_agent.constraints.store import ConditionStore
from motion_agent.generation import (
    GEMGenerationWorker,
    assemble_generation_conditions,
    build_generation_request,
    generate_motion,
)
from motion_agent.generation.candidate_store import CandidateStore
from motion_agent.generation.schemas import CandidateMetadata, SamplerMetadata
from motion_agent.generation.validators import validate_generation_request


class HardAwareFakeGEM:
    def __init__(self, *, fail_on_seed: int | None = None) -> None:
        self.fail_on_seed = fail_on_seed
        self.calls = 0

    def predict(self, data, static_cam=True, postproc=True, seed=None):
        self.calls += 1
        if seed == self.fail_on_seed:
            raise torch.cuda.OutOfMemoryError("synthetic oom")
        length = int(data["length"])
        generator = torch.Generator().manual_seed(int(seed))
        pred_x = torch.randn(1, length, 151, generator=generator)
        if "observed_motion_3d" in data:
            observed = data["observed_motion_3d"].float()
            mask = data["motion_mask_3d"].float()
            pred_x[0] = pred_x[0] * (1 - mask) + observed * mask
        smpl = {
            "body_pose": torch.randn(length, 63, generator=generator),
            "global_orient": torch.randn(length, 3, generator=generator),
            "transl": torch.randn(length, 3, generator=generator),
            "betas": torch.randn(length, 10, generator=generator),
        }
        return {
            "body_params_global": smpl,
            "body_params_incam": smpl,
            "net_outputs": {"model_output": {"pred_x": pred_x}},
        }


def compiled_condition(prompt: str = "walk forward then wave right hand twice then sit down"):
    result = MotionCompiler().compile(
        CompilerRequest(original_request=prompt, duration_s=6.0, fps=30, total_frames=180)
    )
    return result.gem_text_condition


class Phase7GenerationTests(unittest.TestCase):
    def test_k_loop_cache_idempotency_and_seed_diversity(self):
        condition = compiled_condition("walk forward")
        request = build_generation_request(
            condition=condition,
            fps=30,
            generation_id="gen_phase7_k",
            num_candidates=3,
            seeds=[7, 8, 9],
        )
        model = HardAwareFakeGEM()
        with tempfile.TemporaryDirectory() as tmp:
            store = CandidateStore(Path(tmp))
            first = generate_motion(request, model=model, checkpoint_version="fake", candidate_store=store)
            second = generate_motion(request, model=model, checkpoint_version="fake", candidate_store=store)
            self.assertEqual(first.status, "success")
            self.assertEqual(len(first.candidates), 3)
            self.assertEqual(first.inference_calls, 3)
            self.assertEqual(second.cache_hits, 3)
            self.assertEqual(second.inference_calls, 0)
            self.assertEqual(model.calls, 3)
            motions = [store.load(candidate.candidate_id)["motion_repr"] for candidate in first.candidates]
            self.assertFalse(torch.equal(motions[0], motions[1]))

    def test_partial_failure_preserves_successful_candidates_with_typed_oom(self):
        condition = compiled_condition("walk forward")
        request = build_generation_request(
            condition=condition,
            fps=30,
            generation_id="gen_phase7_partial",
            num_candidates=2,
            seeds=[7, 8],
        )
        with tempfile.TemporaryDirectory() as tmp:
            result = generate_motion(
                request,
                model=HardAwareFakeGEM(fail_on_seed=8),
                checkpoint_version="fake",
                candidate_store=CandidateStore(Path(tmp)),
            )
            self.assertEqual(result.status, "partial")
            self.assertEqual(len(result.candidates), 1)
            self.assertEqual(result.failed_candidates[0].error_type, "oom")
            self.assertTrue(result.failed_candidates[0].cleanup_performed)

    def test_guided_generation_is_feature_gated(self):
        request = build_generation_request(
            condition=compiled_condition("walk forward"),
            fps=30,
            strategy="guided",
            seed=7,
        )
        preflight = validate_generation_request(request, guided_generation_available=False)
        self.assertEqual(preflight.status, "blocked")
        self.assertIn("guided_generation_not_available", preflight.errors)

    def test_hard_condition_injection_and_constraint_safe_postprocess(self):
        condition = compiled_condition("walk forward")
        with tempfile.TemporaryDirectory() as tmp:
            condition_store = ConditionStore(Path(tmp) / "conditions")
            values = torch.zeros(180, 151)
            mask = torch.zeros(180, 151)
            values[90, 0:6] = 3.0
            mask[90, 0:6] = 1.0
            handle = condition_store.save_hard_condition(
                HardMotionCondition(values=values, mask=mask, source_constraint_id="constraint_fixture")
            )
            assembled = assemble_generation_conditions(
                text_condition_id="text_fixture",
                text_condition_payload=condition.model_dump(mode="json"),
                total_frames=180,
                hard_condition_handles=[handle],
                active_constraint_ids=["constraint_fixture"],
                store=condition_store,
            )
            request = build_generation_request(
                condition=condition,
                fps=30,
                condition_bundle=assembled.bundle,
                seed=7,
            )
            self.assertEqual(request.postprocess_policy, "constraint_safe")
            result = generate_motion(
                request,
                model=HardAwareFakeGEM(),
                checkpoint_version="fake",
                candidate_store=CandidateStore(Path(tmp) / "candidates"),
                condition_store=condition_store,
            )
            loaded = CandidateStore(Path(tmp) / "candidates").load(result.candidates[0].candidate_id)
            torch.testing.assert_close(loaded["motion_repr"][90, 0:6], torch.full((6,), 3.0))

    def test_segment_inpainting_preserves_outside_target_and_betas(self):
        condition = compiled_condition()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = CandidateStore(root / "candidates")
            previous = torch.arange(180 * 151, dtype=torch.float32).reshape(180, 151) / 1000.0
            metadata = CandidateMetadata(
                seed=1,
                generation_id="gen_previous",
                condition_fingerprint="previous",
                candidate_fingerprint="previous_candidate",
                checkpoint_version="fake",
                sampler=SamplerMetadata(seed=1, checkpoint_version="fake"),
                scope="full",
                postprocess_policy="none",
                runtime_ms=1.0,
                frame_count=180,
                motion_dim=151,
                technical_valid=True,
            )
            previous_record = store.save(
                tensors={
                    "motion_repr": previous,
                    "body_params_global": {
                        "body_pose": torch.zeros(180, 63),
                        "global_orient": torch.zeros(180, 3),
                        "transl": torch.zeros(180, 3),
                        "betas": torch.zeros(180, 10),
                    },
                },
                metadata=metadata,
            )
            request = build_generation_request(
                condition=condition,
                fps=30,
                generation_id="gen_segment",
                scope="segment",
                target_segments=[1],
                previous_candidate_id=previous_record.candidate.candidate_id,
                seed=7,
            )
            result = generate_motion(
                request,
                model=HardAwareFakeGEM(),
                checkpoint_version="fake",
                candidate_store=store,
                condition_store=ConditionStore(root / "conditions"),
            )
            motion = store.load(result.candidates[0].candidate_id)["motion_repr"]
            start = round(condition.window_start[1] * condition.total_frames)
            end = round(condition.window_end[1] * condition.total_frames)
            torch.testing.assert_close(motion[:start], previous[:start])
            torch.testing.assert_close(motion[end:], previous[end:])
            torch.testing.assert_close(motion[:, 126:136], previous[:, 126:136])
            self.assertFalse(torch.equal(motion[start:end, 0:126], previous[start:end, 0:126]))

    def test_worker_health_reports_persistent_model_and_latency(self):
        condition = compiled_condition("walk forward")
        request = build_generation_request(condition=condition, fps=30, seed=7)
        with tempfile.TemporaryDirectory() as tmp:
            worker = GEMGenerationWorker(
                worker_id="worker_test",
                model_factory=HardAwareFakeGEM,
                checkpoint_version="fake",
                candidate_store=CandidateStore(Path(tmp)),
            )
            result = worker.run(request)
            health = worker.health(queue_depth=0)
            self.assertEqual(result.status, "success")
            self.assertTrue(health.model_loaded)
            self.assertTrue(health.available)
            self.assertEqual(health.jobs_completed, 1)
            self.assertIsNotNone(health.last_latency_ms)


if __name__ == "__main__":
    unittest.main()
