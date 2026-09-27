from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from motion_agent.compiler import CompilerRequest, MotionCompiler
from motion_agent.generation import build_generation_request, generate_motion
from motion_agent.generation.candidate_store import CandidateStore
from motion_agent.generation.gem_adapter import prepare_pure_text_input
from motion_agent.generation.output_validator import validate_candidate_output


class FakeGEMModel:
    def predict(self, data, static_cam=True, postproc=True, seed=None):
        length = int(data["length"])
        generator = torch.Generator().manual_seed(int(seed))
        pred_x = torch.randn(1, length, 151, generator=generator)
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


class Phase4GenerationTests(unittest.TestCase):
    def test_pure_text_adapter_preserves_compiler_windows(self):
        compiled = MotionCompiler().compile(
            CompilerRequest(
                original_request="walk forward then wave the right hand three times then sit down",
                duration_s=6.0,
                fps=30,
                total_frames=180,
            )
        )
        request = build_generation_request(
            condition=compiled.gem_text_condition,
            fps=compiled.motion_spec.fps,
            seed=7,
        )
        data, segment_info = prepare_pure_text_input(request)
        multi_text = data["meta"][0]["multi_text_data"]

        self.assertEqual(multi_text["caption"], compiled.gem_text_condition.captions)
        torch.testing.assert_close(
            multi_text["window_start"],
            torch.tensor(compiled.gem_text_condition.window_start, dtype=torch.float32),
        )
        torch.testing.assert_close(
            multi_text["window_end"],
            torch.tensor(compiled.gem_text_condition.window_end, dtype=torch.float32),
        )
        self.assertEqual(segment_info[-1]["end"], 180)
        self.assertFalse(data["mask"]["has_img_mask"].any())

    def test_generate_persists_traceable_candidate(self):
        compiled = MotionCompiler().compile(
            CompilerRequest(original_request="walk forward", duration_s=6.0, fps=30, total_frames=180)
        )
        request = build_generation_request(
            condition=compiled.gem_text_condition,
            fps=compiled.motion_spec.fps,
            generation_id="gen_phase4_test",
            seed=42,
        )
        with tempfile.TemporaryDirectory() as tmp:
            store = CandidateStore(Path(tmp))
            result = generate_motion(
                request,
                model=FakeGEMModel(),
                checkpoint_version="fake_ckpt",
                candidate_store=store,
            )
            self.assertEqual(result.status, "success")
            self.assertEqual(len(result.candidates), 1)
            candidate = result.candidates[0]
            self.assertEqual(candidate.metadata.generation_id, "gen_phase4_test")
            self.assertEqual(candidate.metadata.seed, request.seeds[0])
            self.assertEqual(candidate.metadata.condition_fingerprint, request.condition_bundle.condition_fingerprint)
            loaded = store.load(candidate.candidate_id)
            self.assertEqual(loaded["motion_repr"].shape, (180, 151))
            validate_candidate_output(loaded, expected_frames=180)

    def test_same_seed_reuses_candidate_fingerprint_and_different_seed_differs(self):
        compiled = MotionCompiler().compile(
            CompilerRequest(original_request="walk forward", duration_s=6.0, fps=30, total_frames=180)
        )
        first = build_generation_request(
            condition=compiled.gem_text_condition,
            fps=compiled.motion_spec.fps,
            generation_id="gen_seed_test",
            seed=7,
        )
        second = first.model_copy(deep=True)
        third = build_generation_request(
            condition=compiled.gem_text_condition,
            fps=compiled.motion_spec.fps,
            generation_id="gen_seed_test",
            seed=8,
        )
        with tempfile.TemporaryDirectory() as tmp:
            store = CandidateStore(Path(tmp))
            result_a = generate_motion(first, model=FakeGEMModel(), checkpoint_version="fake_ckpt", candidate_store=store)
            result_b = generate_motion(second, model=FakeGEMModel(), checkpoint_version="fake_ckpt", candidate_store=store)
            result_c = generate_motion(third, model=FakeGEMModel(), checkpoint_version="fake_ckpt", candidate_store=store)
            self.assertEqual(result_a.candidates[0].fingerprint, result_b.candidates[0].fingerprint)
            self.assertNotEqual(result_a.candidates[0].fingerprint, result_c.candidates[0].fingerprint)

    def test_invalid_tensor_output_is_rejected(self):
        bad = {
            "motion_repr": torch.zeros(10, 150),
            "body_params_global": {
                "body_pose": torch.zeros(10, 63),
                "global_orient": torch.zeros(10, 3),
                "transl": torch.zeros(10, 3),
                "betas": torch.zeros(10, 10),
            },
        }
        with self.assertRaisesRegex(ValueError, "151"):
            validate_candidate_output(bad, expected_frames=10)


if __name__ == "__main__":
    unittest.main()
