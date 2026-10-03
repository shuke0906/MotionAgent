import sys
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
GENMO_ROOT = REPO_ROOT / "vendor" / "GENMO"
ADAPTER_ROOT = GENMO_ROOT if (GENMO_ROOT / "gem/motionagent.py").is_file() else REPO_ROOT / "integrations/GENMO"
sys.path.insert(0, str(ADAPTER_ROOT))

from gem.motionagent import (  # noqa: E402
    MotionAgentCameraContext,
    MotionAgentTextSegment,
    build_motionagent_text_data,
)


class MotionAgentGEMAdapterTests(unittest.TestCase):
    def test_builds_pure_text_payload_without_video(self):
        data, segment_info = build_motionagent_text_data(
            [
                MotionAgentTextSegment("walk forward", 30, "walk"),
                MotionAgentTextSegment("sit down", 15, "sit"),
            ],
            MotionAgentCameraContext(width=640, height=480),
            seed=7,
        )

        self.assertEqual(int(data["length"]), 45)
        self.assertEqual(data["kp2d"].shape, (45, 17, 3))
        self.assertEqual(data["f_imgseq"].shape, (45, 1024))
        self.assertEqual(data["K_fullimg"].shape, (45, 3, 3))
        self.assertFalse(data["mask"]["has_img_mask"].any())
        self.assertFalse(data["mask"]["has_2d_mask"].any())
        self.assertEqual(data["caption"], "walk forward")
        self.assertEqual(data["seed"], 7)
        self.assertEqual(segment_info[0]["start"], 0)
        self.assertEqual(segment_info[0]["end"], 30)
        self.assertEqual(segment_info[1]["start"], 30)
        self.assertEqual(segment_info[1]["end"], 45)

        multi_text = data["meta"][0]["multi_text_data"]
        self.assertEqual(multi_text["caption"], ["walk forward", "sit down"])
        torch.testing.assert_close(
            multi_text["window_start"],
            torch.tensor([0.0, 30 / 45], dtype=torch.float32),
        )
        torch.testing.assert_close(
            multi_text["window_end"],
            torch.tensor([30 / 45, 1.0], dtype=torch.float32),
        )

    def test_requires_matching_hard_condition_tensors(self):
        with self.assertRaisesRegex(ValueError, "provided together"):
            build_motionagent_text_data(
                [MotionAgentTextSegment("walk", 10)],
                observed_motion_3d=torch.zeros(10, 151),
            )

        with self.assertRaisesRegex(ValueError, "identical shapes"):
            build_motionagent_text_data(
                [MotionAgentTextSegment("walk", 10)],
                observed_motion_3d=torch.zeros(10, 151),
                motion_mask_3d=torch.zeros(9, 151),
            )

    def test_forwards_valid_hard_condition_tensors(self):
        observed = torch.zeros(10, 151)
        mask = torch.zeros(10, 151)
        mask[0, :6] = 1

        data, _ = build_motionagent_text_data(
            [MotionAgentTextSegment("hold a pose", 10)],
            observed_motion_3d=observed,
            motion_mask_3d=mask,
        )

        self.assertEqual(data["observed_motion_3d"].shape, (10, 151))
        torch.testing.assert_close(data["motion_mask_3d"], mask)


if __name__ == "__main__":
    unittest.main()
