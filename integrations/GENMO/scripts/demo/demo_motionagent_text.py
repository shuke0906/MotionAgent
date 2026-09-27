# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-OneWay-Noncommercial
"""Pure-text GEM-SMPL smoke test entry point for MotionAgent.

Example:
    python scripts/demo/demo_motionagent_text.py \
        --text "a person walks forward" \
        --ckpt_path inputs/pretrained/gem_smpl.ckpt \
        --no_render
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from gem.motionagent import (  # noqa: E402
    MotionAgentCameraContext,
    MotionAgentTextSegment,
    build_motionagent_text_data,
)


def _parse_text_segments(values: list[str], frames: int) -> list[MotionAgentTextSegment]:
    segments: list[MotionAgentTextSegment] = []
    for i, value in enumerate(values):
        if "::" in value:
            length_text, caption = value.split("::", 1)
            length = int(length_text)
        else:
            caption = value
            length = frames
        segments.append(MotionAgentTextSegment(caption=caption.strip(), length=length, name=f"text_{i}"))
    return segments


def main() -> None:
    parser = argparse.ArgumentParser(description="MotionAgent pure-text GEM-SMPL demo")
    parser.add_argument(
        "--text",
        nargs="+",
        required=True,
        help="Text segment(s). Use '<frames>::caption' to override one segment length.",
    )
    parser.add_argument("--text_length", type=int, default=300)
    parser.add_argument("--ckpt_path", default=None)
    parser.add_argument("--output_root", default="outputs")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no_render", action="store_true")
    args = parser.parse_args()

    ckpt_path = args.ckpt_path
    if ckpt_path is None or not Path(ckpt_path).exists():
        from gem.utils.hf_utils import download_checkpoint

        print("[Checkpoint] Not found locally. Downloading from HuggingFace...")
        ckpt_path = download_checkpoint()

    segments = _parse_text_segments(args.text, args.text_length)
    camera = MotionAgentCameraContext(width=args.width, height=args.height, static_camera=True)
    data, segment_info = build_motionagent_text_data(segments, camera, seed=args.seed)

    from scripts.demo.demo_utils import load_model, run_inference

    model = load_model(ckpt_path, load_text_encoder=True)
    pred = run_inference(model, data, static_cam=True)

    output_dir = os.path.join(args.output_root, "motionagent_text")
    os.makedirs(output_dir, exist_ok=True)
    if args.no_render:
        params_path = os.path.join(output_dir, "smpl_params.pt")
        save_dict = {}
        for key_group in ("body_params_incam", "body_params_global"):
            if key_group in pred:
                save_dict[key_group] = {k: v.cpu() for k, v in pred[key_group].items()}
        if "K_fullimg" in pred:
            save_dict["K_fullimg"] = pred["K_fullimg"].cpu()
        save_dict["segment_info"] = segment_info
        torch.save(save_dict, params_path)
        print(f"[MotionAgent] SMPL parameters saved to {params_path}")
    else:
        from scripts.demo.demo_smpl import render_results_mixed

        render_inputs = [
            {"type": "text", "caption": seg.caption, "name": seg.name or f"text_{i}"}
            for i, seg in enumerate(segments)
        ]
        render_results_mixed(
            pred=pred,
            segments=[],
            segment_info=segment_info,
            input_segments=render_inputs,
            output_dir=output_dir,
            no_render=False,
            ref_width=args.width,
            ref_height=args.height,
        )
    print(f"[MotionAgent] Saved pure-text GEM output to {output_dir}")


if __name__ == "__main__":
    with torch.inference_mode():
        main()
