from __future__ import annotations

import argparse
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GENMO_ROOT = PROJECT_ROOT / "vendor" / "GENMO"
if str(GENMO_ROOT) not in sys.path:
    sys.path.insert(0, str(GENMO_ROOT))

from gem.utils.smplx_utils import make_smplx  # noqa: E402
from gem.utils.video_io_utils import save_video  # noqa: E402
from scripts.demo.demo_utils import normalize_global_verts, render_global_frames  # noqa: E402


@contextmanager
def _working_directory(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _load_global_params(path: Path, *, nested_key: str | None = None) -> dict[str, torch.Tensor]:
    value: Any = torch.load(path, map_location="cpu", weights_only=False)
    if nested_key is not None:
        if not isinstance(value, dict) or nested_key not in value:
            raise ValueError(f"{path} does not contain {nested_key}")
        value = value[nested_key]
    if not isinstance(value, dict):
        raise ValueError(f"{path} does not contain an SMPL parameter dictionary")

    required = {"body_pose", "global_orient", "transl", "betas"}
    missing = required.difference(value)
    if missing:
        raise ValueError(f"{path} is missing SMPL keys: {sorted(missing)}")
    frame_counts = {int(value[key].shape[0]) for key in required}
    if len(frame_counts) != 1:
        raise ValueError(f"{path} has inconsistent SMPL frame counts: {sorted(frame_counts)}")
    if not all(bool(torch.isfinite(value[key]).all()) for key in required):
        raise ValueError(f"{path} contains NaN or Inf")
    return {key: value[key] for key in required}


def _render(
    *,
    body_model: Any,
    smpl_faces: torch.Tensor,
    params: dict[str, torch.Tensor],
    output_path: Path,
    width: int,
    height: int,
    fps: int,
) -> None:
    vertices = normalize_global_verts(body_model, params)
    frames = render_global_frames(vertices, smpl_faces, width, height)
    save_video(frames, str(output_path), fps=fps)
    print(f"rendered {output_path} frames={len(frames)} fps={fps} size={width}x{height}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Render saved Phase 0 and Phase 4 SMPL outputs.")
    parser.add_argument("--phase0-bundle", type=Path, required=True)
    parser.add_argument("--phase4-global", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/render_comparison"))
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    if args.width <= 0 or args.height <= 0 or args.fps <= 0:
        raise ValueError("width, height, and fps must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    phase0 = _load_global_params(args.phase0_bundle, nested_key="body_params_global")
    phase4 = _load_global_params(args.phase4_global)

    output_dir = args.output_dir.resolve()
    with _working_directory(GENMO_ROOT):
        body_model = make_smplx("supermotion").cuda().eval()
        smpl_faces = torch.from_numpy(body_model.faces.astype("int32")).long()

        with torch.inference_mode():
            _render(
                body_model=body_model,
                smpl_faces=smpl_faces,
                params=phase0,
                output_path=output_dir / "phase0_render.mp4",
                width=args.width,
                height=args.height,
                fps=args.fps,
            )
            _render(
                body_model=body_model,
                smpl_faces=smpl_faces,
                params=phase4,
                output_path=output_dir / "phase4_render.mp4",
                width=args.width,
                height=args.height,
                fps=args.fps,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
