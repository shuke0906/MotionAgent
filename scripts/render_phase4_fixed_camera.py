from __future__ import annotations

import argparse
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GENMO_ROOT = PROJECT_ROOT / "vendor" / "GENMO"
if str(GENMO_ROOT) not in sys.path:
    sys.path.insert(0, str(GENMO_ROOT))

from gem.utils.cam_utils import create_camera_sensor  # noqa: E402
from gem.utils.smplx_utils import make_smplx  # noqa: E402
from gem.utils.video_io_utils import save_video  # noqa: E402
from gem.utils.vis.o3d_render import Settings, create_meshes, get_ground  # noqa: E402
from scripts.demo.demo_utils import normalize_global_verts, render_side_by_side  # noqa: E402


CAMERA_POSITION = np.array([8.0, 5.5, 9.0], dtype=np.float64)
CAMERA_TARGET = np.array([0.0, 1.0, 2.5], dtype=np.float64)
CAMERA_UP = np.array([0.0, 1.0, 0.0], dtype=np.float64)
CAMERA_FOV_DEG = 45.0
GROUND_SIZE = 12.0
GROUND_CENTER_X = 0.0
GROUND_CENTER_Z = 2.5


@contextmanager
def _working_directory(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _load_global_params(path: Path) -> dict[str, torch.Tensor]:
    value: Any = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(value, dict):
        raise ValueError(f"{path} does not contain an SMPL parameter dictionary")
    required = {"body_pose", "global_orient", "transl", "betas"}
    missing = required.difference(value)
    if missing:
        raise ValueError(f"{path} is missing SMPL keys: {sorted(missing)}")
    frame_counts = {int(value[key].shape[0]) for key in required}
    if frame_counts != {300}:
        raise ValueError(f"{path} must contain exactly 300 frames, got {sorted(frame_counts)}")
    if not all(bool(torch.isfinite(value[key]).all()) for key in required):
        raise ValueError(f"{path} contains NaN or Inf")
    return {key: value[key] for key in required}


def render_fixed_camera_frames(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    width: int,
    height: int,
) -> np.ndarray:
    import open3d as o3d
    from tqdm import tqdm

    settings = Settings()
    lit_material = settings._materials[Settings.LIT]
    _, _, intrinsics = create_camera_sensor(width, height, fov_deg=CAMERA_FOV_DEG)

    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
    renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])
    renderer.scene.set_lighting(
        renderer.scene.LightingProfile.NO_SHADOWS,
        np.array([0.577, -0.577, -0.577]),
    )
    renderer.scene.camera.set_projection(
        intrinsics.cpu().double().numpy(), 0.1, 100.0, float(width), float(height)
    )
    renderer.scene.camera.look_at(CAMERA_TARGET, CAMERA_POSITION, CAMERA_UP)

    ground_vertices, ground_faces, ground_colors = get_ground(
        GROUND_SIZE, GROUND_CENTER_X, GROUND_CENTER_Z
    )
    ground_mesh = create_meshes(ground_vertices, ground_faces, ground_colors[..., :3])
    ground_material = o3d.visualization.rendering.MaterialRecord()
    ground_material.shader = Settings.LIT
    renderer.scene.add_geometry("mesh_ground", ground_mesh, ground_material)

    body_color = torch.tensor([0.69019608, 0.39215686, 0.95686275])
    frames = []
    for index in tqdm(range(vertices.shape[0]), desc="Fixed-camera render", leave=False):
        mesh = create_meshes(vertices[index], faces, body_color)
        if index > 0:
            renderer.scene.remove_geometry(f"mesh_{index - 1}")
        renderer.scene.add_geometry(f"mesh_{index}", mesh, lit_material)
        frames.append(np.asarray(renderer.render_to_image()))
    return np.stack(frames)


def main() -> int:
    parser = argparse.ArgumentParser(description="Render a controlled Phase 4 A/B with one fixed camera.")
    parser.add_argument("--baseline-global", type=Path, required=True)
    parser.add_argument("--phase4-global", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/render_comparison_fixed"))
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    if args.width <= 0 or args.height <= 0 or args.fps != 30:
        raise ValueError("positive dimensions and exactly 30 fps are required")

    baseline_params = _load_global_params(args.baseline_global)
    phase4_params = _load_global_params(args.phase4_global)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with _working_directory(GENMO_ROOT):
        body_model = make_smplx("supermotion").cuda().eval()
        faces = torch.from_numpy(body_model.faces.astype("int32")).long()
        with torch.inference_mode():
            baseline_vertices = normalize_global_verts(body_model, baseline_params)
            phase4_vertices = normalize_global_verts(body_model, phase4_params)
            baseline_frames = render_fixed_camera_frames(
                baseline_vertices, faces, args.width, args.height
            )
            phase4_frames = render_fixed_camera_frames(
                phase4_vertices, faces, args.width, args.height
            )

    baseline_path = output_dir / "baseline_compound_fixed_camera.mp4"
    phase4_path = output_dir / "phase4_compound_fixed_camera.mp4"
    side_by_side_path = output_dir / "side_by_side_fixed_camera.mp4"
    save_video(baseline_frames, str(baseline_path), fps=args.fps)
    save_video(phase4_frames, str(phase4_path), fps=args.fps)
    save_video(render_side_by_side(baseline_frames, phase4_frames), str(side_by_side_path), fps=args.fps)

    viewing_direction = CAMERA_TARGET - CAMERA_POSITION
    viewing_direction = viewing_direction / np.linalg.norm(viewing_direction)
    print(
        {
            "camera_position": CAMERA_POSITION.tolist(),
            "camera_target": CAMERA_TARGET.tolist(),
            "camera_up": CAMERA_UP.tolist(),
            "camera_fov_deg": CAMERA_FOV_DEG,
            "camera_height": float(CAMERA_POSITION[1]),
            "viewing_direction": viewing_direction.tolist(),
            "ground_size": GROUND_SIZE,
            "frames": int(baseline_frames.shape[0]),
            "fps": args.fps,
            "outputs": [str(baseline_path), str(phase4_path), str(side_by_side_path)],
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
