#!/usr/bin/env python3
"""Download checksum-pinned GVHMR body-model support tensors for GENMO."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import tempfile
import urllib.request
from pathlib import Path


BASE_URL = (
    "https://raw.githubusercontent.com/zju3dv/GVHMR/main/"
    "hmr4d/utils/body_model"
)
FILES = {
    "coco_aug_dict.pth": "9d045cc3e507e1f7d91ef89904cdfa26e29cf18710f52f9fd305a88ae5d4e539",
    "smplx2smpl_sparse.pt": "0fc821a9e79ec3e76d6a9796b96d5bef8cd67055e18497bbe370d8aed9e07e06",
    "smpl_coco17_J_regressor.pt": "bacdaf756629493994cc869f4c27d179f5e4a5d06b8797ee3dcb94571522079f",
    "smplx_verts437.pt": "ef0ea64c470a1fea80adb5e4b866c78a792a04f5f98744c9367b15e57bfb4a4d",
    "smpl_neutral_J_regressor.pt": "70e3213bd30fe8d8ce37b54675282745e406f915a51511a003aeff99b6da04cf",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(name: str, expected_hash: str, output_dir: Path, force: bool) -> None:
    destination = output_dir / name
    if destination.exists():
        actual_hash = sha256(destination)
        if actual_hash == expected_hash:
            print(f"[ok] {destination}")
            return
        if not force:
            raise SystemExit(
                f"Refusing to replace {destination}: SHA256 is {actual_hash}. "
                "Use --force to replace it."
            )

    with tempfile.NamedTemporaryFile(dir=output_dir, delete=False) as temporary:
        temporary_path = Path(temporary.name)
        try:
            with urllib.request.urlopen(f"{BASE_URL}/{name}") as response:
                shutil.copyfileobj(response, temporary)
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise

    actual_hash = sha256(temporary_path)
    if actual_hash != expected_hash:
        temporary_path.unlink(missing_ok=True)
        raise SystemExit(
            f"Checksum mismatch for {name}: expected {expected_hash}, got {actual_hash}"
        )
    temporary_path.replace(destination)
    print(f"[downloaded] {destination}")


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    default_output = repo_root / "gem" / "utils" / "body_model"
    parser = argparse.ArgumentParser(
        description="Download the external GVHMR tensors referenced by GENMO."
    )
    parser.add_argument("--output-dir", type=Path, default=default_output)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, expected_hash in FILES.items():
        download_file(name, expected_hash, args.output_dir, args.force)


if __name__ == "__main__":
    main()
