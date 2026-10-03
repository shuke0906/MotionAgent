"""Fetch only the official NVIDIA GEM-SMPL checkpoint and verify its LFS SHA256."""

import hashlib
import json
from pathlib import Path
import time
import urllib.request


ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main():
    with urllib.request.urlopen("https://huggingface.co/api/models/nvidia/GEM-X?blobs=true", timeout=45) as response:
        metadata = json.load(response)
    asset = next(item for item in metadata["siblings"] if item["rfilename"] == "gem_smpl.ckpt")
    expected = asset["lfs"]["sha256"]
    revision = metadata["sha"]
    target = ROOT / "vendor/GENMO/inputs/pretrained/gem_smpl.ckpt"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file():
        if digest(target) != expected:
            raise RuntimeError("EXISTING_GEM_CHECKSUM_MISMATCH")
        print("GEM_CHECKPOINT_ALREADY_VALID", flush=True)
        return
    partial = target.with_suffix(".ckpt.partial")
    offset = partial.stat().st_size if partial.exists() else 0
    url = "https://huggingface.co/nvidia/GEM-X/resolve/" + revision + "/gem_smpl.ckpt?download=true"
    request = urllib.request.Request(url, headers={"Range": "bytes=" + str(offset) + "-"} if offset else {})
    started = time.monotonic()
    progress_time = started
    with urllib.request.urlopen(request, timeout=120) as response:
        append = bool(offset and response.status == 206)
        downloaded = offset if append else 0
        with partial.open("ab" if append else "wb") as stream:
            while chunk := response.read(8 * 1024 * 1024):
                stream.write(chunk)
                downloaded += len(chunk)
                if time.monotonic() - progress_time >= 15:
                    print("GEM_DOWNLOADED_BYTES=" + str(downloaded), flush=True)
                    progress_time = time.monotonic()
    if partial.stat().st_size != asset["lfs"]["size"] or digest(partial) != expected:
        raise RuntimeError("GEM_CHECKSUM_MISMATCH")
    partial.replace(target)
    result = {"status": "PASS", "repository": "nvidia/GEM-X", "revision": revision,
              "filename": "gem_smpl.ckpt", "bytes": target.stat().st_size,
              "sha256": expected, "download_seconds": round(time.monotonic() - started, 2)}
    output = ROOT / "outputs/phase11_setup/gem_asset.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
