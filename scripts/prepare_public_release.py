"""Package selected real demo evidence and audit a separate release checkout."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path
import re
import shutil
import subprocess


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/phase11_real_e2e_download/phase11_real_e2e"
TREES = ("motion_agent", "configs", "scripts", "tests", "planner_skills", "doc", "docs", "integrations")
SUFFIXES = {".py", ".md", ".json", ".yaml", ".yml", ".txt", ".patch", ".sh", ".gif"}
OMIT = {"__pycache__", ".git", ".pytest_cache", ".cache"}


def git(checkout, *args, binary=False):
    return subprocess.check_output(["git", "-c", "safe.directory=" + checkout.resolve().as_posix(),
                                    "-C", str(checkout), *args], text=not binary)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sanitize(value):
    if isinstance(value, dict):
        return {key: sanitize(item) for key, item in value.items()
                if key not in {"ssh_endpoint", "authorization", "api_key", "headers"}}
    if isinstance(value, list):
        return [sanitize(item) for item in value]
    if isinstance(value, str):
        value = re.sub(r"/workspace/MotionAgent/", "", value)
        value = re.sub(r"C:[\\/]+Users[\\/]+[^\\/]+[\\/]+Downloads[\\/]+MotionAgent_Final_Implementation_Guide[\\/]+", "", value)
        value = re.sub(r"\b(?:\d{1,3}\.){3}\d{1,3}:\d+\b", "<REMOTE_ENDPOINT>", value)
        value = re.sub(r"/root/\.config/motionagent/api\.env", "<PRIVATE_API_CONFIG>", value)
    return value


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sanitize(value), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def finding_excerpt(report):
    return {"candidate_id": report["candidate_id"], "verification_id": report["verification_id"],
            "status": report["status"], "overall_pass": report["overall_pass"],
            "findings": [{key: item.get(key) for key in
                          ("check_id", "direction", "status", "required", "diagnostic_code", "expected", "observed", "measured_value", "threshold")}
                         for item in report["findings"]]}


def package_media():
    import imageio_ffmpeg

    destination = ROOT / "assets/demo"
    destination.mkdir(parents=True, exist_ok=True)
    sources = {"baseline.mp4": "baseline/baseline.mp4", "agent_round0.mp4": "agent/round_0/video.mp4",
               "agent_final.mp4": "agent/final.mp4", "comparison.mp4": "comparison.mp4",
               "pipeline_trace.png": "trace/pipeline_trace.png"}
    for name, relative in sources.items():
        shutil.copy2(SOURCE / relative, destination / name)
    encoder = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run([encoder, "-y", "-loglevel", "error", "-i", str(destination / "comparison.mp4"),
                    "-vf", "crop=1280:480:640:0", "-an", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart", str(destination / "repair_comparison.mp4")], check=True)
    subprocess.run([encoder, "-y", "-loglevel", "error", "-i", str(destination / "repair_comparison.mp4"),
                    "-filter_complex", "fps=8,scale=960:-1:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse",
                    "-loop", "0", str(destination / "repair_preview.gif")], check=True)
    summary = read(SOURCE / "run_summary.json")
    summary["checkpoint"] = "vendor/GENMO/inputs/pretrained/gem_smpl.ckpt"
    summary["public_provenance"] = "Selected real-run evidence; machine paths removed; no post-fix inference"
    write(destination / "demo_summary.json", summary)
    write(destination / "motion_specification.json", read(SOURCE / "agent/round_0/motion_specification.json"))
    rounds = [finding_excerpt(read(SOURCE / f"agent/round_{i}/verification_report.json")) for i in range(2)]
    write(destination / "verification.json", rounds)
    diagnosis = read(SOURCE / "diagnosis.json")
    write(destination / "diagnosis.json", {key: diagnosis[key] for key in
          ("diagnosis_id", "source_verification_report_id", "status", "llm_status", "diagnosis_summary", "root_causes", "repair_proposals")})
    trace = read(SOURCE / "trace/trace.json")
    steps = []
    for step in trace["steps"]:
        public = {key: value for key, value in step.items() if key != "artifacts"}
        public["executed_artifacts"] = []
        for handle, artifact in step["artifacts"].items():
            item = {"handle": handle}
            if artifact.get("action"):
                item["planner_decision"] = artifact
            if "motion_spec" in artifact:
                item["compiler_segments"] = [{key: segment.get(key) for key in
                    ("segment_id", "action", "body_parts", "direction", "repetition", "temporal_relation", "simultaneous_with")}
                    for segment in artifact["motion_spec"]["segments"]]
            if "generation_result" in artifact:
                item["generation"] = {"status": artifact["generation_result"]["status"],
                    "candidates": [{"candidate_id": candidate["candidate_id"], "seed": candidate["metadata"]["seed"]}
                                   for candidate in artifact["generation_result"]["candidates"]]}
            if "verification_report" in artifact:
                item["verification"] = finding_excerpt(artifact["verification_report"])
            if "diagnosis_result" in artifact:
                item["diagnosis"] = {key: artifact["diagnosis_result"].get(key) for key in
                                     ("diagnosis_id", "status", "diagnosis_summary", "repair_proposals", "terminal_hint")}
            if "champion_candidate_id" in artifact:
                item["k1_selection"] = artifact["champion_candidate_id"]
            public["executed_artifacts"].append(item)
        steps.append(public)
    write(destination / "trace.json", {"run_id": trace["run_id"], "raw_prompt": trace["raw_prompt"],
          "steps": steps, "final_status": summary["final_status"], "terminal_reason": summary["terminal_reason"],
          "provenance": "Actual EventLog/state export; compact artifact excerpts; generation_round is 1-based"})
    write(destination / "event_log.json", read(SOURCE / "trace/event_log.json"))
    page = sanitize((SOURCE / "trace/pipeline_trace.html").read_text(encoding="utf-8"))
    (destination / "pipeline_trace.html").write_text(page, encoding="utf-8")
    manifest = {path.name: {"bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                for path in destination.iterdir() if path.is_file() and path.name != "manifest.json"}
    write(destination / "manifest.json", manifest)


def overlay(checkout):
    for tree in TREES:
        for source in (ROOT / tree).rglob("*"):
            if not source.is_file() or any(part in OMIT for part in source.relative_to(ROOT).parts):
                continue
            if source.suffix not in SUFFIXES and source.name != "LICENSE":
                continue
            target = checkout / source.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    for source in (ROOT / "assets/demo").iterdir():
        target = checkout / "assets/demo" / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    for name in ("README.md", ".gitignore", ".gitattributes", ".env.example"):
        shutil.copy2(ROOT / name, checkout / name)
    for source in ROOT.glob("requirements*.txt"):
        shutil.copy2(source, checkout / source.name)
    report = checkout / "reports/phase_11_real_e2e.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(sanitize((ROOT / "reports/phase_11_real_e2e.md").read_text(encoding="utf-8")), encoding="utf-8")


def integration_patch():
    destination = ROOT / "integrations/GENMO"
    upstream = ROOT / "outputs/genmo_release_upstream"
    base = upstream / "gem/gem.py"
    current = ROOT / "vendor/GENMO/gem/gem.py"
    result = subprocess.run(["git", "diff", "--no-index", "--", str(base), str(current)], capture_output=True)
    if result.returncode not in (0, 1):
        raise RuntimeError("GENMO_DIFF_FAILED")
    patch = result.stdout.decode("utf-8")
    patch = re.sub(r"^diff --git .*", "diff --git a/gem/gem.py b/gem/gem.py", patch, flags=re.MULTILINE)
    patch = re.sub(r"^--- .*", "--- a/gem/gem.py", patch, flags=re.MULTILINE)
    patch = re.sub(r"^\+\+\+ .*", "+++ b/gem/gem.py", patch, flags=re.MULTILINE)
    (destination / "compatibility.patch").write_text(patch, encoding="utf-8")
    shutil.copy2(ROOT / "vendor/GENMO/LICENSE", destination / "LICENSE")
    shutil.copy2(ROOT / "vendor/GENMO/scripts/download_gvhmr_support.py", destination / "scripts/download_gvhmr_support.py")


def known_secrets():
    values = []
    path = ROOT / ".env"
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            name, separator, value = line.partition("=")
            if separator and re.search(r"KEY|TOKEN|PASSWORD|SECRET", name):
                value = value.strip().strip("\"'")
                if len(value) >= 8:
                    values.append(value.encode())
    return values


def scan(checkout, history=False, staged=False):
    secrets = known_secrets()
    objects = []
    if history:
        for line in git(checkout, "rev-list", "--objects", "--all").splitlines():
            oid, _, name = line.partition(" ")
            if name and git(checkout, "cat-file", "-t", oid).strip() == "blob":
                objects.append((name, git(checkout, "cat-file", "blob", oid, binary=True)))
    elif staged:
        for name in git(checkout, "ls-files", "-z").split("\0"):
            if name:
                objects.append((name, git(checkout, "show", ":" + name, binary=True)))
    else:
        names = git(checkout, "ls-files", "--cached", "--others", "--exclude-standard", "-z").split("\0")
        for name in sorted(set(names)):
            if name and (checkout / name).is_file():
                objects.append((name, (checkout / name).read_bytes()))
    findings = []
    patterns = [rb"sk-(?:proj-)?[A-Za-z0-9_-]{24,}", rb"(?:ghp_|github_pat_|hf_)[A-Za-z0-9_]{25,}",
                rb"-----BEGIN (?:OPENSSH|RSA|EC) PRIVATE KEY-----[\r\n]+[A-Za-z0-9+/=]{20,}",
                rb"(?i)(?:Authorization\s*:\s*Bearer|Bearer)\s+[A-Za-z0-9._-]{25,}"]
    for name, data in objects:
        if Path(name).name.startswith(".env") and Path(name).name != ".env.example":
            findings.append({"file": name, "kind": "FORBIDDEN_ENV_FILE"})
        for secret in secrets:
            position = data.find(secret)
            if position >= 0:
                findings.append({"file": name, "line": data[:position].count(b"\n") + 1, "kind": "ACTUAL_LOCAL_SECRET"})
        for pattern in patterns:
            match = re.search(pattern, data)
            if match:
                findings.append({"file": name, "line": data[:match.start()].count(b"\n") + 1, "kind": "SECRET_PATTERN"})
        if not history and re.search(rb"C:[\\/]+Users[\\/]+Lenovo|\b64\.247\.196\.128\b", data):
            findings.append({"file": name, "kind": "PRIVATE_MACHINE_DETAILS"})
        if Path(name).suffix in {".pt", ".pth", ".npz", ".npy", ".ckpt"} or name.startswith("vendor/"):
            findings.append({"file": name, "kind": "RESTRICTED_MODEL_OR_VENDOR"})
        if len(data) > 20 * 1024 * 1024:
            findings.append({"file": name, "kind": "LARGE_FILE"})
    print(json.dumps({"scope": "history" if history else "index" if staged else "checkout", "scanned_blobs": len(objects),
                      "status": "FAIL" if findings else "PASS", "findings": findings}, indent=2))
    if findings:
        raise SystemExit(1)


def validate(checkout):
    import cv2
    from PIL import Image

    for relative in ("README.md", "docs/phase11_demo.md"):
        path = checkout / relative
        text = path.read_text(encoding="utf-8")
        for link in re.findall(r"\]\(([^)]+)\)", text):
            if "://" not in link and not link.startswith("#"):
                target = path.parent / link.split("#")[0]
                assert target.exists(), (relative, link)
        assert text.count("```") % 2 == 0, relative
    for path in (checkout / "assets/demo").glob("*.mp4"):
        capture = cv2.VideoCapture(str(path))
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = capture.get(cv2.CAP_PROP_FPS)
        assert frames == 420 and fps == 30, path.name
        for index in (0, frames // 2, frames - 1):
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            success, pixels = capture.read()
            assert success and pixels.std() > 5, (path.name, index)
        capture.release()
        print("VIDEO_OK", path.name, frames, fps)
    for name in ("pipeline_trace.png", "repair_preview.gif"):
        with Image.open(checkout / "assets/demo" / name) as image:
            image.verify()
        print("IMAGE_OK", name)
    assert all(not line.partition("=")[2].strip() for line in (checkout / ".env.example").read_text().splitlines() if "=" in line)
    print("LINKS_AND_EMPTY_ENV_EXAMPLE=PASS")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, default=ROOT / "outputs/public_release_checkout")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--history", action="store_true")
    parser.add_argument("--staged", action="store_true")
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        integration_patch()
        package_media()
        overlay(args.checkout)
    scan(args.checkout, history=args.history, staged=args.staged)
    if args.validate:
        validate(args.checkout)


if __name__ == "__main__":
    main()
