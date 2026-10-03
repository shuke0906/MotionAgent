"""Package the current runtime and already downloaded learned verifier assets."""

from pathlib import Path
import tarfile


ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "outputs/phase11_setup/code_and_learned_assets.tar.gz"
TREES = ["motion_agent", "configs", "scripts", "planner_skills", "doc",
         "tests/unit", "tests/contracts", "tests/integration", "tests/gpu", "tests/e2e",
         "vendor/TMR/src", "vendor/TMR/stats", "vendor/TMR/configs",
         "vendor/TMR/models/tmr_humanml3d_guoh3dfeats",
         "vendor/MotionCritic/MotionCritic/lib",
         "outputs/temporal_compiler_validation_final/p00_s7/fixed/candidates/cand_bddf0f0eb4785474"]
FILES = ["requirements.txt", "requirements-phase1.txt", "requirements-phase9b.txt",
         "vendor/MotionCritic/MotionCritic/pretrained/motioncritic_pre.pth",
         "vendor/MotionCritic/MotionCritic/visexample.pth",
         "outputs/temporal_compiler_validation_final/p00_s7/fixed/generation_request.json"]
EXCLUDED = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", ".cache", "cache"}


def main():
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(DESTINATION, "w:gz") as archive:
        paths = [ROOT / path for path in FILES]
        for tree in TREES:
            folder = ROOT / tree
            if folder.is_dir():
                paths.extend(folder.rglob("*"))
        count = 0
        for path in sorted(set(paths)):
            relative = path.relative_to(ROOT)
            if (not path.is_file() or any(part in EXCLUDED for part in relative.parts)
                    or path.name.startswith(".env") or path.suffix in {".pyc", ".pyo"}):
                continue
            archive.add(path, arcname=str(relative).replace("\\", "/"), recursive=False)
            count += 1
    print("RUNTIME_AND_LEARNED_ASSETS_FILES=" + str(count))
    print("ARCHIVE_BYTES=" + str(DESTINATION.stat().st_size))
    print("ENV_FILES_INCLUDED=0")


if __name__ == "__main__":
    main()
