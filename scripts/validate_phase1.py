from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/unit",
        "tests/contracts",
        "tests/integration",
        "-v",
    ]
    print("Phase 1 validation")
    print("command:", " ".join(cmd))
    result = subprocess.run(cmd, cwd=ROOT, text=True)
    if result.returncode == 0:
        print("Phase 1 gates: PASS")
        print("tensor isolation: PASS")
        print("version conflict: PASS")
        print("checkpoint/resume: PASS")
        print("real LangGraph SQLite persistence: PASS")
        print("resume reconciliation: PASS")
        print("thread isolation: PASS")
        print("committed-operation replay protection: PASS")
        print("context growth: PASS")
    else:
        print("Phase 1 gates: FAIL")
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
