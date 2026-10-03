"""Transfer only authorized API settings over SSH stdin, without logging values."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    from motion_agent.llm.config import load_dotenv

    host = os.environ.get("MOTIONAGENT_SSH_HOST")
    if not host:
        print("MOTIONAGENT_SSH_HOST=MISSING")
        return 1
    port = os.environ.get("MOTIONAGENT_SSH_PORT", "22")

    names = ("OPENAI_API_KEY", "OPENAI_MODEL", "OPENAI_BASE_URL",
             "OPENAI_TIMEOUT_S", "OPENAI_MAX_RETRIES",
             "MLLM_API_KEY", "MLLM_MODEL", "MLLM_BASE_URL")
    for name in names:
        os.environ.pop(name, None)
    load_dotenv(ROOT / ".env")
    source = os.environ
    settings = {name: source[name] for name in names if source.get(name)}
    for name in ("OPENAI_API_KEY", "OPENAI_MODEL"):
        print(name + "=" + ("FOUND" if settings.get(name) else "MISSING"))
        if not settings.get(name):
            return 1
    if any("\n" in value or "\r" in value or "\x00" in value for value in settings.values()):
        print("CONFIG_MULTILINE_VALUE_REJECTED")
        return 1
    settings.setdefault("MLLM_API_KEY", settings["OPENAI_API_KEY"])
    settings.setdefault("MLLM_MODEL", settings["OPENAI_MODEL"])
    if settings.get("OPENAI_BASE_URL"):
        settings.setdefault("MLLM_BASE_URL", settings["OPENAI_BASE_URL"])
    receiver = """import json, os, pathlib, sys
settings = json.load(sys.stdin)
project_path = pathlib.Path('/workspace/MotionAgent/.env')
path = pathlib.Path.home() / '.config/motionagent/api.env'
path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
os.chmod(path.parent, 0o700)
allowed = {'OPENAI_API_KEY', 'OPENAI_MODEL', 'OPENAI_BASE_URL', 'OPENAI_TIMEOUT_S', 'OPENAI_MAX_RETRIES', 'MLLM_API_KEY', 'MLLM_MODEL', 'MLLM_BASE_URL'}
assert set(settings) <= allowed
assert all(isinstance(v, str) and not any(c in v for c in '\\n\\r\\x00') for v in settings.values())
previous = {}
if project_path.exists():
    for line in project_path.read_text().splitlines():
        if '=' in line and not line.lstrip().startswith('#'):
            key, value = line.split('=', 1)
            previous[key.strip()] = value
previous.update({k: json.dumps(v) for k, v in settings.items()})
fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
os.fchmod(fd, 0o600)
with os.fdopen(fd, 'w') as stream:
    stream.write(''.join(k + '=' + v + '\\n' for k, v in previous.items()))
assert path.stat().st_mode & 0o777 == 0o600
if not project_path.is_symlink():
    project_path.unlink(missing_ok=True)
    project_path.symlink_to(path)
assert project_path.resolve() == path
print('REMOTE_API_SETTINGS_INSTALLED_MODE_600')
print('OPENAI_API_KEY=FOUND')
print('OPENAI_MODEL=FOUND')
"""
    import shlex

    command = ["ssh", host, "-p", port, "-i",
               str(Path.home() / ".ssh/id_ed25519"), "-o", "BatchMode=yes",
               "-o", "ConnectTimeout=15", "python -c " + shlex.quote(receiver)]
    result = subprocess.run(command, input=json.dumps(settings), text=True,
                            capture_output=True, timeout=45)
    if result.returncode:
        print("REMOTE_API_SETTINGS_INSTALL_FAILED_EXIT_" + str(result.returncode))
        return 1
    allowed_output = {"REMOTE_API_SETTINGS_INSTALLED_MODE_600", "OPENAI_API_KEY=FOUND", "OPENAI_MODEL=FOUND"}
    for line in result.stdout.splitlines():
        if line in allowed_output:
            print(line)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print("SETTINGS_TRANSFER_ERROR=" + type(exc).__name__)
        raise SystemExit(1) from None
