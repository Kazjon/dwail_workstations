"""
dwail-agent-update — upgrade the agent package and restart the service.

Usage:
    dwail-agent-update

Must be run as root (or with sudo) on the workstation.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

SERVICE_NAME = "dwail-agent"
PACKAGE_URL = "git+https://github.com/Kazjon/dwail_workstations.git"


def main() -> None:
    _require_root()
    _do_update()
    print(f"\n[dwail-update] Done. Agent service restarted.")


def _require_root() -> None:
    if os.geteuid() != 0:
        print("ERROR: this script must be run as root (use sudo).", file=sys.stderr)
        sys.exit(1)


def _find_pip(python: str) -> str | None:
    bin_dir = Path(python).parent
    candidate = bin_dir / "pip"
    if candidate.exists():
        return str(candidate)
    return shutil.which("pip3") or shutil.which("pip")


def _do_update() -> None:
    pip = _find_pip(sys.executable)
    if not pip:
        print("ERROR: 'pip' not found. Cannot upgrade package.", file=sys.stderr)
        sys.exit(1)

    print(f"[dwail-update] Upgrading package from {PACKAGE_URL} ...")
    _run([pip, "install", "--upgrade", PACKAGE_URL])

    print(f"[dwail-update] Restarting {SERVICE_NAME} ...")
    _run(["systemctl", "restart", SERVICE_NAME])


def _run(cmd: list[str]) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR running {' '.join(cmd)}:\n{result.stderr}", file=sys.stderr)
        sys.exit(result.returncode)
