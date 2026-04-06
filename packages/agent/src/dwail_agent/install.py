"""
dwail-agent-install — set up the agent as a systemd service and start Ray.

Usage:
    dwail-agent-install [--model-dir /mnt/models] [--port 8765] [--ray-head] [--ray-worker HEAD_IP]

Must be run as root (or with sudo) on the workstation.
Safe to re-run (idempotent).
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path


SERVICE_NAME = "dwail-agent"
SERVICE_FILE = Path(f"/etc/systemd/system/{SERVICE_NAME}.service")
DEFAULT_MODEL_DIR = "/mnt/models"
DEFAULT_PORT = 8765


def main() -> None:
    parser = argparse.ArgumentParser(description="Install dwail-agent as a systemd service.")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR,
                        help=f"Directory where models are stored (default: {DEFAULT_MODEL_DIR})")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help=f"Agent listen port (default: {DEFAULT_PORT})")
    parser.add_argument("--ray-head", action="store_true",
                        help="Start Ray as a head node on this machine")
    parser.add_argument("--ray-worker", metavar="HEAD_IP",
                        help="Start Ray as a worker connecting to HEAD_IP:6379")
    parser.add_argument("--uninstall", action="store_true",
                        help="Stop and remove the systemd service")
    args = parser.parse_args()

    _require_root()

    if args.uninstall:
        _uninstall()
        return

    if args.ray_head and args.ray_worker:
        print("ERROR: --ray-head and --ray-worker are mutually exclusive.", file=sys.stderr)
        sys.exit(1)

    python = _find_python()
    agent_bin = _find_agent_bin(python)
    model_dir = Path(args.model_dir)

    print(f"[dwail-install] Python:    {python}")
    print(f"[dwail-install] Agent bin: {agent_bin}")
    print(f"[dwail-install] Model dir: {model_dir}")
    print(f"[dwail-install] Port:      {args.port}")

    _ensure_model_dir(model_dir)
    _write_service(python, agent_bin, model_dir, args.port)
    _enable_and_start_service()
    _setup_ray(args.ray_head, args.ray_worker)

    print("\n[dwail-install] Done.")
    print(f"  Agent running at http://0.0.0.0:{args.port}")
    print(f"  Check status: systemctl status {SERVICE_NAME}")
    print(f"  View logs:    journalctl -u {SERVICE_NAME} -f")


def _require_root() -> None:
    if os.geteuid() != 0:
        print("ERROR: this script must be run as root (use sudo).", file=sys.stderr)
        sys.exit(1)


def _find_python() -> str:
    # Prefer the Python running this script
    return sys.executable


def _find_agent_bin(python: str) -> str:
    # Look for dwail-agent in the same bin dir as python
    bin_dir = Path(python).parent
    candidate = bin_dir / "dwail-agent"
    if candidate.exists():
        return str(candidate)
    # Fall back to which
    found = shutil.which("dwail-agent")
    if found:
        return found
    # Last resort: run as module
    return f"{python} -m dwail_agent.main"


def _ensure_model_dir(model_dir: Path) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    print(f"[dwail-install] Model dir ensured: {model_dir}")


def _write_service(python: str, agent_bin: str, model_dir: Path, port: int) -> None:
    venv_bin = Path(python).parent
    unit = textwrap.dedent(f"""\
        [Unit]
        Description=dwail workstation agent
        After=network.target

        [Service]
        Type=simple
        ExecStart={agent_bin}
        Environment=PATH={venv_bin}:/usr/local/bin:/usr/bin:/bin
        Environment=DWAIL_MODEL_DIR={model_dir}
        Environment=DWAIL_PORT={port}
        Restart=on-failure
        RestartSec=5

        [Install]
        WantedBy=multi-user.target
    """)
    SERVICE_FILE.write_text(unit)
    print(f"[dwail-install] Wrote {SERVICE_FILE}")


def _enable_and_start_service() -> None:
    _run(["systemctl", "daemon-reload"])
    _run(["systemctl", "enable", SERVICE_NAME])
    # Restart rather than start so re-runs pick up config changes
    _run(["systemctl", "restart", SERVICE_NAME])
    print(f"[dwail-install] Service {SERVICE_NAME} enabled and started.")


def _setup_ray(as_head: bool, worker_head_ip: str | None) -> None:
    if not as_head and not worker_head_ip:
        # Default: start as head node
        as_head = True

    # Look for ray in the same bin dir as our Python first (venv-aware, sudo-safe)
    bin_dir = Path(sys.executable).parent
    ray = str(bin_dir / "ray") if (bin_dir / "ray").exists() else shutil.which("ray")
    if not ray:
        print("[dwail-install] WARNING: 'ray' not found in PATH. Skipping Ray setup.")
        print("                Install with: pip install ray")
        return

    # Stop any existing Ray instance first
    subprocess.run([ray, "stop"], capture_output=True)

    if as_head:
        _run([ray, "start", "--head", "--port=6379"])
        print("[dwail-install] Ray started as head node on port 6379.")
    else:
        _run([ray, "start", f"--address={worker_head_ip}:6379"])
        print(f"[dwail-install] Ray started as worker, connected to {worker_head_ip}:6379.")


def _uninstall() -> None:
    print(f"[dwail-install] Uninstalling {SERVICE_NAME}...")
    subprocess.run(["systemctl", "stop", SERVICE_NAME], capture_output=True)
    subprocess.run(["systemctl", "disable", SERVICE_NAME], capture_output=True)
    if SERVICE_FILE.exists():
        SERVICE_FILE.unlink()
        print(f"[dwail-install] Removed {SERVICE_FILE}")
    _run(["systemctl", "daemon-reload"])
    ray = shutil.which("ray")
    if ray:
        subprocess.run([ray, "stop"], capture_output=True)
    print(f"[dwail-install] {SERVICE_NAME} uninstalled.")


def _run(cmd: list[str]) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR running {' '.join(cmd)}:\n{result.stderr}", file=sys.stderr)
        sys.exit(result.returncode)
