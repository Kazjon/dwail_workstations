"""Manage the Ray cluster on this workstation."""

from __future__ import annotations
import subprocess


def is_running() -> bool:
    """Check whether Ray is currently running."""
    result = subprocess.run(["ray", "status"], capture_output=True)
    return result.returncode == 0


def start_head(port: int = 6379) -> None:
    """Start this node as a Ray head node."""
    subprocess.run(["ray", "start", "--head", f"--port={port}"], check=True)


def start_worker(head_address: str) -> None:
    """Connect this node to an existing Ray cluster."""
    subprocess.run(["ray", "start", f"--address={head_address}"], check=True)


def stop() -> None:
    subprocess.run(["ray", "stop"], check=True)
