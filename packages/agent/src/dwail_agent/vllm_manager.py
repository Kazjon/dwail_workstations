"""Manage the vLLM server process on this workstation."""

from __future__ import annotations

import subprocess
import threading
import time
from threading import Event, Lock

import httpx

from dwail_shared.models import StartVLLMRequest, VLLMState

VLLM_PORT = 8000
VLLM_HEALTH_URL = f"http://127.0.0.1:{VLLM_PORT}/health"
VLLM_READY_TIMEOUT = 600  # seconds to wait for vLLM to become healthy
POLL_INTERVAL = 2          # seconds between health checks

_state: VLLMState = VLLMState.idle
_current_model: str | None = None
_process: subprocess.Popen | None = None
_lock = Lock()
_stop_polling: Event = Event()


def get_state() -> VLLMState:
    return _state


def get_current_model() -> str | None:
    return _current_model


def start(request: StartVLLMRequest) -> None:
    """Launch vLLM as a subprocess and begin polling for readiness.
    Non-blocking — state immediately transitions to 'loading'."""
    global _state, _current_model, _process

    with _lock:
        cmd = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", request.model_id,
            "--tensor-parallel-size", str(request.tensor_parallel_size),
            "--pipeline-parallel-size", str(request.pipeline_parallel_size),
        ]
        if request.ray_address:
            cmd += ["--distributed-executor-backend", "ray"]

        _process = subprocess.Popen(cmd)
        _state = VLLMState.loading
        _current_model = request.model_id

    _start_poll_thread()


def stop() -> None:
    """Terminate the vLLM process and stop the polling thread."""
    global _state, _current_model, _process

    _stop_polling.set()

    with _lock:
        if _process is not None:
            _process.terminate()
            _process.wait()
            _process = None
        _state = VLLMState.idle
        _current_model = None


def _start_poll_thread() -> None:
    """Start a background thread that polls vLLM's /health endpoint."""
    _stop_polling.clear()
    t = threading.Thread(target=_poll_until_ready, daemon=True)
    t.start()


def _poll_until_ready() -> None:
    """Poll vLLM /health until running, process exits, or timeout."""
    global _state

    deadline = time.monotonic() + VLLM_READY_TIMEOUT

    while not _stop_polling.is_set():
        # Check if the process has exited unexpectedly
        if _process is not None and _process.poll() is not None:
            with _lock:
                _state = VLLMState.error
            return

        # Check if we've exceeded the startup timeout
        if time.monotonic() > deadline:
            with _lock:
                _state = VLLMState.error
            return

        # Poll vLLM's health endpoint
        try:
            r = httpx.get(VLLM_HEALTH_URL, timeout=2.0)
            if r.status_code == 200:
                with _lock:
                    _state = VLLMState.running
                return
        except Exception:
            pass  # vLLM not up yet — keep polling

        _stop_polling.wait(timeout=POLL_INTERVAL)
