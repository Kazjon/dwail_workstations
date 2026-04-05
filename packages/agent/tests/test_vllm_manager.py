"""
Tests for vllm_manager — focuses on the polling loop state machine.
vLLM subprocess is mocked; httpx calls to /health are mocked.
"""

import time
import threading
from unittest.mock import MagicMock, patch, call

import pytest

from dwail_shared.models import StartVLLMRequest, VLLMState


# Reset module state before each test
@pytest.fixture(autouse=True)
def reset_vllm_manager():
    import dwail_agent.vllm_manager as mgr
    mgr._state = VLLMState.idle
    mgr._current_model = None
    mgr._process = None
    mgr._stop_polling.set()  # stop any lingering poll thread
    yield
    mgr._state = VLLMState.idle
    mgr._current_model = None
    mgr._process = None
    mgr._stop_polling.set()


@pytest.fixture
def small_request():
    return StartVLLMRequest(
        model_id="facebook/opt-125m",
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
    )


# --- Initial state ---

def test_initial_state_is_idle():
    import dwail_agent.vllm_manager as mgr
    assert mgr.get_state() == VLLMState.idle
    assert mgr.get_current_model() is None


# --- start() ---

def test_start_transitions_to_loading(mocker, small_request):
    import dwail_agent.vllm_manager as mgr
    mocker.patch("dwail_agent.vllm_manager.subprocess.Popen", return_value=MagicMock())
    mocker.patch("dwail_agent.vllm_manager._start_poll_thread")

    mgr.start(small_request)

    assert mgr.get_state() == VLLMState.loading
    assert mgr.get_current_model() == "facebook/opt-125m"


def test_start_spawns_correct_vllm_command(mocker, small_request):
    import dwail_agent.vllm_manager as mgr
    mock_popen = mocker.patch("dwail_agent.vllm_manager.subprocess.Popen", return_value=MagicMock())
    mocker.patch("dwail_agent.vllm_manager._start_poll_thread")

    mgr.start(small_request)

    cmd = mock_popen.call_args[0][0]
    assert "--model" in cmd
    assert "facebook/opt-125m" in cmd
    assert "--tensor-parallel-size" in cmd
    assert "1" in cmd


def test_start_includes_ray_backend_for_distributed(mocker):
    import dwail_agent.vllm_manager as mgr
    mock_popen = mocker.patch("dwail_agent.vllm_manager.subprocess.Popen", return_value=MagicMock())
    mocker.patch("dwail_agent.vllm_manager._start_poll_thread")

    req = StartVLLMRequest(
        model_id="meta-llama/Llama-3.1-70B-Instruct",
        tensor_parallel_size=2,
        pipeline_parallel_size=2,
        ray_address="10.147.20.5:6379",
    )
    mgr.start(req)

    cmd = mock_popen.call_args[0][0]
    assert "--distributed-executor-backend" in cmd
    assert "ray" in cmd


def test_start_launches_poll_thread(mocker, small_request):
    import dwail_agent.vllm_manager as mgr
    mocker.patch("dwail_agent.vllm_manager.subprocess.Popen", return_value=MagicMock())
    mock_poll = mocker.patch("dwail_agent.vllm_manager._start_poll_thread")

    mgr.start(small_request)

    mock_poll.assert_called_once()


# --- Polling loop: transitions loading → running ---

def test_poll_transitions_to_running_when_vllm_healthy(mocker, small_request):
    import dwail_agent.vllm_manager as mgr

    mock_proc = MagicMock()
    mock_proc.poll.return_value = None  # process still running
    mocker.patch("dwail_agent.vllm_manager.subprocess.Popen", return_value=mock_proc)

    # vLLM health endpoint responds 200 immediately
    mock_get = mocker.patch("dwail_agent.vllm_manager.httpx.get")
    mock_get.return_value = MagicMock(status_code=200)

    mgr.start(small_request)

    # Give the poll thread time to run one cycle
    deadline = time.time() + 5
    while time.time() < deadline:
        if mgr.get_state() == VLLMState.running:
            break
        time.sleep(0.05)

    assert mgr.get_state() == VLLMState.running


def test_poll_stays_loading_while_vllm_not_yet_healthy(mocker, small_request):
    import dwail_agent.vllm_manager as mgr

    mock_proc = MagicMock()
    mock_proc.poll.return_value = None
    mocker.patch("dwail_agent.vllm_manager.subprocess.Popen", return_value=mock_proc)

    # vLLM health always returns 503 (still loading)
    mock_get = mocker.patch("dwail_agent.vllm_manager.httpx.get")
    mock_get.return_value = MagicMock(status_code=503)

    mgr.start(small_request)
    time.sleep(0.2)

    assert mgr.get_state() == VLLMState.loading


def test_poll_transitions_to_error_when_process_exits(mocker, small_request):
    import dwail_agent.vllm_manager as mgr

    mock_proc = MagicMock()
    mock_proc.poll.return_value = 1  # process exited with error
    mocker.patch("dwail_agent.vllm_manager.subprocess.Popen", return_value=mock_proc)

    mock_get = mocker.patch("dwail_agent.vllm_manager.httpx.get",
                            side_effect=Exception("connection refused"))

    mgr.start(small_request)

    deadline = time.time() + 5
    while time.time() < deadline:
        if mgr.get_state() == VLLMState.error:
            break
        time.sleep(0.05)

    assert mgr.get_state() == VLLMState.error


def test_poll_transitions_to_error_on_connection_refused_after_timeout(mocker, small_request):
    """If vLLM never becomes healthy within the timeout, state → error."""
    import dwail_agent.vllm_manager as mgr

    mock_proc = MagicMock()
    mock_proc.poll.return_value = None
    mocker.patch("dwail_agent.vllm_manager.subprocess.Popen", return_value=mock_proc)
    mocker.patch("dwail_agent.vllm_manager.httpx.get",
                 side_effect=Exception("connection refused"))
    # Use a very short timeout so the test doesn't take minutes
    mocker.patch("dwail_agent.vllm_manager.VLLM_READY_TIMEOUT", 0.2)

    mgr.start(small_request)

    deadline = time.time() + 5
    while time.time() < deadline:
        if mgr.get_state() == VLLMState.error:
            break
        time.sleep(0.05)

    assert mgr.get_state() == VLLMState.error


# --- stop() ---

def test_stop_terminates_process_and_resets_state(mocker, small_request):
    import dwail_agent.vllm_manager as mgr

    mock_proc = MagicMock()
    mock_proc.poll.return_value = None
    mocker.patch("dwail_agent.vllm_manager.subprocess.Popen", return_value=mock_proc)
    mocker.patch("dwail_agent.vllm_manager.httpx.get",
                 return_value=MagicMock(status_code=200))

    mgr.start(small_request)
    # Wait until running
    deadline = time.time() + 5
    while mgr.get_state() != VLLMState.running and time.time() < deadline:
        time.sleep(0.05)

    mgr.stop()

    assert mgr.get_state() == VLLMState.idle
    assert mgr.get_current_model() is None
    mock_proc.terminate.assert_called_once()


def test_stop_halts_poll_thread(mocker, small_request):
    import dwail_agent.vllm_manager as mgr

    mock_proc = MagicMock()
    mock_proc.poll.return_value = None
    mocker.patch("dwail_agent.vllm_manager.subprocess.Popen", return_value=mock_proc)
    mocker.patch("dwail_agent.vllm_manager.httpx.get",
                 return_value=MagicMock(status_code=200))

    mgr.start(small_request)
    deadline = time.time() + 5
    while mgr.get_state() != VLLMState.running and time.time() < deadline:
        time.sleep(0.05)

    mgr.stop()

    # Poll thread should have exited within a short window
    time.sleep(0.3)
    assert mgr._stop_polling.is_set()
