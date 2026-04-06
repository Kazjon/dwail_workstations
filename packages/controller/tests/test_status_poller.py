"""
Tests for the controller's background status poller.

The poller periodically fetches /status from each registered agent
and updates the registry. All HTTP calls are mocked.
"""

import asyncio
import pytest
import httpx
import respx

from dwail_shared.models import GPUInfo, VLLMState, WorkstationStatus, RegisteredWorkstation
from dwail_controller import registry
from dwail_controller.status_poller import poll_once, start_poller, POLL_INTERVAL


MOCK_STATUS = {
    "ip": "10.147.18.230",
    "agent_version": "0.1.0",
    "gpu_info": [
        {"index": 0, "name": "RTX 3090", "vram_total_mb": 24576, "vram_free_mb": 20000},
        {"index": 1, "name": "RTX 3090", "vram_total_mb": 24576, "vram_free_mb": 20000},
    ],
    "vllm_state": "idle",
    "current_model": None,
    "ray_running": True,
}

MOCK_STATUS_RUNNING = {**MOCK_STATUS, "vllm_state": "running", "current_model": "facebook/opt-125m"}


@pytest.fixture(autouse=True)
def clean_registry():
    registry.clear()
    yield
    registry.clear()


@pytest.fixture
def ws_registered():
    """A workstation registered with no status (offline at registration time)."""
    return registry.add(ip="10.147.18.230", agent_port=8765, status=None)


# --- poll_once ---

async def test_poll_once_updates_status_on_success(ws_registered):
    with respx.mock:
        respx.get("http://10.147.18.230:8765/status").mock(
            return_value=httpx.Response(200, json=MOCK_STATUS)
        )
        await poll_once()

    ws = registry.get(ws_registered.id)
    assert ws.status is not None
    assert ws.status.vllm_state == VLLMState.idle
    assert len(ws.status.gpu_info) == 2
    assert ws.status.gpu_info[0].vram_free_mb == 20000


async def test_poll_once_sets_status_to_none_on_unreachable(ws_registered):
    # First give it a real status
    registry.update_status(ws_registered.id, WorkstationStatus(**MOCK_STATUS))

    with respx.mock:
        respx.get("http://10.147.18.230:8765/status").mock(
            side_effect=httpx.ConnectError("unreachable")
        )
        await poll_once()

    ws = registry.get(ws_registered.id)
    assert ws.status is None


async def test_poll_once_handles_non_200_response(ws_registered):
    with respx.mock:
        respx.get("http://10.147.18.230:8765/status").mock(
            return_value=httpx.Response(500)
        )
        await poll_once()

    ws = registry.get(ws_registered.id)
    assert ws.status is None


async def test_poll_once_polls_all_registered_workstations():
    registry.add(ip="10.147.18.230", agent_port=8765, status=None)
    registry.add(ip="10.147.18.61", agent_port=8765, status=None)

    status_b = {**MOCK_STATUS, "ip": "10.147.18.61"}

    with respx.mock:
        respx.get("http://10.147.18.230:8765/status").mock(
            return_value=httpx.Response(200, json=MOCK_STATUS)
        )
        respx.get("http://10.147.18.61:8765/status").mock(
            return_value=httpx.Response(200, json=status_b)
        )
        await poll_once()

    for ws in registry.list_workstations():
        assert ws.status is not None


async def test_poll_once_updates_running_state(ws_registered):
    with respx.mock:
        respx.get("http://10.147.18.230:8765/status").mock(
            return_value=httpx.Response(200, json=MOCK_STATUS_RUNNING)
        )
        await poll_once()

    ws = registry.get(ws_registered.id)
    assert ws.status.vllm_state == VLLMState.running
    assert ws.status.current_model == "facebook/opt-125m"


async def test_poll_once_is_no_op_with_no_workstations():
    # Should not raise
    with respx.mock:
        await poll_once()


# --- POLL_INTERVAL ---

def test_poll_interval_is_reasonable():
    assert 5 <= POLL_INTERVAL <= 60


# --- start_poller / stop_poller integration ---

async def test_poller_runs_and_updates_registry(ws_registered):
    """start_poller returns a task; after one cycle the registry is updated."""
    call_count = 0

    async def fake_poll_once():
        nonlocal call_count
        call_count += 1
        registry.update_status(ws_registered.id, WorkstationStatus(**MOCK_STATUS))

    task = await start_poller(poll_fn=fake_poll_once, interval=0.05)

    await asyncio.sleep(0.2)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    assert call_count >= 2
    assert registry.get(ws_registered.id).status is not None


async def test_poller_continues_after_poll_error(ws_registered):
    """A single poll failure does not stop the poller."""
    call_count = 0

    async def flaky_poll():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise Exception("transient error")
        registry.update_status(ws_registered.id, WorkstationStatus(**MOCK_STATUS))

    task = await start_poller(poll_fn=flaky_poll, interval=0.05)
    await asyncio.sleep(0.2)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    assert call_count >= 2
    assert registry.get(ws_registered.id).status is not None
