"""
Tests for the POST /models/stop endpoint.

Tells all active workstations to stop vLLM.
Returns 404 if nothing is running or loading.
Returns 200 with a list of workstations that were stopped.
Handles agent unreachability gracefully (still returns 200, notes failure).
"""

import pytest
import respx
import httpx
from httpx import ASGITransport, AsyncClient

from dwail_controller.main import app
from dwail_controller import registry
from dwail_shared.models import GPUInfo, VLLMState, WorkstationStatus


@pytest.fixture(autouse=True)
def clean_registry():
    registry.clear()
    yield
    registry.clear()


@pytest.fixture
async def client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


@pytest.fixture
def ws_running():
    return registry.add(
        ip="10.147.18.230",
        agent_port=8765,
        status=WorkstationStatus(
            ip="10.147.18.230",
            agent_version="0.1.0",
            gpu_info=[],
            vllm_state=VLLMState.running,
            current_model="facebook/opt-125m",
            ray_running=True,
        ),
    )


@pytest.fixture
def ws_loading():
    return registry.add(
        ip="10.147.18.230",
        agent_port=8765,
        status=WorkstationStatus(
            ip="10.147.18.230",
            agent_version="0.1.0",
            gpu_info=[],
            vllm_state=VLLMState.loading,
            current_model="facebook/opt-125m",
            ray_running=True,
        ),
    )


@pytest.fixture
def ws_idle():
    return registry.add(
        ip="10.147.18.230",
        agent_port=8765,
        status=WorkstationStatus(
            ip="10.147.18.230",
            agent_version="0.1.0",
            gpu_info=[],
            vllm_state=VLLMState.idle,
            current_model=None,
            ray_running=True,
        ),
    )


# --- 404 when nothing active ---

async def test_stop_returns_404_with_no_workstations(client):
    r = await client.post("/models/stop")
    assert r.status_code == 404


async def test_stop_returns_404_when_all_idle(client, ws_idle):
    r = await client.post("/models/stop")
    assert r.status_code == 404


# --- 200 when model is active ---

@respx.mock
async def test_stop_returns_200_when_running(client, ws_running):
    respx.post("http://10.147.18.230:8765/vllm/stop").mock(
        return_value=httpx.Response(202)
    )
    r = await client.post("/models/stop")
    assert r.status_code == 200


@respx.mock
async def test_stop_returns_200_when_loading(client, ws_loading):
    respx.post("http://10.147.18.230:8765/vllm/stop").mock(
        return_value=httpx.Response(202)
    )
    r = await client.post("/models/stop")
    assert r.status_code == 200


@respx.mock
async def test_stop_response_lists_stopped_workstation(client, ws_running):
    respx.post("http://10.147.18.230:8765/vllm/stop").mock(
        return_value=httpx.Response(202)
    )
    r = await client.post("/models/stop")
    data = r.json()
    assert "10.147.18.230" in data["stopped"]


@respx.mock
async def test_stop_sends_request_to_agent(client, ws_running):
    route = respx.post("http://10.147.18.230:8765/vllm/stop").mock(
        return_value=httpx.Response(202)
    )
    await client.post("/models/stop")
    assert route.called


# --- distributed: stops all active workstations ---

@respx.mock
async def test_stop_distributed_stops_both_workstations(client):
    registry.add(
        ip="10.147.18.230", agent_port=8765,
        status=WorkstationStatus(
            ip="10.147.18.230", agent_version="0.1.0", gpu_info=[],
            vllm_state=VLLMState.running, current_model="facebook/opt-125m", ray_running=True,
        ),
    )
    registry.add(
        ip="10.147.18.61", agent_port=8765,
        status=WorkstationStatus(
            ip="10.147.18.61", agent_version="0.1.0", gpu_info=[],
            vllm_state=VLLMState.running, current_model="facebook/opt-125m", ray_running=True,
        ),
    )
    respx.post("http://10.147.18.230:8765/vllm/stop").mock(return_value=httpx.Response(202))
    respx.post("http://10.147.18.61:8765/vllm/stop").mock(return_value=httpx.Response(202))

    r = await client.post("/models/stop")
    assert r.status_code == 200
    assert set(r.json()["stopped"]) == {"10.147.18.230", "10.147.18.61"}


# --- agent unreachable: still returns 200, records failure ---

@respx.mock
async def test_stop_tolerates_unreachable_agent(client, ws_running):
    respx.post("http://10.147.18.230:8765/vllm/stop").mock(
        side_effect=httpx.ConnectError("unreachable")
    )
    r = await client.post("/models/stop")
    assert r.status_code == 200
    data = r.json()
    assert "10.147.18.230" in data["failed"]
    assert "10.147.18.230" not in data.get("stopped", [])
