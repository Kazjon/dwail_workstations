"""
Tests for the /models/current endpoint.

Returns the currently loaded model, its endpoint URL, capability,
and which workstation(s) it's running on. Returns 404 if no model loaded.
"""

import pytest
import httpx
import respx
from httpx import ASGITransport, AsyncClient

from dwail_controller.main import app
from dwail_controller import registry
from dwail_shared.models import GPUInfo, VLLMState, WorkstationStatus, RegisteredWorkstation


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
    """A workstation with vLLM running opt-125m."""
    return registry.add(
        ip="10.147.18.230",
        agent_port=8765,
        status=WorkstationStatus(
            ip="10.147.18.230",
            agent_version="0.1.0",
            gpu_info=[
                GPUInfo(index=0, name="RTX 3090", vram_total_mb=24576, vram_free_mb=10000),
                GPUInfo(index=1, name="RTX 3090", vram_total_mb=24576, vram_free_mb=10000),
            ],
            vllm_state=VLLMState.running,
            current_model="facebook/opt-125m",
            ray_running=True,
        ),
    )


@pytest.fixture
def ws_idle():
    """A workstation with vLLM idle."""
    return registry.add(
        ip="10.147.18.230",
        agent_port=8765,
        status=WorkstationStatus(
            ip="10.147.18.230",
            agent_version="0.1.0",
            gpu_info=[
                GPUInfo(index=0, name="RTX 3090", vram_total_mb=24576, vram_free_mb=24000),
                GPUInfo(index=1, name="RTX 3090", vram_total_mb=24576, vram_free_mb=24000),
            ],
            vllm_state=VLLMState.idle,
            current_model=None,
            ray_running=True,
        ),
    )


# --- GET /models/current ---

async def test_current_model_returns_404_when_none_loaded(client, ws_idle):
    r = await client.get("/models/current")
    assert r.status_code == 404


async def test_current_model_returns_404_with_no_workstations(client):
    r = await client.get("/models/current")
    assert r.status_code == 404


async def test_current_model_returns_model_info(client, ws_running, mocker):
    mocker.patch("dwail_controller.model_capability.hf_hub_download",
                 side_effect=Exception("offline"))

    r = await client.get("/models/current")
    assert r.status_code == 200

    data = r.json()
    assert data["model_id"] == "facebook/opt-125m"
    assert data["endpoint"] == "http://10.147.18.230:8000/v1"
    assert data["vllm_state"] == "running"
    assert "supports_chat" in data
    assert "workstations" in data
    assert "10.147.18.230" in data["workstations"]


async def test_current_model_includes_capability(client, ws_running, mocker):
    mocker.patch("dwail_controller.model_capability.hf_hub_download",
                 side_effect=Exception("offline"))

    r = await client.get("/models/current")
    data = r.json()
    # opt-125m has no chat template — heuristic should say False
    assert data["supports_chat"] is False
    assert data["capability_confidence"] in ("high", "low")


async def test_current_model_loading_state(client):
    """A workstation in 'loading' state should be reported."""
    registry.add(
        ip="10.147.18.230",
        agent_port=8765,
        status=WorkstationStatus(
            ip="10.147.18.230",
            agent_version="0.1.0",
            gpu_info=[],
            vllm_state=VLLMState.loading,
            current_model="meta-llama/Llama-3.1-8B-Instruct",
            ray_running=True,
        ),
    )
    r = await client.get("/models/current")
    assert r.status_code == 200
    assert r.json()["vllm_state"] == "loading"
    assert r.json()["model_id"] == "meta-llama/Llama-3.1-8B-Instruct"


async def test_current_model_error_state_returns_200(client):
    """A workstation in 'error' state should be reported so the UI can stop polling."""
    registry.add(
        ip="10.147.18.230",
        agent_port=8765,
        status=WorkstationStatus(
            ip="10.147.18.230",
            agent_version="0.1.0",
            gpu_info=[],
            vllm_state=VLLMState.error,
            current_model="google/gemma-3-27b-it",
            ray_running=True,
        ),
    )
    r = await client.get("/models/current")
    assert r.status_code == 200
    data = r.json()
    assert data["vllm_state"] == "error"
    assert data["model_id"] == "google/gemma-3-27b-it"
    assert data["supports_chat"] is None


async def test_current_model_distributed_lists_both_workstations(client, mocker):
    """Two workstations running the same model → both appear in workstations list."""
    mocker.patch("dwail_controller.model_capability.hf_hub_download",
                 side_effect=Exception("offline"))

    running_status_a = WorkstationStatus(
        ip="10.147.18.230", agent_version="0.1.0", gpu_info=[],
        vllm_state=VLLMState.running, current_model="facebook/opt-125m", ray_running=True,
    )
    running_status_b = WorkstationStatus(
        ip="10.147.18.61", agent_version="0.1.0", gpu_info=[],
        vllm_state=VLLMState.running, current_model="facebook/opt-125m", ray_running=True,
    )
    registry.add(ip="10.147.18.230", agent_port=8765, status=running_status_a)
    registry.add(ip="10.147.18.61", agent_port=8765, status=running_status_b)

    r = await client.get("/models/current")
    assert r.status_code == 200
    data = r.json()
    assert set(data["workstations"]) == {"10.147.18.230", "10.147.18.61"}
    assert data["mode"] == "distributed"
