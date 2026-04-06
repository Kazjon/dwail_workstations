"""
Level 2 tests — one real workstation running the dwail-agent.

Requires:
  DWAIL_WS1_IP=<zerotier-ip>  (agent must be running on port 8765)

Model defaults to facebook/opt-125m (always loaded fresh).
Override with DWAIL_TEST_MODEL=<hf-id> to use a larger pre-downloaded model.

Run with:
  DWAIL_WS1_IP=10.x.x.x uv run pytest -m workstation1 -v -s
"""

import asyncio
import time

import httpx
import pytest
from httpx import ASGITransport, AsyncClient

from dwail_controller.main import app as controller_app
from dwail_controller import registry
from dwail_controller.status_poller import poll_once

AGENT_PORT = 8765
VLLM_READY_TIMEOUT = 600
POLL_INTERVAL = 5

# opt-125m is GPT-2 family — no chat template, must use /v1/completions
CHAT_MODELS = ()  # add model id prefixes here as chat models are tested


def _is_chat_model(model_id: str) -> bool:
    return any(model_id.startswith(p) for p in CHAT_MODELS)


@pytest.fixture(scope="module")
async def ws1_agent(ws1_ip):
    """Verify WS1 agent is reachable, then stop any running vLLM before tests."""
    url = f"http://{ws1_ip}:{AGENT_PORT}"
    async with httpx.AsyncClient(timeout=5.0) as client:
        try:
            r = await client.get(f"{url}/health")
            assert r.status_code == 200, f"Agent at {url} not healthy: {r.status_code}"
        except Exception as e:
            pytest.fail(f"Cannot reach agent at {url}: {e}")

        status = await client.get(f"{url}/status")
        if status.json()["vllm_state"] != "idle":
            async with httpx.AsyncClient(timeout=60.0) as stop_client:
                await stop_client.post(f"{url}/vllm/stop")
            await asyncio.sleep(3)

    return url


@pytest.fixture
async def controller_client():
    """In-process controller client — no need to run dwail-controller separately."""
    registry.clear()
    async with AsyncClient(transport=ASGITransport(app=controller_app), base_url="http://test") as c:
        yield c
    registry.clear()


@pytest.mark.workstation1
async def test_ws1_agent_health(ws1_agent):
    async with httpx.AsyncClient(timeout=5.0) as client:
        r = await client.get(f"{ws1_agent}/health")
    assert r.status_code == 200


@pytest.mark.workstation1
async def test_ws1_status_has_gpus(ws1_agent):
    async with httpx.AsyncClient(timeout=5.0) as client:
        r = await client.get(f"{ws1_agent}/status")
    data = r.json()
    assert len(data["gpu_info"]) >= 1, "Expected at least one GPU"
    for gpu in data["gpu_info"]:
        assert gpu["vram_total_mb"] > 0


@pytest.mark.workstation1
async def test_ws1_ray_running(ws1_agent):
    async with httpx.AsyncClient(timeout=5.0) as client:
        r = await client.get(f"{ws1_agent}/status")
    assert r.json()["ray_running"] is True, "Ray should be running on the workstation"


@pytest.mark.workstation1
async def test_ws1_model_list(ws1_agent):
    async with httpx.AsyncClient(timeout=5.0) as client:
        r = await client.get(f"{ws1_agent}/models")
    assert r.status_code == 200
    assert isinstance(r.json(), list)


@pytest.mark.workstation1
async def test_ws1_load_and_infer(ws1_agent, test_model):
    """
    Load the test model on WS1, wait for ready, run an inference, then stop.
    Uses /v1/chat/completions for chat models, /v1/completions for base models.
    """
    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.post(f"{ws1_agent}/vllm/start", json={
            "model_id": test_model,
            "tensor_parallel_size": 2,
            "pipeline_parallel_size": 1,
        })
        assert r.status_code == 202, r.text

        # Poll until running
        deadline = time.time() + VLLM_READY_TIMEOUT
        while time.time() < deadline:
            s = await client.get(f"{ws1_agent}/status")
            state = s.json()["vllm_state"]
            if state == "running":
                break
            if state == "error":
                pytest.fail(f"vLLM error: {s.json()}")
            await asyncio.sleep(POLL_INTERVAL)
        else:
            pytest.fail(f"vLLM did not reach 'running' in {VLLM_READY_TIMEOUT}s")

        vllm_url = ws1_agent.replace(str(AGENT_PORT), "8000")

        if _is_chat_model(test_model):
            infer_r = await client.post(f"{vllm_url}/v1/chat/completions", json={
                "model": test_model,
                "messages": [{"role": "user", "content": "Say hello."}],
                "max_tokens": 20,
            })
            assert infer_r.status_code == 200, infer_r.text
            assert len(infer_r.json()["choices"]) > 0
        else:
            infer_r = await client.post(f"{vllm_url}/v1/completions", json={
                "model": test_model,
                "prompt": "Hello",
                "max_tokens": 20,
            })
            assert infer_r.status_code == 200, infer_r.text
            assert len(infer_r.json()["choices"]) > 0

        stop_r = await client.post(f"{ws1_agent}/vllm/stop")
        assert stop_r.status_code == 202


@pytest.mark.workstation1
async def test_controller_current_model(ws1_ip, ws1_agent, test_model, controller_client):
    """
    Load a model via the controller, wait for vLLM to be running,
    then verify GET /models/current returns correct data.
    """
    add_r = await controller_client.post("/workstations", json={"ip": ws1_ip})
    assert add_r.status_code == 201
    ws_id = add_r.json()["id"]

    try:
        load_r = await controller_client.post("/models/load", json={"model_id": test_model})
        assert load_r.status_code == 202

        # Sync registry with the agent's updated state (poller not running in-process)
        await poll_once()

        # /models/current should show loading or running state
        current_r = await controller_client.get("/models/current")
        assert current_r.status_code == 200
        data = current_r.json()
        assert data["model_id"] == test_model
        assert data["vllm_state"] in ("loading", "running")
        assert data["endpoint"] == f"http://{ws1_ip}:8000/v1"
        assert ws1_ip in data["workstations"]
        assert data["mode"] == "single"
        assert "supports_chat" in data
        assert data["capability_confidence"] in ("high", "low")

        # Wait for fully running, then re-poll and re-check
        async with httpx.AsyncClient(timeout=30.0) as client:
            deadline = time.time() + VLLM_READY_TIMEOUT
            while time.time() < deadline:
                s = await client.get(f"{ws1_agent}/status")
                state = s.json()["vllm_state"]
                if state == "running":
                    break
                if state == "error":
                    pytest.fail(f"vLLM error: {s.json()}")
                await asyncio.sleep(POLL_INTERVAL)
            else:
                pytest.fail(f"vLLM did not reach 'running' in {VLLM_READY_TIMEOUT}s")

        await poll_once()  # sync registry now that vLLM is running
        current_r2 = await controller_client.get("/models/current")
        assert current_r2.status_code == 200
        assert current_r2.json()["vllm_state"] == "running"
        # opt-125m has no chat template — should report False
        assert current_r2.json()["supports_chat"] is False

    finally:
        # Always clean up — wait for vLLM to fully stop before the next test
        async with httpx.AsyncClient(timeout=60.0) as client:
            await client.post(f"{ws1_agent}/vllm/stop")
            deadline = time.time() + 60
            while time.time() < deadline:
                s = await client.get(f"{ws1_agent}/status")
                if s.json()["vllm_state"] == "idle":
                    break
                await asyncio.sleep(2)
        await controller_client.delete(f"/workstations/{ws_id}")


@pytest.mark.workstation1
async def test_controller_add_ws1_and_load(ws1_ip, test_model, controller_client):
    """
    Full round-trip through the controller: register WS1, load model, verify response.
    Uses an in-process controller — no need to start dwail-controller separately.
    """
    add_r = await controller_client.post("/workstations", json={"ip": ws1_ip})
    assert add_r.status_code == 201
    ws_id = add_r.json()["id"]

    load_r = await controller_client.post("/models/load", json={"model_id": test_model})
    assert load_r.status_code == 202
    assert load_r.json()["mode"] == "single"

    # Clean up
    async with httpx.AsyncClient(timeout=10.0) as client:
        await client.post(f"http://{ws1_ip}:{AGENT_PORT}/vllm/stop")
    await controller_client.delete(f"/workstations/{ws_id}")
