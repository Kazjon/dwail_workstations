"""
Level 2 tests — one real workstation running the dwail-agent.

Requires:
  DWAIL_WS1_IP=<zerotier-ip>  (agent must be running on port 8765)

Model defaults to facebook/opt-125m (always loaded fresh).
Override with DWAIL_TEST_MODEL=<hf-id> to use a larger pre-downloaded model.

Run with:
  DWAIL_WS1_IP=10.x.x.x uv run pytest -m workstation1
"""

import asyncio
import time

import httpx
import pytest

AGENT_PORT = 8765
VLLM_READY_TIMEOUT = 600
POLL_INTERVAL = 5
CONTROLLER_URL = "http://127.0.0.1:8080"


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

        # Stop any currently running vLLM so tests start from known state
        status = await client.get(f"{url}/status")
        if status.json()["vllm_state"] != "idle":
            await client.post(f"{url}/vllm/stop")
            await asyncio.sleep(3)

    return url


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
    Uses test_model (default: opt-125m, override with DWAIL_TEST_MODEL).
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

        # Infer via vLLM's OpenAI endpoint
        ws1_vllm_url = ws1_agent.replace(str(AGENT_PORT), "8000")
        infer_r = await client.post(f"{ws1_vllm_url}/v1/chat/completions", json={
            "model": test_model,
            "messages": [{"role": "user", "content": "Say hello."}],
            "max_tokens": 20,
        })
        assert infer_r.status_code == 200
        assert len(infer_r.json()["choices"]) > 0

        # Stop
        stop_r = await client.post(f"{ws1_agent}/vllm/stop")
        assert stop_r.status_code == 202


@pytest.mark.workstation1
async def test_controller_add_ws1_and_load(ws1_ip, test_model):
    """
    Full round-trip through the controller: register WS1, load model, verify endpoint.
    Requires the controller to be running locally (uv run dwail-controller).
    """
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Clean slate — remove WS1 if already registered
        wss = (await client.get(f"{CONTROLLER_URL}/workstations")).json()
        for ws in wss:
            if ws["ip"] == ws1_ip:
                await client.delete(f"{CONTROLLER_URL}/workstations/{ws['id']}")

        # Register
        add_r = await client.post(f"{CONTROLLER_URL}/workstations", json={"ip": ws1_ip})
        assert add_r.status_code == 201
        ws_id = add_r.json()["id"]

        # Load model
        load_r = await client.post(f"{CONTROLLER_URL}/models/load", json={"model_id": test_model})
        assert load_r.status_code == 202
        assert load_r.json()["mode"] == "single"

        # Clean up
        await client.post(f"http://{ws1_ip}:{AGENT_PORT}/vllm/stop")
        await client.delete(f"{CONTROLLER_URL}/workstations/{ws_id}")
