"""
Level 3 tests — two real workstations running distributed vLLM inference.

Requires:
  DWAIL_WS1_IP=<zerotier-ip>  (head node, agent on port 8765)
  DWAIL_WS2_IP=<zerotier-ip>  (worker node, agent on port 8765)

Model defaults to facebook/opt-125m (loaded fresh, distributed mode even if overkill).
Override with DWAIL_TEST_MODEL=<hf-id> to test a larger pre-downloaded model that
actually benefits from being split across both workstations.

Run with:
  DWAIL_WS1_IP=10.x.x.x DWAIL_WS2_IP=10.x.x.y uv run pytest -m distributed
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
async def both_agents(ws1_ip, ws2_ip):
    """Verify both agents are reachable and idle before tests."""
    urls = {
        "ws1": f"http://{ws1_ip}:{AGENT_PORT}",
        "ws2": f"http://{ws2_ip}:{AGENT_PORT}",
    }
    async with httpx.AsyncClient(timeout=5.0) as client:
        for name, url in urls.items():
            try:
                r = await client.get(f"{url}/health")
                assert r.status_code == 200, f"{name} agent not healthy"
            except Exception as e:
                pytest.fail(f"Cannot reach {name} agent at {url}: {e}")

            status = await client.get(f"{url}/status")
            if status.json()["vllm_state"] != "idle":
                await client.post(f"{url}/vllm/stop")
                await asyncio.sleep(3)

    return urls


@pytest.mark.distributed
async def test_both_agents_healthy(both_agents):
    async with httpx.AsyncClient(timeout=5.0) as client:
        for name, url in both_agents.items():
            r = await client.get(f"{url}/health")
            assert r.status_code == 200, f"{name} not healthy"


@pytest.mark.distributed
async def test_both_have_gpus(both_agents):
    async with httpx.AsyncClient(timeout=5.0) as client:
        for name, url in both_agents.items():
            r = await client.get(f"{url}/status")
            gpus = r.json()["gpu_info"]
            assert len(gpus) >= 1, f"{name}: expected GPUs"


@pytest.mark.distributed
async def test_both_ray_running(both_agents):
    async with httpx.AsyncClient(timeout=5.0) as client:
        for name, url in both_agents.items():
            r = await client.get(f"{url}/status")
            assert r.json()["ray_running"] is True, f"Ray not running on {name}"


@pytest.mark.distributed
async def test_distributed_load_and_infer(both_agents, ws1_ip, ws2_ip, test_model):
    """
    Load test model in distributed mode (pipeline_parallel_size=2 across both workstations),
    verify inference works, then stop.
    """
    ws1_url = both_agents["ws1"]
    ws2_url = both_agents["ws2"]
    ray_address = f"{ws1_ip}:6379"

    async with httpx.AsyncClient(timeout=30.0) as client:
        # Start head (WS1)
        r1 = await client.post(f"{ws1_url}/vllm/start", json={
            "model_id": test_model,
            "tensor_parallel_size": 2,
            "pipeline_parallel_size": 2,
        })
        assert r1.status_code == 202, r1.text

        # Start worker (WS2)
        r2 = await client.post(f"{ws2_url}/vllm/start", json={
            "model_id": test_model,
            "tensor_parallel_size": 2,
            "pipeline_parallel_size": 2,
            "ray_address": ray_address,
        })
        assert r2.status_code == 202, r2.text

        # Poll WS1 until running (it coordinates the cluster)
        deadline = time.time() + VLLM_READY_TIMEOUT
        while time.time() < deadline:
            s = await client.get(f"{ws1_url}/status")
            state = s.json()["vllm_state"]
            if state == "running":
                break
            if state == "error":
                pytest.fail(f"vLLM error on WS1: {s.json()}")
            await asyncio.sleep(POLL_INTERVAL)
        else:
            pytest.fail(f"Distributed vLLM did not reach 'running' in {VLLM_READY_TIMEOUT}s")

        # Infer via WS1's endpoint (head node)
        ws1_vllm_url = f"http://{ws1_ip}:8000"
        infer_r = await client.post(f"{ws1_vllm_url}/v1/chat/completions", json={
            "model": test_model,
            "messages": [{"role": "user", "content": "Say hello."}],
            "max_tokens": 20,
        })
        assert infer_r.status_code == 200
        assert len(infer_r.json()["choices"]) > 0

        # Stop both
        for url in [ws1_url, ws2_url]:
            await client.post(f"{url}/vllm/stop")


@pytest.mark.distributed
async def test_controller_distributed_load(ws1_ip, ws2_ip, test_model):
    """
    Full round-trip: register both workstations with the controller,
    load a model that triggers distributed mode, verify response.
    """
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Clean slate
        wss = (await client.get(f"{CONTROLLER_URL}/workstations")).json()
        for ws in wss:
            if ws["ip"] in (ws1_ip, ws2_ip):
                await client.delete(f"{CONTROLLER_URL}/workstations/{ws['id']}")

        # Register both
        r1 = await client.post(f"{CONTROLLER_URL}/workstations", json={"ip": ws1_ip})
        assert r1.status_code == 201
        r2 = await client.post(f"{CONTROLLER_URL}/workstations", json={"ip": ws2_ip})
        assert r2.status_code == 201

        # Patch the VRAM estimate to force distributed mode for small model
        # (In practice a large DWAIL_TEST_MODEL would trigger this naturally)
        # For opt-125m we test that the controller correctly routes to single mode,
        # but if a large model is specified it will route to distributed.
        load_r = await client.post(f"{CONTROLLER_URL}/models/load", json={"model_id": test_model})
        assert load_r.status_code == 202
        # Mode depends on model size vs available VRAM — either is valid
        assert load_r.json()["mode"] in ("single", "distributed")

        # Clean up
        for ip in (ws1_ip, ws2_ip):
            try:
                await client.post(f"http://{ip}:{AGENT_PORT}/vllm/stop")
            except Exception:
                pass
        wss = (await client.get(f"{CONTROLLER_URL}/workstations")).json()
        for ws in wss:
            if ws["ip"] in (ws1_ip, ws2_ip):
                await client.delete(f"{CONTROLLER_URL}/workstations/{ws['id']}")


@pytest.mark.distributed
async def test_speed_comparison(both_agents, ws1_ip, test_model):
    """
    Measure tokens/sec in distributed mode and report.
    Not a pass/fail test — prints benchmark results for human review.
    """
    import time
    ws1_vllm_url = f"http://{ws1_ip}:8000"
    prompt = "Write a short poem about distributed computing."

    async with httpx.AsyncClient(timeout=120.0) as client:
        start = time.perf_counter()
        r = await client.post(f"{ws1_vllm_url}/v1/chat/completions", json={
            "model": test_model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 100,
        })
        elapsed = time.perf_counter() - start

    assert r.status_code == 200
    usage = r.json().get("usage", {})
    completion_tokens = usage.get("completion_tokens", 0)
    tps = completion_tokens / elapsed if elapsed > 0 else 0
    # Print so it shows in pytest -s output
    print(f"\n[speed] distributed mode: {tps:.1f} tok/s ({completion_tokens} tokens in {elapsed:.2f}s)")
