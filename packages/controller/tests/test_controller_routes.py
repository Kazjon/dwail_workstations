"""
Controller route tests — written before implementation (red).
"""

import pytest
import respx
import httpx

from dwail_shared.models import VLLMState


# --- GET /health ---

async def test_health(client):
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# --- GET /workstations ---

async def test_workstations_empty_initially(client):
    response = await client.get("/workstations")
    assert response.status_code == 200
    assert response.json() == []


async def test_workstations_returns_registered(client, mocker, workstation_online):
    mocker.patch("dwail_controller.registry.list_workstations", return_value=[workstation_online])
    response = await client.get("/workstations")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["ip"] == "10.147.20.5"
    assert data[0]["status"]["vllm_state"] == "idle"


# --- POST /workstations ---

async def test_add_workstation_success(client, mocker):
    mock_status = {
        "ip": "10.147.20.5",
        "agent_version": "0.1.0",
        "gpu_info": [
            {"index": 0, "name": "RTX 3090", "vram_total_mb": 24576, "vram_free_mb": 24000},
            {"index": 1, "name": "RTX 3090", "vram_total_mb": 24576, "vram_free_mb": 24000},
        ],
        "vllm_state": "idle",
        "current_model": None,
        "ray_running": True,
    }
    with respx.mock:
        respx.get("http://10.147.20.5:8765/status").mock(return_value=httpx.Response(200, json=mock_status))
        response = await client.post("/workstations", json={"ip": "10.147.20.5"})

    assert response.status_code == 201
    data = response.json()
    assert data["ip"] == "10.147.20.5"
    assert data["status"]["gpu_info"][0]["vram_total_mb"] == 24576
    assert "id" in data


async def test_add_workstation_unreachable(client):
    with respx.mock:
        respx.get("http://10.147.20.99:8765/status").mock(side_effect=httpx.ConnectError("unreachable"))
        response = await client.post("/workstations", json={"ip": "10.147.20.99"})

    assert response.status_code == 201  # still registers, but status is null
    assert response.json()["status"] is None


async def test_add_workstation_rejects_invalid_ip(client):
    response = await client.post("/workstations", json={"ip": "not-an-ip"})
    assert response.status_code == 422


async def test_add_duplicate_workstation_returns_409(client, mocker, workstation_online):
    mocker.patch("dwail_controller.registry.find_by_ip", return_value=workstation_online)
    response = await client.post("/workstations", json={"ip": "10.147.20.5"})
    assert response.status_code == 409


# --- DELETE /workstations/{id} ---

async def test_remove_workstation(client, mocker, workstation_online):
    mocker.patch("dwail_controller.registry.get", return_value=workstation_online)
    mocker.patch("dwail_controller.registry.remove", return_value=None)

    response = await client.delete(f"/workstations/{workstation_online.id}")
    assert response.status_code == 204


async def test_remove_nonexistent_workstation(client, mocker):
    mocker.patch("dwail_controller.registry.get", return_value=None)
    response = await client.delete("/workstations/does-not-exist")
    assert response.status_code == 404


# --- GET /models/estimate ---

async def test_estimate_vram_known_model(client, mocker):
    mocker.patch("dwail_controller.vram_estimator.estimate", return_value={
        "model_id": "meta-llama/Llama-3.1-8B-Instruct",
        "estimated_vram_mb": 16_384,
        "fits_single_workstation": True,
        "confidence": "high",
    })
    response = await client.get("/models/estimate", params={"model_id": "meta-llama/Llama-3.1-8B-Instruct"})
    assert response.status_code == 200
    data = response.json()
    assert data["fits_single_workstation"] is True
    assert data["confidence"] == "high"


async def test_estimate_vram_missing_model_id(client):
    response = await client.get("/models/estimate")
    assert response.status_code == 422


# --- POST /models/load ---

async def test_load_model_single_workstation(client, mocker, workstation_online):
    mocker.patch("dwail_controller.registry.list_workstations", return_value=[workstation_online])
    mocker.patch("dwail_controller.vram_estimator.estimate", return_value={
        "model_id": "meta-llama/Llama-3.1-8B-Instruct",
        "estimated_vram_mb": 16_384,
        "fits_single_workstation": True,
        "confidence": "high",
    })
    with respx.mock:
        respx.post("http://10.147.20.5:8765/vllm/start").mock(
            return_value=httpx.Response(202, json={"status": "loading"})
        )
        response = await client.post("/models/load", json={"model_id": "meta-llama/Llama-3.1-8B-Instruct"})

    assert response.status_code == 202
    data = response.json()
    assert data["model_id"] == "meta-llama/Llama-3.1-8B-Instruct"
    assert data["mode"] == "single"


async def test_load_model_distributed(client, mocker, two_workstations_online):
    mocker.patch("dwail_controller.registry.list_workstations", return_value=two_workstations_online)
    mocker.patch("dwail_controller.vram_estimator.estimate", return_value={
        "model_id": "meta-llama/Llama-3.1-70B-Instruct",
        "estimated_vram_mb": 60_000,
        "fits_single_workstation": False,
        "confidence": "high",
    })
    with respx.mock:
        respx.post("http://10.147.20.5:8765/vllm/start").mock(
            return_value=httpx.Response(202, json={"status": "loading"})
        )
        respx.post("http://10.147.20.6:8765/vllm/start").mock(
            return_value=httpx.Response(202, json={"status": "loading"})
        )
        response = await client.post("/models/load", json={"model_id": "meta-llama/Llama-3.1-70B-Instruct"})

    assert response.status_code == 202
    assert response.json()["mode"] == "distributed"


async def test_load_model_no_workstations_returns_503(client, mocker):
    mocker.patch("dwail_controller.registry.list_workstations", return_value=[])
    response = await client.post("/models/load", json={"model_id": "some/model"})
    assert response.status_code == 503


async def test_load_model_too_large_for_available_vram(client, mocker, workstation_online):
    mocker.patch("dwail_controller.registry.list_workstations", return_value=[workstation_online])
    mocker.patch("dwail_controller.vram_estimator.estimate", return_value={
        "model_id": "some/huge-model",
        "estimated_vram_mb": 200_000,
        "fits_single_workstation": False,
        "confidence": "high",
    })
    response = await client.post("/models/load", json={"model_id": "some/huge-model"})
    assert response.status_code == 507  # Insufficient Storage
