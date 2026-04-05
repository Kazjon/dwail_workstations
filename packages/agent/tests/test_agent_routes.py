"""
Agent route tests — written before implementation (red).
Each test documents the expected API contract for the workstation agent.
"""

import pytest
from unittest.mock import patch, MagicMock


# --- GET /health ---

async def test_health_returns_200(client):
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# --- GET /status ---

async def test_status_returns_workstation_status(client, mock_gpu_info, mocker):
    mocker.patch("dwail_agent.gpu.get_gpu_info", return_value=mock_gpu_info)
    mocker.patch("dwail_agent.vllm_manager.get_state", return_value="idle")
    mocker.patch("dwail_agent.vllm_manager.get_current_model", return_value=None)
    mocker.patch("dwail_agent.ray_manager.is_running", return_value=True)

    response = await client.get("/status")
    assert response.status_code == 200

    data = response.json()
    assert data["vllm_state"] == "idle"
    assert data["current_model"] is None
    assert data["ray_running"] is True
    assert len(data["gpu_info"]) == 2
    assert data["gpu_info"][0]["vram_total_mb"] == 24576


async def test_status_reflects_running_model(client, mock_gpu_info, mocker):
    mocker.patch("dwail_agent.gpu.get_gpu_info", return_value=mock_gpu_info)
    mocker.patch("dwail_agent.vllm_manager.get_state", return_value="running")
    mocker.patch("dwail_agent.vllm_manager.get_current_model", return_value="meta-llama/Llama-3.1-8B-Instruct")
    mocker.patch("dwail_agent.ray_manager.is_running", return_value=True)

    response = await client.get("/status")
    assert response.status_code == 200
    assert response.json()["vllm_state"] == "running"
    assert response.json()["current_model"] == "meta-llama/Llama-3.1-8B-Instruct"


# --- GET /models ---

async def test_models_returns_list(client, mocker):
    mocker.patch("dwail_agent.model_scanner.scan", return_value=[
        {"model_id": "meta-llama/Llama-3.1-8B-Instruct", "path": "/mnt/models/Llama-3.1-8B-Instruct", "size_bytes": 16_000_000_000},
    ])

    response = await client.get("/models")
    assert response.status_code == 200

    data = response.json()
    assert isinstance(data, list)
    assert data[0]["model_id"] == "meta-llama/Llama-3.1-8B-Instruct"
    assert data[0]["size_bytes"] == 16_000_000_000


async def test_models_returns_empty_when_none(client, mocker):
    mocker.patch("dwail_agent.model_scanner.scan", return_value=[])
    response = await client.get("/models")
    assert response.status_code == 200
    assert response.json() == []


# --- POST /vllm/start ---

async def test_vllm_start_accepted(client, mocker):
    mock_start = mocker.patch("dwail_agent.vllm_manager.start", return_value=None)

    response = await client.post("/vllm/start", json={
        "model_id": "meta-llama/Llama-3.1-8B-Instruct",
        "tensor_parallel_size": 2,
        "pipeline_parallel_size": 1,
    })
    assert response.status_code == 202
    mock_start.assert_called_once()


async def test_vllm_start_distributed(client, mocker):
    mock_start = mocker.patch("dwail_agent.vllm_manager.start", return_value=None)

    response = await client.post("/vllm/start", json={
        "model_id": "meta-llama/Llama-3.1-70B-Instruct",
        "tensor_parallel_size": 2,
        "pipeline_parallel_size": 2,
        "ray_address": "10.147.20.6:6379",
    })
    assert response.status_code == 202
    call_kwargs = mock_start.call_args
    assert call_kwargs is not None


async def test_vllm_start_rejects_invalid_request(client):
    response = await client.post("/vllm/start", json={"tensor_parallel_size": 2})
    assert response.status_code == 422  # missing model_id


async def test_vllm_start_returns_409_when_already_running(client, mocker):
    mocker.patch("dwail_agent.vllm_manager.get_state", return_value="running")
    mocker.patch("dwail_agent.vllm_manager.start", return_value=None)

    response = await client.post("/vllm/start", json={"model_id": "some/model"})
    assert response.status_code == 409


# --- POST /vllm/stop ---

async def test_vllm_stop_accepted(client, mocker):
    mocker.patch("dwail_agent.vllm_manager.get_state", return_value="running")
    mock_stop = mocker.patch("dwail_agent.vllm_manager.stop", return_value=None)

    response = await client.post("/vllm/stop")
    assert response.status_code == 202
    mock_stop.assert_called_once()


async def test_vllm_stop_when_idle_returns_409(client, mocker):
    mocker.patch("dwail_agent.vllm_manager.get_state", return_value="idle")

    response = await client.post("/vllm/stop")
    assert response.status_code == 409
