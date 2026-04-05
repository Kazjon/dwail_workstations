"""
Tests for shared Pydantic models.
These define the API contract between agent and controller.
"""

import pytest
from pydantic import ValidationError

from dwail_shared.models import (
    GPUInfo,
    VLLMState,
    WorkstationStatus,
    ModelInfo,
    StartVLLMRequest,
    VRAMEstimate,
    AddWorkstationRequest,
    LoadModelRequest,
    RegisteredWorkstation,
)


# --- GPUInfo ---

def test_gpu_info_valid():
    gpu = GPUInfo(index=0, name="NVIDIA RTX 3090", vram_total_mb=24576, vram_free_mb=20000)
    assert gpu.index == 0
    assert gpu.vram_total_mb == 24576


def test_gpu_info_rejects_negative_vram():
    with pytest.raises(ValidationError):
        GPUInfo(index=0, name="RTX 3090", vram_total_mb=-1, vram_free_mb=0)


def test_gpu_info_rejects_free_exceeds_total():
    with pytest.raises(ValidationError):
        GPUInfo(index=0, name="RTX 3090", vram_total_mb=1000, vram_free_mb=2000)


# --- VLLMState ---

def test_vllm_state_values():
    assert VLLMState.idle == "idle"
    assert VLLMState.loading == "loading"
    assert VLLMState.running == "running"
    assert VLLMState.error == "error"


# --- WorkstationStatus ---

def test_workstation_status_idle():
    status = WorkstationStatus(
        ip="10.0.0.1",
        agent_version="0.1.0",
        gpu_info=[
            GPUInfo(index=0, name="RTX 3090", vram_total_mb=24576, vram_free_mb=24000),
            GPUInfo(index=1, name="RTX 3090", vram_total_mb=24576, vram_free_mb=24000),
        ],
        vllm_state=VLLMState.idle,
        ray_running=True,
    )
    assert status.current_model is None
    assert status.vllm_state == VLLMState.idle


def test_workstation_status_running_requires_no_model_field():
    # current_model is optional even when running — vLLM may still be initialising
    status = WorkstationStatus(
        ip="10.0.0.1",
        agent_version="0.1.0",
        gpu_info=[],
        vllm_state=VLLMState.running,
        current_model="meta-llama/Llama-3.1-8B-Instruct",
        ray_running=True,
    )
    assert status.current_model == "meta-llama/Llama-3.1-8B-Instruct"


def test_workstation_status_total_vram_mb():
    status = WorkstationStatus(
        ip="10.0.0.1",
        agent_version="0.1.0",
        gpu_info=[
            GPUInfo(index=0, name="RTX 3090", vram_total_mb=24576, vram_free_mb=10000),
            GPUInfo(index=1, name="RTX 3090", vram_total_mb=24576, vram_free_mb=8000),
        ],
        vllm_state=VLLMState.idle,
        ray_running=True,
    )
    assert status.total_vram_mb == 49152


def test_workstation_status_free_vram_mb():
    status = WorkstationStatus(
        ip="10.0.0.1",
        agent_version="0.1.0",
        gpu_info=[
            GPUInfo(index=0, name="RTX 3090", vram_total_mb=24576, vram_free_mb=10000),
            GPUInfo(index=1, name="RTX 3090", vram_total_mb=24576, vram_free_mb=8000),
        ],
        vllm_state=VLLMState.idle,
        ray_running=True,
    )
    assert status.free_vram_mb == 18000


# --- ModelInfo ---

def test_model_info_minimal():
    m = ModelInfo(model_id="meta-llama/Llama-3.1-8B-Instruct", path="/mnt/models/Llama-3.1-8B-Instruct")
    assert m.size_bytes is None


def test_model_info_with_size():
    m = ModelInfo(
        model_id="meta-llama/Llama-3.1-8B-Instruct",
        path="/mnt/models/Llama-3.1-8B-Instruct",
        size_bytes=16_000_000_000,
    )
    assert m.size_bytes == 16_000_000_000


# --- StartVLLMRequest ---

def test_start_vllm_request_defaults():
    req = StartVLLMRequest(model_id="meta-llama/Llama-3.1-8B-Instruct")
    assert req.tensor_parallel_size == 2
    assert req.pipeline_parallel_size == 1
    assert req.ray_address is None


def test_start_vllm_request_distributed():
    req = StartVLLMRequest(
        model_id="meta-llama/Llama-3.1-70B-Instruct",
        tensor_parallel_size=2,
        pipeline_parallel_size=2,
        ray_address="10.0.0.2:6379",
    )
    assert req.pipeline_parallel_size == 2
    assert req.ray_address == "10.0.0.2:6379"


def test_start_vllm_request_rejects_zero_parallelism():
    with pytest.raises(ValidationError):
        StartVLLMRequest(model_id="x", tensor_parallel_size=0)


# --- VRAMEstimate ---

def test_vram_estimate_single_workstation():
    est = VRAMEstimate(
        model_id="meta-llama/Llama-3.1-8B-Instruct",
        estimated_vram_mb=16_384,
        fits_single_workstation=True,
        confidence="high",
    )
    assert est.fits_single_workstation is True


def test_vram_estimate_rejects_invalid_confidence():
    with pytest.raises(ValidationError):
        VRAMEstimate(
            model_id="x",
            estimated_vram_mb=1000,
            fits_single_workstation=True,
            confidence="maybe",
        )


# --- AddWorkstationRequest ---

def test_add_workstation_default_port():
    req = AddWorkstationRequest(ip="10.147.20.5")
    assert req.agent_port == 8765


def test_add_workstation_rejects_invalid_ip():
    with pytest.raises(ValidationError):
        AddWorkstationRequest(ip="not-an-ip")


# --- LoadModelRequest ---

def test_load_model_request():
    req = LoadModelRequest(model_id="meta-llama/Llama-3.1-70B-Instruct")
    assert req.model_id == "meta-llama/Llama-3.1-70B-Instruct"


# --- RegisteredWorkstation ---

def test_registered_workstation_offline():
    ws = RegisteredWorkstation(id="abc-123", ip="10.147.20.5", agent_port=8765)
    assert ws.status is None  # unreachable


def test_registered_workstation_serialises():
    ws = RegisteredWorkstation(id="abc-123", ip="10.147.20.5", agent_port=8765)
    data = ws.model_dump()
    assert data["id"] == "abc-123"
    assert data["status"] is None
