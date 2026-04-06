from __future__ import annotations

from enum import Enum
from ipaddress import IPv4Address
from typing import Literal

from pydantic import BaseModel, computed_field, field_validator, model_validator


class VLLMState(str, Enum):
    idle = "idle"
    loading = "loading"
    running = "running"
    error = "error"


class GPUInfo(BaseModel):
    index: int
    name: str
    vram_total_mb: int
    vram_free_mb: int

    @field_validator("vram_total_mb", "vram_free_mb")
    @classmethod
    def non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError("VRAM values must be non-negative")
        return v

    @model_validator(mode="after")
    def free_not_exceeds_total(self) -> GPUInfo:
        if self.vram_free_mb > self.vram_total_mb:
            raise ValueError("vram_free_mb cannot exceed vram_total_mb")
        return self


class WorkstationStatus(BaseModel):
    ip: str
    agent_version: str
    gpu_info: list[GPUInfo]
    vllm_state: VLLMState
    current_model: str | None = None
    ray_running: bool

    @computed_field
    @property
    def total_vram_mb(self) -> int:
        return sum(g.vram_total_mb for g in self.gpu_info)

    @computed_field
    @property
    def free_vram_mb(self) -> int:
        return sum(g.vram_free_mb for g in self.gpu_info)


class ModelInfo(BaseModel):
    model_id: str
    path: str
    size_bytes: int | None = None


class StartVLLMRequest(BaseModel):
    model_id: str
    tensor_parallel_size: int = 2
    pipeline_parallel_size: int = 1
    ray_address: str | None = None

    @field_validator("tensor_parallel_size", "pipeline_parallel_size")
    @classmethod
    def at_least_one(cls, v: int) -> int:
        if v < 1:
            raise ValueError("Parallelism size must be >= 1")
        return v


class VRAMEstimate(BaseModel):
    model_id: str
    estimated_vram_mb: int
    fits_single_workstation: bool
    confidence: Literal["high", "low"]


class AddWorkstationRequest(BaseModel):
    ip: str
    agent_port: int = 8765

    @field_validator("ip")
    @classmethod
    def valid_ip(cls, v: str) -> str:
        IPv4Address(v)  # raises ValueError if invalid
        return v


class LoadModelRequest(BaseModel):
    model_id: str


class RegisteredWorkstation(BaseModel):
    id: str
    ip: str
    agent_port: int
    status: WorkstationStatus | None = None
