from __future__ import annotations
import socket

import uvicorn
from fastapi import FastAPI, HTTPException

from dwail_shared.models import (
    GPUInfo,
    ModelInfo,
    StartVLLMRequest,
    VLLMState,
    WorkstationStatus,
)
from dwail_agent import gpu, model_scanner, ray_manager, vllm_manager

app = FastAPI(title="dwail-agent")

AGENT_VERSION = "0.1.0"


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/status", response_model=WorkstationStatus)
async def status():
    try:
        ip = socket.gethostbyname(socket.gethostname())
    except socket.gaierror:
        ip = "127.0.0.1"

    return WorkstationStatus(
        ip=ip,
        agent_version=AGENT_VERSION,
        gpu_info=[GPUInfo(**g) for g in gpu.get_gpu_info()],
        vllm_state=vllm_manager.get_state(),
        current_model=vllm_manager.get_current_model(),
        ray_running=ray_manager.is_running(),
    )


@app.get("/models", response_model=list[ModelInfo])
async def models():
    return [ModelInfo(**m) for m in model_scanner.scan()]


@app.post("/vllm/start", status_code=202)
async def vllm_start(request: StartVLLMRequest):
    if vllm_manager.get_state() in (VLLMState.running, VLLMState.loading):
        raise HTTPException(status_code=409, detail="vLLM is already running. Stop it first.")
    vllm_manager.start(request)
    return {"status": "loading", "model_id": request.model_id}


@app.post("/vllm/stop", status_code=202)
async def vllm_stop():
    if vllm_manager.get_state() == VLLMState.idle:
        raise HTTPException(status_code=409, detail="vLLM is not running.")
    vllm_manager.stop()
    return {"status": "stopped"}


def run():
    import os
    port = int(os.environ.get("DWAIL_PORT", 8765))
    uvicorn.run("dwail_agent.main:app", host="0.0.0.0", port=port, reload=False)
