from __future__ import annotations

import uvicorn
import httpx
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from dwail_shared.models import (
    AddWorkstationRequest,
    LoadModelRequest,
    RegisteredWorkstation,
    StartVLLMRequest,
    VRAMEstimate,
    WorkstationStatus,
)
from dwail_controller import registry, vram_estimator
from dwail_controller.status_poller import start_poller


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = await start_poller()
    yield
    task.cancel()


app = FastAPI(title="dwail-controller", lifespan=lifespan)

# Total VRAM across all workstations (2x RTX 3090 each)
SINGLE_WS_VRAM_MB = 48 * 1024
RAY_PORT = 6379


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/workstations", response_model=list[RegisteredWorkstation])
async def get_workstations():
    return registry.list_workstations()


@app.post("/workstations", response_model=RegisteredWorkstation, status_code=201)
async def add_workstation(req: AddWorkstationRequest):
    existing = registry.find_by_ip(req.ip)
    if existing:
        raise HTTPException(status_code=409, detail=f"Workstation {req.ip} is already registered.")

    status = await _fetch_status(req.ip, req.agent_port)
    ws = registry.add(ip=req.ip, agent_port=req.agent_port, status=status)
    return ws


@app.delete("/workstations/{ws_id}", status_code=204)
async def remove_workstation(ws_id: str):
    ws = registry.get(ws_id)
    if not ws:
        raise HTTPException(status_code=404, detail="Workstation not found.")
    registry.remove(ws_id)
    return Response(status_code=204)


@app.get("/models/estimate", response_model=VRAMEstimate)
async def estimate_vram(model_id: str = Query(...)):
    return vram_estimator.estimate(model_id)


@app.post("/models/load", status_code=202)
async def load_model(req: LoadModelRequest):
    workstations = registry.list_workstations()
    if not workstations:
        raise HTTPException(status_code=503, detail="No workstations registered.")

    estimate = vram_estimator.estimate(req.model_id)
    total_available_mb = sum(
        (ws.status.free_vram_mb if ws.status else 0) for ws in workstations
    )

    if estimate["estimated_vram_mb"] > total_available_mb:
        raise HTTPException(
            status_code=507,
            detail=f"Model requires ~{estimate['estimated_vram_mb']}MB VRAM but only {total_available_mb}MB available.",
        )

    if estimate["fits_single_workstation"]:
        # Use the workstation with the most free VRAM
        target = max(
            (ws for ws in workstations if ws.status),
            key=lambda ws: ws.status.free_vram_mb,
        )
        await _start_vllm(target, req.model_id, tensor_parallel=2, pipeline_parallel=1)
        return {"model_id": req.model_id, "mode": "single", "workstation": target.ip}
    else:
        # Distributed: head on first workstation, worker on second
        if len(workstations) < 2:
            raise HTTPException(
                status_code=507,
                detail="Model requires distributed inference but only one workstation is available.",
            )
        head, worker = workstations[0], workstations[1]
        ray_address = f"{head.ip}:{RAY_PORT}"
        await _start_vllm(head, req.model_id, tensor_parallel=2, pipeline_parallel=2)
        await _start_vllm(worker, req.model_id, tensor_parallel=2, pipeline_parallel=2, ray_address=ray_address)
        return {"model_id": req.model_id, "mode": "distributed", "workstations": [head.ip, worker.ip]}


async def _fetch_status(ip: str, port: int) -> WorkstationStatus | None:
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"http://{ip}:{port}/status")
            resp.raise_for_status()
            return WorkstationStatus.model_validate(resp.json())
    except Exception:
        return None


async def _start_vllm(
    ws: RegisteredWorkstation,
    model_id: str,
    tensor_parallel: int,
    pipeline_parallel: int,
    ray_address: str | None = None,
) -> None:
    req = StartVLLMRequest(
        model_id=model_id,
        tensor_parallel_size=tensor_parallel,
        pipeline_parallel_size=pipeline_parallel,
        ray_address=ray_address,
    )
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(
            f"http://{ws.ip}:{ws.agent_port}/vllm/start",
            json=req.model_dump(),
        )
        resp.raise_for_status()


# Serve UI static files
_ui_dir = Path(__file__).parent.parent.parent.parent / "ui" / "src"
if _ui_dir.exists():
    app.mount("/", StaticFiles(directory=str(_ui_dir), html=True), name="ui")


def run():
    uvicorn.run("dwail_controller.main:app", host="127.0.0.1", port=8080, reload=False)
