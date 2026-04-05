import pytest
from httpx import AsyncClient, ASGITransport

from dwail_controller.main import app
from dwail_controller import registry


@pytest.fixture(autouse=True)
def reset_registry():
    """Clear the in-memory registry before each test to prevent state leakage."""
    registry.clear()
    yield
    registry.clear()


@pytest.fixture
async def client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


@pytest.fixture
def workstation_online():
    """A registered workstation that is reachable and idle."""
    from dwail_shared.models import GPUInfo, VLLMState, WorkstationStatus, RegisteredWorkstation
    return RegisteredWorkstation(
        id="ws-aaa",
        ip="10.147.20.5",
        agent_port=8765,
        status=WorkstationStatus(
            ip="10.147.20.5",
            agent_version="0.1.0",
            gpu_info=[
                GPUInfo(index=0, name="RTX 3090", vram_total_mb=24576, vram_free_mb=24000),
                GPUInfo(index=1, name="RTX 3090", vram_total_mb=24576, vram_free_mb=24000),
            ],
            vllm_state=VLLMState.idle,
            ray_running=True,
        ),
    )


@pytest.fixture
def two_workstations_online(workstation_online):
    """Two registered workstations, both idle."""
    from dwail_shared.models import GPUInfo, VLLMState, WorkstationStatus, RegisteredWorkstation
    ws_b = RegisteredWorkstation(
        id="ws-bbb",
        ip="10.147.20.6",
        agent_port=8765,
        status=WorkstationStatus(
            ip="10.147.20.6",
            agent_version="0.1.0",
            gpu_info=[
                GPUInfo(index=0, name="RTX 3090", vram_total_mb=24576, vram_free_mb=24000),
                GPUInfo(index=1, name="RTX 3090", vram_total_mb=24576, vram_free_mb=24000),
            ],
            vllm_state=VLLMState.idle,
            ray_running=True,
        ),
    )
    return [workstation_online, ws_b]
