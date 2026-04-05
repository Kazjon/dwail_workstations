import pytest
from httpx import AsyncClient, ASGITransport

from dwail_agent.main import app


@pytest.fixture
async def client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


@pytest.fixture
def mock_gpu_info():
    """Two RTX 3090s, mostly free."""
    return [
        {"index": 0, "name": "NVIDIA GeForce RTX 3090", "vram_total_mb": 24576, "vram_free_mb": 24000},
        {"index": 1, "name": "NVIDIA GeForce RTX 3090", "vram_total_mb": 24576, "vram_free_mb": 24000},
    ]
