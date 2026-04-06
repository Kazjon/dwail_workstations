"""Background task that periodically polls each registered agent for status."""

from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable

import httpx

from dwail_shared.models import WorkstationStatus
from dwail_controller import registry

log = logging.getLogger(__name__)

POLL_INTERVAL = 10  # seconds


async def poll_once() -> None:
    """Fetch /status from every registered workstation and update the registry."""
    workstations = registry.list_workstations()
    if not workstations:
        return

    async with httpx.AsyncClient(timeout=5.0) as client:
        for ws in workstations:
            url = f"http://{ws.ip}:{ws.agent_port}/status"
            try:
                r = await client.get(url)
                if r.status_code == 200:
                    status = WorkstationStatus.model_validate(r.json())
                    registry.update_status(ws.id, status)
                else:
                    log.warning("Agent %s returned %s", url, r.status_code)
                    registry.update_status(ws.id, None)
            except Exception as e:
                log.warning("Agent %s unreachable: %s", url, e)
                registry.update_status(ws.id, None)


async def start_poller(
    poll_fn: Callable[[], Awaitable[None]] = poll_once,
    interval: float = POLL_INTERVAL,
) -> asyncio.Task:
    """Start the background polling loop and return its Task."""

    async def _loop() -> None:
        while True:
            try:
                await poll_fn()
            except Exception as e:
                log.error("Poller error: %s", e)
            await asyncio.sleep(interval)

    return asyncio.create_task(_loop())
