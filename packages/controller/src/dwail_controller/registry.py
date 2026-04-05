"""In-memory workstation registry (backed by a JSON config file)."""

from __future__ import annotations
import json
import uuid
from pathlib import Path

from dwail_shared.models import RegisteredWorkstation, WorkstationStatus

CONFIG_PATH = Path.home() / ".dwail" / "workstations.json"


_workstations: dict[str, RegisteredWorkstation] = {}


def _load() -> None:
    if CONFIG_PATH.exists():
        data = json.loads(CONFIG_PATH.read_text())
        for item in data:
            ws = RegisteredWorkstation.model_validate(item)
            _workstations[ws.id] = ws


def _save() -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(
        json.dumps([ws.model_dump() for ws in _workstations.values()], indent=2)
    )


def list_workstations() -> list[RegisteredWorkstation]:
    return list(_workstations.values())


def get(ws_id: str) -> RegisteredWorkstation | None:
    return _workstations.get(ws_id)


def find_by_ip(ip: str) -> RegisteredWorkstation | None:
    return next((ws for ws in _workstations.values() if ws.ip == ip), None)


def add(ip: str, agent_port: int, status: WorkstationStatus | None) -> RegisteredWorkstation:
    ws = RegisteredWorkstation(id=str(uuid.uuid4()), ip=ip, agent_port=agent_port, status=status)
    _workstations[ws.id] = ws
    _save()
    return ws


def update_status(ws_id: str, status: WorkstationStatus | None) -> None:
    if ws_id in _workstations:
        _workstations[ws_id] = _workstations[ws_id].model_copy(update={"status": status})
        _save()


def remove(ws_id: str) -> None:
    _workstations.pop(ws_id, None)
    _save()


def clear() -> None:
    """Reset in-memory state (for tests only)."""
    _workstations.clear()


# Load persisted state on import
_load()
