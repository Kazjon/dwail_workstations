"""Scan the model directory and return available models."""

from __future__ import annotations
import os
from pathlib import Path

MODEL_DIR = Path(os.environ.get("DWAIL_MODEL_DIR", "/mnt/models"))


def scan() -> list[dict]:
    """
    Scan MODEL_DIR for model directories.
    Expects layout: /mnt/models/<org>/<model_name> or /mnt/models/<model_name>
    Returns list of dicts compatible with ModelInfo.
    """
    if not MODEL_DIR.exists():
        return []

    models = []
    for entry in MODEL_DIR.iterdir():
        if entry.is_dir():
            # Check for HuggingFace-style layout: org/model
            sub = list(entry.iterdir())
            if any(s.is_dir() for s in sub):
                for sub_entry in sub:
                    if sub_entry.is_dir():
                        model_id = f"{entry.name}/{sub_entry.name}"
                        size = _dir_size(sub_entry)
                        models.append({"model_id": model_id, "path": str(sub_entry), "size_bytes": size})
            else:
                size = _dir_size(entry)
                models.append({"model_id": entry.name, "path": str(entry), "size_bytes": size})
    return models


def _dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
