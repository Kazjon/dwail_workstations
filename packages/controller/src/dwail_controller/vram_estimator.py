"""
Estimate VRAM requirements for a model using HuggingFace config.json.

Formula: num_parameters * bytes_per_param * 1.2 overhead
Falls back to low-confidence estimate if config.json is unavailable.
"""

from __future__ import annotations

import httpx
from huggingface_hub import hf_hub_download
import json

# RTX 3090 VRAM per workstation (2x 24GB)
SINGLE_WS_VRAM_MB = 48 * 1024


def estimate(model_id: str) -> dict:
    try:
        config_path = hf_hub_download(repo_id=model_id, filename="config.json")
        config = json.loads(open(config_path).read())
        params = _count_params(config)
        if params:
            # Assume bf16 (2 bytes/param) + 20% overhead for KV cache / activations
            vram_mb = int(params * 2 * 1.2 / (1024 * 1024))
            return {
                "model_id": model_id,
                "estimated_vram_mb": vram_mb,
                "fits_single_workstation": vram_mb <= SINGLE_WS_VRAM_MB,
                "confidence": "high",
            }
    except Exception:
        pass

    # Low-confidence fallback: use model name heuristics
    vram_mb = _heuristic_estimate(model_id)
    return {
        "model_id": model_id,
        "estimated_vram_mb": vram_mb,
        "fits_single_workstation": vram_mb <= SINGLE_WS_VRAM_MB,
        "confidence": "low",
    }


def _count_params(config: dict) -> int | None:
    """Extract parameter count from HF config.json."""
    # Some configs have this directly
    if "num_parameters" in config:
        return config["num_parameters"]

    # Otherwise estimate from architecture
    hidden = config.get("hidden_size")
    layers = config.get("num_hidden_layers")
    intermediate = config.get("intermediate_size")
    vocab = config.get("vocab_size")

    if not all([hidden, layers, intermediate, vocab]):
        return None

    # Rough transformer param count: embedding + layers * (attn + ffn)
    embedding_params = vocab * hidden
    attn_params = 4 * hidden * hidden  # Q, K, V, O projections
    ffn_params = 3 * hidden * intermediate  # gate, up, down (SwiGLU)
    layer_params = attn_params + ffn_params
    return embedding_params + (layers * layer_params)


def _heuristic_estimate(model_id: str) -> int:
    """Rough VRAM estimate from model name (B parameter count)."""
    name = model_id.lower()
    for size_b, vram_mb in [
        (405, 405 * 2 * 1024),
        (70, 70 * 2 * 1024),
        (34, 34 * 2 * 1024),
        (13, 13 * 2 * 1024),
        (8, 8 * 2 * 1024),
        (7, 7 * 2 * 1024),
        (3, 3 * 2 * 1024),
    ]:
        if f"{size_b}b" in name:
            return int(vram_mb * 1.2)
    return 16 * 1024  # default: assume ~16GB
