"""
Detect whether a model supports chat completions or is a base model.

Strategy:
  1. Download tokenizer_config.json from HuggingFace and check for
     a non-empty 'chat_template' field — high confidence result.
  2. Fall back to name heuristics (instruct/chat/it suffixes) — low confidence.
"""

from __future__ import annotations

import json
from typing import Literal

from huggingface_hub import hf_hub_download
from pydantic import BaseModel

# Model name fragments that indicate chat/instruction-tuned variants
_CHAT_KEYWORDS = ("instruct", "chat", "-it", "_it", "sft", "rlhf", "dpo", "assistant")


class ModelCapability(BaseModel):
    model_id: str
    supports_chat: bool
    confidence: Literal["high", "low"]


def detect_capability(model_id: str) -> ModelCapability:
    """Return chat capability for a model, querying HF or falling back to heuristics."""
    try:
        config_path = hf_hub_download(repo_id=model_id, filename="tokenizer_config.json")
        config = json.loads(open(config_path).read())
        supports_chat = bool(config.get("chat_template", ""))
        return ModelCapability(model_id=model_id, supports_chat=supports_chat, confidence="high")
    except Exception:
        pass

    return ModelCapability(
        model_id=model_id,
        supports_chat=_heuristic_chat(model_id),
        confidence="low",
    )


def _heuristic_chat(model_id: str) -> bool:
    name = model_id.lower()
    return any(kw in name for kw in _CHAT_KEYWORDS)
