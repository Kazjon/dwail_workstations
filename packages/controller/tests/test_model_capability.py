"""
Tests for model capability detection.

Detects whether a model supports chat completions (has a chat template)
or is a base model (completions only), by inspecting HuggingFace
tokenizer_config.json. Falls back to heuristics on failure.
"""

import pytest
from unittest.mock import patch, mock_open
import json

from dwail_controller.model_capability import detect_capability, ModelCapability


# --- ModelCapability ---

def test_model_capability_fields():
    cap = ModelCapability(
        model_id="meta-llama/Llama-3.1-8B-Instruct",
        supports_chat=True,
        confidence="high",
    )
    assert cap.supports_chat is True
    assert cap.confidence == "high"


def test_model_capability_rejects_invalid_confidence():
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        ModelCapability(model_id="x", supports_chat=True, confidence="maybe")


# --- detect_capability: HF tokenizer_config path ---

def test_detects_chat_model_via_tokenizer_config(mocker):
    """Model with chat_template in tokenizer_config → supports_chat=True, high confidence."""
    tokenizer_config = {"chat_template": "{% for message in messages %}...{% endfor %}"}
    mocker.patch(
        "dwail_controller.model_capability.hf_hub_download",
        return_value="/fake/path/tokenizer_config.json",
    )
    mocker.patch("builtins.open", mock_open(read_data=json.dumps(tokenizer_config)))

    result = detect_capability("meta-llama/Llama-3.1-8B-Instruct")

    assert result.supports_chat is True
    assert result.confidence == "high"


def test_detects_base_model_via_tokenizer_config(mocker):
    """Model without chat_template → supports_chat=False, high confidence."""
    tokenizer_config = {"model_type": "gpt2", "vocab_size": 50257}
    mocker.patch(
        "dwail_controller.model_capability.hf_hub_download",
        return_value="/fake/path/tokenizer_config.json",
    )
    mocker.patch("builtins.open", mock_open(read_data=json.dumps(tokenizer_config)))

    result = detect_capability("facebook/opt-125m")

    assert result.supports_chat is False
    assert result.confidence == "high"


def test_detects_base_model_with_empty_chat_template(mocker):
    """Empty string chat_template counts as no chat support."""
    tokenizer_config = {"chat_template": ""}
    mocker.patch(
        "dwail_controller.model_capability.hf_hub_download",
        return_value="/fake/path/tokenizer_config.json",
    )
    mocker.patch("builtins.open", mock_open(read_data=json.dumps(tokenizer_config)))

    result = detect_capability("some/base-model")

    assert result.supports_chat is False


# --- detect_capability: heuristic fallback ---

def test_heuristic_instruct_model_is_chat(mocker):
    """HF unavailable → fall back to name heuristics; 'instruct' → chat."""
    mocker.patch(
        "dwail_controller.model_capability.hf_hub_download",
        side_effect=Exception("network error"),
    )

    result = detect_capability("meta-llama/Llama-3.1-8B-Instruct")

    assert result.supports_chat is True
    assert result.confidence == "low"


def test_heuristic_chat_model_is_chat(mocker):
    mocker.patch(
        "dwail_controller.model_capability.hf_hub_download",
        side_effect=Exception("network error"),
    )

    result = detect_capability("mistralai/Mistral-7B-v0.1-chat")

    assert result.supports_chat is True
    assert result.confidence == "low"


def test_heuristic_base_model_is_not_chat(mocker):
    mocker.patch(
        "dwail_controller.model_capability.hf_hub_download",
        side_effect=Exception("network error"),
    )

    result = detect_capability("facebook/opt-125m")

    assert result.supports_chat is False
    assert result.confidence == "low"


def test_heuristic_unknown_model_defaults_to_not_chat(mocker):
    mocker.patch(
        "dwail_controller.model_capability.hf_hub_download",
        side_effect=Exception("network error"),
    )

    result = detect_capability("some-org/mystery-model-7b")

    assert result.supports_chat is False
    assert result.confidence == "low"


# --- detect_capability returns correct model_id ---

def test_result_contains_model_id(mocker):
    mocker.patch(
        "dwail_controller.model_capability.hf_hub_download",
        side_effect=Exception("network error"),
    )
    result = detect_capability("facebook/opt-125m")
    assert result.model_id == "facebook/opt-125m"
