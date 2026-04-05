"""Tests for gpu.py — including graceful fallback on non-NVIDIA machines."""

import pytest
from unittest.mock import MagicMock, patch


def test_get_gpu_info_returns_gpu_list(mocker):
    """Normal path: pynvml available and reports two GPUs."""
    mock_nvml = MagicMock()
    mock_nvml.nvmlDeviceGetCount.return_value = 2

    def fake_handle(i):
        return f"handle_{i}"

    def fake_mem(handle):
        m = MagicMock()
        m.total = 24 * 1024 * 1024 * 1024  # 24GB in bytes
        m.free = 20 * 1024 * 1024 * 1024
        return m

    mock_nvml.nvmlDeviceGetHandleByIndex.side_effect = fake_handle
    mock_nvml.nvmlDeviceGetMemoryInfo.side_effect = fake_mem
    mock_nvml.nvmlDeviceGetName.return_value = "NVIDIA GeForce RTX 3090"

    with patch.dict("sys.modules", {"pynvml": mock_nvml}):
        # Re-import to pick up the mock
        import importlib
        import dwail_agent.gpu as gpu_mod
        importlib.reload(gpu_mod)
        result = gpu_mod.get_gpu_info()

    assert len(result) == 2
    assert result[0]["index"] == 0
    assert result[0]["vram_total_mb"] == 24 * 1024
    assert result[0]["vram_free_mb"] == 20 * 1024
    assert result[0]["name"] == "NVIDIA GeForce RTX 3090"


def test_get_gpu_info_returns_empty_when_no_nvidia_driver(mocker):
    """pynvml raises NVMLError on init (no NVIDIA driver) — should return empty list."""
    mock_nvml = MagicMock()
    mock_nvml.NVMLError = Exception
    mock_nvml.nvmlInit.side_effect = Exception("NVML Shared Library Not Found")

    with patch.dict("sys.modules", {"pynvml": mock_nvml}):
        import importlib
        import dwail_agent.gpu as gpu_mod
        importlib.reload(gpu_mod)
        result = gpu_mod.get_gpu_info()

    assert result == []


def test_get_gpu_info_returns_empty_when_pynvml_not_installed():
    """pynvml not installed at all — should return empty list."""
    import sys
    import importlib
    import dwail_agent.gpu as gpu_mod

    original = sys.modules.pop("pynvml", None)
    try:
        sys.modules["pynvml"] = None  # simulate import failure
        importlib.reload(gpu_mod)
        result = gpu_mod.get_gpu_info()
        assert result == []
    finally:
        if original is not None:
            sys.modules["pynvml"] = original
        else:
            sys.modules.pop("pynvml", None)
        importlib.reload(gpu_mod)
