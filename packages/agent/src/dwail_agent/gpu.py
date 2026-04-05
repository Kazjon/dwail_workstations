"""Query GPU info via pynvml. Returns empty list on non-NVIDIA machines."""

from __future__ import annotations


def get_gpu_info() -> list[dict]:
    """Return a list of GPU info dicts (index, name, vram_total_mb, vram_free_mb).
    Returns [] if pynvml is unavailable or no NVIDIA driver is present."""
    try:
        import pynvml
    except (ImportError, ModuleNotFoundError):
        return []

    try:
        pynvml.nvmlInit()
    except Exception:
        return []

    try:
        count = pynvml.nvmlDeviceGetCount()
        gpus = []
        for i in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            name = pynvml.nvmlDeviceGetName(handle)
            gpus.append({
                "index": i,
                "name": name if isinstance(name, str) else name.decode(),
                "vram_total_mb": mem.total // (1024 * 1024),
                "vram_free_mb": mem.free // (1024 * 1024),
            })
        return gpus
    finally:
        pynvml.nvmlShutdown()
