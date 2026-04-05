# dwail_workstations

A vLLM-based system for running LLM inference across multiple GPU workstations connected via a ZeroTier VPN, with a minimal localhost UI for model management and copy-paste code snippets.

## Overview

```
[Client Mac]  ← localhost UI at :8080
     |
  ZeroTier VPN
     |
     +── [Workstation A]  2× RTX 3090  ← dwail-agent on :8765
     +── [Workstation B]  2× RTX 3090  ← dwail-agent on :8765
```

One model runs at a time. The controller automatically routes to a single workstation (tensor parallel across 2 GPUs) or distributes across both workstations via Ray (pipeline parallel, up to ~96GB VRAM pooled) depending on model size.

## Packages

| Package | Runs on | Purpose |
|---------|---------|---------|
| `dwail-agent` | Each workstation | FastAPI daemon: GPU reporting, vLLM lifecycle, Ray management |
| `dwail-controller` | Client Mac | Routes requests, serves UI, manages workstation registry |
| `dwail-shared` | Both | Pydantic models defining the agent↔controller API contract |

## Workstation Setup

On each workstation (Linux, NVIDIA GPU, CUDA installed):

```bash
pip install vllm ray dwail-agent  # or install from this repo
sudo dwail-agent-install --model-dir /mnt/models --ray-head

# Second workstation connects to the first:
sudo dwail-agent-install --model-dir /mnt/models --ray-worker 10.x.x.x
```

The install script sets up a systemd service (`dwail-agent`) that starts on boot and launches Ray automatically.

## Controller Setup (Mac)

```bash
git clone https://github.com/Kazjon/dwail_workstations
cd dwail_workstations
uv sync --all-packages
dwail-controller  # opens UI at http://localhost:8080
```

## UI

Open `http://localhost:8080` in a browser:

1. **Add workstations** — paste a ZeroTier IP, capabilities are discovered automatically
2. **Load a model** — paste a HuggingFace model ID, the controller picks single vs. distributed mode based on VRAM
3. **Get code snippets** — copy-paste Python or JavaScript with the endpoint URL already filled in
4. **Test** — send a prompt and see streaming output with tokens/sec and time-to-first-token

## Development

```bash
uv sync --all-packages

# Level 0 — mocked unit tests (Mac, no hardware needed)
uv run pytest packages/ tests/ -v

# Level 2 — one real workstation
DWAIL_WS1_IP=10.x.x.x uv run pytest -m workstation1 -v

# Level 3 — two workstations, distributed inference
DWAIL_WS1_IP=10.x.x.x DWAIL_WS2_IP=10.x.x.y uv run pytest -m distributed -v

# Use a specific pre-downloaded model for hardware tests
DWAIL_WS1_IP=10.x.x.x DWAIL_TEST_MODEL=meta-llama/Llama-3.1-70B-Instruct uv run pytest -m workstation1 -v
```

Tests follow red/green TDD. Level 0 tests always run and require no hardware.
