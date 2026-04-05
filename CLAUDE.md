# dwail_workstations

A vLLM-based wrapper for running LLMs across two dual-GPU workstations connected via a ZeroTier VPN.

## Project Context

- **Target hardware**: Two workstations, each with dual GPUs, reachable over a ZeroTier VPN
- **Inference backend**: vLLM
- **Goal**: Expose a unified interface for LLM inference that abstracts over the two workstations

## Repository Layout (evolving)

```
dwail_workstations/
├── CLAUDE.md           # This file
├── SDD.md              # Software Design Document
└── ...
```

## Development Notes

- The client machine (this one) is a Mac on the same ZeroTier network
- Workstations are accessed over ZeroTier VPN IPs
- vLLM runs on the workstations, not the client

## Commands

```bash
# Install all workspace dependencies (must use --all-packages)
uv sync --all-packages

# Level 0 — mocked unit tests (always run)
uv run pytest packages/ tests/ -v

# Level 2 — one real workstation
DWAIL_WS1_IP=10.x.x.x uv run pytest -m workstation1 -v

# Level 3 — two workstations, distributed
DWAIL_WS1_IP=10.x.x.x DWAIL_WS2_IP=10.x.x.y uv run pytest -m distributed -v

# Level 2/3 with a large pre-downloaded model instead of opt-125m
DWAIL_WS1_IP=10.x.x.x DWAIL_TEST_MODEL=meta-llama/Llama-3.1-70B-Instruct uv run pytest -m workstation1 -v
```
