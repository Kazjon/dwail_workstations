# Software Design Document: dwail_workstations

**Status**: Draft  
**Last updated**: 2026-04-07

---

## 1. Overview

`dwail_workstations` manages vLLM inference across two dual-GPU workstations on a ZeroTier VPN. One model runs at a time, potentially spanning all 4 GPUs. A minimal localhost web UI lets the user specify a model and get copy-paste Python/JS code to use that model's endpoint in any project.

---

## 2. Hardware Topology

```
[Client Mac]  <-- localhost UI + code snippets
     |
  ZeroTier VPN
     |
     +-- [Workstation A] (2x RTX 3090, 24GB each)
     +-- [Workstation B] (2x RTX 3090, 24GB each)
```

- 4x RTX 3090 = 96GB VRAM total
- One model active at a time; may be loaded on one or both workstations depending on size
- Workstations have static ZeroTier IPs
- Designed to add more workstations later

---

## 3. Goals

- Run a single vLLM model at a time across available GPU hardware
- Provide a minimal localhost UI: paste a model ID → get working Python/JS code snippets
- Support models that fit on 1 workstation (≤48GB) or need both (≤96GB, via pipeline parallel)
- Keep it simple now; design for future extensibility (more workstations, model switching UI)

---

## 4. Architecture

### Components

**Workstation Agent** (runs on each workstation, always-on daemon):
- Small FastAPI service (e.g. port 8765)
- Reports: GPU count, VRAM per GPU, CUDA version
- Reports: vLLM status (idle / loading / running + current model)
- Lists models available on disk (scans configured model directory)
- Accepts start/stop vLLM commands with a given model ID
- This is the only thing that needs to be set up manually on each workstation

**Mac Controller** (runs on the client Mac):
- FastAPI server on localhost
- Maintains a config file of known workstations (ZeroTier IP + agent port)
- Polls agents for status; discovers capabilities when a workstation is added
- Serves the static UI
- Handles model load decisions: which workstation(s) to use, whether to use distributed mode
- Proxies or redirects inference traffic to the appropriate vLLM endpoint

**vLLM** (on workstations, managed by agent):
- Single-workstation mode: `--tensor-parallel-size 2` (both GPUs on one machine)
- Distributed mode: Ray cluster across workstations, `--tensor-parallel-size 2 --pipeline-parallel-size 2`
- Exposes OpenAI-compatible API

### Why distributed now
vLLM + Ray is the canonical way to run a single model across multiple nodes — ollama cannot do this. The main benefit is VRAM pooling (96GB total), not throughput. Pipeline parallelism over ZeroTier (~1-5ms latency) adds inter-stage overhead that reduces tokens/sec vs. local NVLink, but the system still works and fits larger models.

### Single-workstation mode (models ≤ ~48GB VRAM)
- Agent starts vLLM with `--tensor-parallel-size 2` on one workstation
- Inference endpoint: `http://<workstation-zt-ip>:8000/v1`

### Distributed mode (models ≤ ~96GB VRAM)
- Agent on workstation A starts Ray head + vLLM head
- Agent on workstation B starts Ray worker + vLLM worker
- Head node coordinates; inference endpoint still on workstation A
- Controller selects this mode automatically based on model size vs. available VRAM

---

## 5. UI Design (localhost web page)

Single-page app served by the Mac controller. Two sections:

### Workstation Management
- List of known workstations with live status (idle / running model X / offline)
- GPU count + VRAM shown per workstation
- **Add Workstation**: paste ZeroTier IP → controller pings agent → capabilities auto-discovered
- **Remove Workstation**: removes from config

**Setup flow for a new workstation:**
1. SSH into the machine, run the agent install script (one command)
2. Open the UI, click Add Workstation, paste the ZeroTier IP
3. Done — capabilities appear automatically

### Model Loader + Code Snippets
```
[ Model ID (e.g. meta-llama/Llama-3.1-70B-Instruct): _______ ] [Load Model]

Status: Loading on workstation A + B (distributed mode, ~70GB)...

Endpoint: http://10.x.x.x:8000/v1

--- Python ---
from openai import OpenAI
client = OpenAI(base_url="http://10.x.x.x:8000/v1", api_key="none")
response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-70B-Instruct",
    messages=[{"role": "user", "content": "Hello"}]
)

--- JavaScript ---
const res = await fetch("http://10.x.x.x:8000/v1/chat/completions", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    model: "meta-llama/Llama-3.1-70B-Instruct",
    messages: [{ role: "user", content: "Hello" }]
  })
})
```

- Snippets are **write-once templates** with madlibs-style substitution (endpoint URL, model name injected at generation time)
- Tabs: Basic · Async · Streaming — for both Python and JavaScript
- Snippets are copy-paste ready with the actual endpoint URL and model name filled in
- Controller auto-selects single vs. distributed mode based on model size estimate vs. available VRAM

### In-Browser Test Panel
Below the snippets, a test panel:
- Text input for a prompt
- Submit button → sends streamed request to the active endpoint
- Response streams into a text area in real time
- Readout shows: time-to-first-token, tokens/sec, total tokens, total time
- Useful for benchmarking latency across single vs. distributed configurations

---

## 6. Decisions Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-03-31 | One model at a time | Simplicity; 3090s have enough VRAM for large models if pooled |
| 2026-03-31 | 2x RTX 3090 per workstation (2 workstations initially) | Current hardware; more workstations added later |
| 2026-03-31 | Client UI = static localhost page, output = code snippets | Simplest useful interface for personal use |
| 2026-03-31 | OpenAI-compatible API via vLLM | Maximizes compatibility with existing tools/SDKs |
| 2026-03-31 | Thin FastAPI controller on Mac | Serves UI + knows workstation topology |
| 2026-04-05 | UI includes "Load Model" button | User wants model management from the UI |
| 2026-04-05 | Distributed inference (Ray + pipeline parallel) in scope | Main advantage over ollama; VRAM pooling for large models |
| 2026-04-05 | Workstations added via UI (paste IP) | Agent on workstation handles discovery; no config file editing |
| 2026-04-05 | Workstation agent = always-on FastAPI daemon | Minimal setup; reports capabilities + manages vLLM lifecycle |
| 2026-04-05 | Code snippets: Basic + Async + Streaming tabs, Python + JS | Write-once templates with madlibs substitution |
| 2026-04-05 | In-browser test panel with speed readout | Benchmark latency; useful for single vs. distributed comparison |
| 2026-04-05 | VRAM estimation via HF config.json (params × dtype × 1.2) | Reliable structured data; always catch vLLM OOM as fallback |
| 2026-04-05 | Fixed model directory on workstations (configurable at agent install) | Simplicity; no per-workstation UI config needed |
| 2026-04-05 | Agent install: pip + systemd | Simpler than Docker for dedicated GPU machines; no container GPU passthrough friction |
| 2026-04-05 | Ray starts with the agent; persistent cluster | Workstations are dedicated; ~200-400MB idle overhead is acceptable |
| 2026-04-07 | `POST /models/stop` broadcasts stop to all active workstations | Needed for model switching via UI; tolerates agent unreachability |
| 2026-04-07 | Model capability detection via HF tokenizer_config.json + name heuristics | Determines chat vs. base model; affects code snippet format |
| 2026-04-07 | WorkstationStatus VRAM totals as computed_field | Plain @property not serialized by Pydantic v2; computed_field required for JSON output |
| 2026-04-07 | /models/current includes error state | Polling loop would run forever on vLLM failure without it; UI needs a terminal state to stop on |

---

## 7. Open Questions

- **UI testing**: pytest + vitest for JS, or skip JS tests in v1?

---

## 8. Known Gaps / Return-To Items

These are intentionally deferred — record here so they're easy to pick up after a break.

| Item | Status | Notes |
|------|--------|-------|
| Level 3 distributed tests | **Blocked** | Both WS1 and WS2 have disk space issues; needs cleanup before large model downloads possible |
| Large model testing | **Blocked** | Disk space on both workstations exhausted; opt-125m is the only tested model |
| Multi-workstation distributed inference | **Blocked** | WS2 disk space issue; Ray cluster across nodes untested end-to-end |
| UI wiring | **Complete** | Alpine.js FSM (idle/loading/running), polls /models/current, chat/base snippet switching, test panel |
| HF token threading | **Not started** | Gated models (Gemma 4, Llama 3) require an HF token on both the workstation (for download/serving) and the Mac controller (for capability detection via hf_hub_download). Goal: user provides token once via the UI or controller config; controller passes it to agents so workstations never need manual ssh intervention after initial setup. |
| vLLM version management | **Not started** | vLLM must support the model architecture (e.g. Gemma4ForCausalLM). Currently requires manual `pip install --upgrade vllm` on each workstation. Goal: controller should surface the installed vLLM version per workstation and warn when a requested model's architecture may not be supported. |

---

## 8. Repository Structure

Monorepo with `uv` workspaces:

```
dwail_workstations/
├── packages/
│   ├── agent/          # FastAPI daemon on workstations
│   │   ├── src/
│   │   ├── tests/
│   │   └── pyproject.toml
│   ├── controller/     # FastAPI server on Mac
│   │   ├── src/
│   │   ├── tests/
│   │   └── pyproject.toml
│   ├── ui/             # Static frontend
│   │   └── src/
│   └── shared/         # Pydantic models shared by agent + controller
│       ├── src/
│       ├── tests/
│       └── pyproject.toml
├── pyproject.toml      # uv workspace root
├── CLAUDE.md
└── SDD.md
```

**Development approach**: red/green TDD throughout — tests written before implementation.

---

## 8. Out of Scope

- Training or fine-tuning
- Model storage/registry (models assumed already on disk on workstations)
- Auth (private ZeroTier network is sufficient)
- Multi-user / multi-client
