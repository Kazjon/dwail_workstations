// dwail UI — communicates with the controller at localhost:8080

const API = "";  // relative, controller serves this file

let activeModel = null;
let activeEndpoint = null;

// --- Workstations ---

async function loadWorkstations() {
  const res = await fetch(`${API}/workstations`);
  const workstations = await res.json();
  const el = document.getElementById("workstation-list");
  if (workstations.length === 0) {
    el.textContent = "No workstations registered.";
    return;
  }
  el.innerHTML = workstations.map(ws => {
    const s = ws.status;
    const vram = s ? `${s.gpu_info.length}x GPU, ${Math.round(s.total_vram_mb / 1024)}GB total` : "unreachable";
    const state = s ? s.vllm_state : "offline";
    const model = s?.current_model ? ` — ${s.current_model}` : "";
    return `<div><strong>${ws.ip}</strong> [${state}${model}] ${vram}
      <button onclick="removeWorkstation('${ws.id}')">Remove</button></div>`;
  }).join("");
}

async function addWorkstation() {
  const ip = document.getElementById("add-ip").value.trim();
  const errEl = document.getElementById("add-error");
  errEl.textContent = "";
  const res = await fetch(`${API}/workstations`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ip }),
  });
  if (!res.ok) {
    const err = await res.json();
    errEl.textContent = err.detail || "Failed to add workstation.";
    return;
  }
  document.getElementById("add-ip").value = "";
  await loadWorkstations();
}

async function removeWorkstation(id) {
  await fetch(`${API}/workstations/${id}`, { method: "DELETE" });
  await loadWorkstations();
}

// --- Model Loading ---

async function loadModel() {
  const modelId = document.getElementById("model-id").value.trim();
  const statusEl = document.getElementById("load-status");
  const errEl = document.getElementById("load-error");
  errEl.textContent = "";
  statusEl.textContent = "Loading...";

  const res = await fetch(`${API}/models/load`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model_id: modelId }),
  });

  if (!res.ok) {
    const err = await res.json();
    errEl.textContent = err.detail || "Failed to load model.";
    statusEl.textContent = "";
    return;
  }

  const data = await res.json();
  activeModel = modelId;
  const ip = Array.isArray(data.workstations) ? data.workstations[0] : data.workstation;
  activeEndpoint = `http://${ip}:8000/v1`;

  statusEl.textContent = `Mode: ${data.mode} — endpoint: ${activeEndpoint}`;
  renderSnippets(modelId, activeEndpoint);
  document.getElementById("snippets-section").style.display = "";
  document.getElementById("test-section").style.display = "";
  await loadWorkstations();
}

// --- Snippets ---

function renderSnippets(modelId, endpoint) {
  document.getElementById("endpoint-url").textContent = endpoint;

  document.getElementById("py-basic-code").textContent =
`from openai import OpenAI

client = OpenAI(base_url="${endpoint}", api_key="none")

response = client.chat.completions.create(
    model="${modelId}",
    messages=[{"role": "user", "content": "Hello"}],
)
print(response.choices[0].message.content)`;

  document.getElementById("py-async-code").textContent =
`import asyncio
from openai import AsyncOpenAI

client = AsyncOpenAI(base_url="${endpoint}", api_key="none")

async def main():
    response = await client.chat.completions.create(
        model="${modelId}",
        messages=[{"role": "user", "content": "Hello"}],
    )
    print(response.choices[0].message.content)

asyncio.run(main())`;

  document.getElementById("py-stream-code").textContent =
`from openai import OpenAI

client = OpenAI(base_url="${endpoint}", api_key="none")

with client.chat.completions.stream(
    model="${modelId}",
    messages=[{"role": "user", "content": "Hello"}],
) as stream:
    for chunk in stream:
        print(chunk.choices[0].delta.content or "", end="", flush=True)`;

  document.getElementById("js-basic-code").textContent =
`const response = await fetch("${endpoint}/chat/completions", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    model: "${modelId}",
    messages: [{ role: "user", content: "Hello" }],
  }),
});
const data = await response.json();
console.log(data.choices[0].message.content);`;

  document.getElementById("js-stream-code").textContent =
`const response = await fetch("${endpoint}/chat/completions", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    model: "${modelId}",
    messages: [{ role: "user", content: "Hello" }],
    stream: true,
  }),
});
const reader = response.body.getReader();
const decoder = new TextDecoder();
while (true) {
  const { done, value } = await reader.read();
  if (done) break;
  const lines = decoder.decode(value).split("\\n");
  for (const line of lines) {
    if (!line.startsWith("data: ") || line === "data: [DONE]") continue;
    const chunk = JSON.parse(line.slice(6));
    process.stdout.write(chunk.choices[0].delta.content ?? "");
  }
}`;
}

function showTab(name) {
  document.querySelectorAll(".tab").forEach(el => el.classList.remove("active"));
  document.getElementById(name).classList.add("active");
}

// --- Test Panel ---

async function runTest() {
  if (!activeEndpoint || !activeModel) return;
  const prompt = document.getElementById("test-prompt").value.trim();
  const outputEl = document.getElementById("test-output");
  const statsEl = document.getElementById("test-stats");
  outputEl.textContent = "";
  statsEl.textContent = "";

  const startTime = performance.now();
  let firstTokenTime = null;
  let tokenCount = 0;

  const response = await fetch(`${activeEndpoint}/chat/completions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: activeModel,
      messages: [{ role: "user", content: prompt }],
      stream: true,
    }),
  });

  const reader = response.body.getReader();
  const decoder = new TextDecoder();

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    const lines = decoder.decode(value).split("\n");
    for (const line of lines) {
      if (!line.startsWith("data: ") || line === "data: [DONE]") continue;
      try {
        const chunk = JSON.parse(line.slice(6));
        const delta = chunk.choices[0].delta.content ?? "";
        if (delta) {
          if (firstTokenTime === null) firstTokenTime = performance.now();
          tokenCount++;
          outputEl.textContent += delta;
        }
      } catch (_) {}
    }
  }

  const totalMs = performance.now() - startTime;
  const ttft = firstTokenTime ? (firstTokenTime - startTime).toFixed(0) : "—";
  const tps = tokenCount > 0 ? (tokenCount / (totalMs / 1000)).toFixed(1) : "—";
  statsEl.textContent = `TTFT: ${ttft}ms  |  ${tps} tok/s  |  ${tokenCount} tokens  |  ${(totalMs / 1000).toFixed(2)}s total`;
}

// Init
loadWorkstations();
