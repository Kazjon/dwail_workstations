// dwail UI — Alpine.js component
// Served by the controller at localhost:8080; all API calls are relative.

function dwailApp() {
  return {
    // --- Workstation state ---
    workstations: [],
    addIp: "",
    addError: "",

    // --- Model FSM: idle | loading | running ---
    modelState: "idle",
    modelInput: "",
    currentModel: "",
    endpoint: "",
    mode: "",
    supportsChat: false,
    modelError: "",

    // --- Snippets ---
    activeTab: "py-basic",

    // --- Test panel ---
    testPrompt: "",
    testOutput: "",
    testStats: "",
    testRunning: false,

    // Internal
    _pollTimer: null,

    // ------------------------------------------------------------------ init

    init() {
      this.refreshWorkstations();
      setInterval(() => this.refreshWorkstations(), 5000);

      // Re-attach to any model already running when the page loads
      fetch("/models/current").then(r => {
        if (!r.ok) return;
        return r.json();
      }).then(data => {
        if (!data) return;
        this.currentModel = data.model_id;
        this.endpoint     = data.endpoint;
        this.mode         = data.mode;
        this.supportsChat = data.supports_chat;
        if (data.vllm_state === "running") {
          this.modelState = "running";
        } else if (data.vllm_state === "loading") {
          this.modelState = "loading";
          this._startPolling();
        } else if (data.vllm_state === "error") {
          this.modelError = `vLLM failed to start "${data.model_id}". Check workstation logs.`;
        }
      });
    },

    // --------------------------------------------------------- Workstations

    async refreshWorkstations() {
      const r = await fetch("/workstations");
      if (r.ok) this.workstations = await r.json();
    },

    async addWorkstation() {
      this.addError = "";
      const r = await fetch("/workstations", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ip: this.addIp.trim() }),
      });
      if (!r.ok) {
        const err = await r.json();
        this.addError = err.detail || "Failed to add workstation.";
        return;
      }
      this.addIp = "";
      await this.refreshWorkstations();
    },

    async removeWorkstation(id) {
      await fetch(`/workstations/${id}`, { method: "DELETE" });
      await this.refreshWorkstations();
    },

    wsBadgeClass(ws) {
      if (!ws.status) return "offline";
      return ws.status.vllm_state || "offline";
    },

    wsStateLabel(ws) {
      if (!ws.status) return "offline";
      const s = ws.status;
      return s.current_model ? `${s.vllm_state} — ${s.current_model}` : s.vllm_state;
    },

    wsDetail(ws) {
      if (!ws.status) return "unreachable";
      const s = ws.status;
      const gpus = s.gpu_info.length;
      const gb = Math.round(s.total_vram_mb / 1024);
      const freeGb = (s.free_vram_mb / 1024).toFixed(0);
      return `${gpus}× GPU  ${freeGb}/${gb}GB free`;
    },

    // ------------------------------------------------------------- Model FSM

    async loadModel() {
      this.modelError = "";
      this.currentModel = this.modelInput.trim();
      this.modelState = "loading";

      const r = await fetch("/models/load", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model_id: this.currentModel }),
      });

      if (!r.ok) {
        const err = await r.json();
        this.modelError = err.detail || "Failed to load model.";
        this.modelState = "idle";
        return;
      }

      const data = await r.json();
      this.mode = data.mode;
      this._startPolling();
    },

    async stopModel() {
      this._stopPolling();
      const r = await fetch("/models/stop", { method: "POST" });
      if (!r.ok && r.status !== 404) {
        const err = await r.json();
        this.modelError = err.detail || "Failed to stop model.";
        return;
      }
      this.modelState = "idle";
      this.currentModel = "";
      this.endpoint = "";
      this.supportsChat = false;
      this.modelError = "";
      await this.refreshWorkstations();
    },

    _startPolling() {
      this._stopPolling();
      this._pollTimer = setInterval(() => this._pollCurrent(), 3000);
    },

    _stopPolling() {
      if (this._pollTimer) {
        clearInterval(this._pollTimer);
        this._pollTimer = null;
      }
    },

    async _pollCurrent() {
      const r = await fetch("/models/current");
      if (!r.ok) return; // 404 = nothing active yet, keep polling
      const data = await r.json();
      if (data.vllm_state === "running") {
        this._stopPolling();
        this.endpoint     = data.endpoint;
        this.supportsChat = data.supports_chat;
        this.mode         = data.mode;
        this.modelState   = "running";
        await this.refreshWorkstations();
      } else if (data.vllm_state === "error") {
        this._stopPolling();
        this.modelError = `vLLM failed to start "${data.model_id}". Check workstation logs.`;
        this.modelState = "idle";
        await this.refreshWorkstations();
      }
    },

    // ------------------------------------------------------------ Snippets

    snippet() {
      const m = this.currentModel;
      const e = this.endpoint;
      const chatUrl   = `${e}/chat/completions`;
      const compUrl   = `${e}/completions`;
      const chat = this.supportsChat;

      const snippets = {
        "py-basic": chat
          ? `from openai import OpenAI\n\nclient = OpenAI(base_url="${e}", api_key="none")\n\nresponse = client.chat.completions.create(\n    model="${m}",\n    messages=[{"role": "user", "content": "Hello"}],\n)\nprint(response.choices[0].message.content)`
          : `from openai import OpenAI\n\nclient = OpenAI(base_url="${e}", api_key="none")\n\nresponse = client.completions.create(\n    model="${m}",\n    prompt="Hello",\n)\nprint(response.choices[0].text)`,

        "py-async": chat
          ? `import asyncio\nfrom openai import AsyncOpenAI\n\nclient = AsyncOpenAI(base_url="${e}", api_key="none")\n\nasync def main():\n    response = await client.chat.completions.create(\n        model="${m}",\n        messages=[{"role": "user", "content": "Hello"}],\n    )\n    print(response.choices[0].message.content)\n\nasyncio.run(main())`
          : `import asyncio\nfrom openai import AsyncOpenAI\n\nclient = AsyncOpenAI(base_url="${e}", api_key="none")\n\nasync def main():\n    response = await client.completions.create(\n        model="${m}",\n        prompt="Hello",\n    )\n    print(response.choices[0].text)\n\nasyncio.run(main())`,

        "py-stream": chat
          ? `from openai import OpenAI\n\nclient = OpenAI(base_url="${e}", api_key="none")\n\nwith client.chat.completions.stream(\n    model="${m}",\n    messages=[{"role": "user", "content": "Hello"}],\n) as stream:\n    for chunk in stream:\n        print(chunk.choices[0].delta.content or "", end="", flush=True)`
          : `from openai import OpenAI\n\nclient = OpenAI(base_url="${e}", api_key="none")\n\nfor chunk in client.completions.create(\n    model="${m}",\n    prompt="Hello",\n    stream=True,\n):\n    print(chunk.choices[0].text or "", end="", flush=True)`,

        "js-basic": chat
          ? `const response = await fetch("${chatUrl}", {\n  method: "POST",\n  headers: { "Content-Type": "application/json" },\n  body: JSON.stringify({\n    model: "${m}",\n    messages: [{ role: "user", content: "Hello" }],\n  }),\n});\nconst data = await response.json();\nconsole.log(data.choices[0].message.content);`
          : `const response = await fetch("${compUrl}", {\n  method: "POST",\n  headers: { "Content-Type": "application/json" },\n  body: JSON.stringify({\n    model: "${m}",\n    prompt: "Hello",\n  }),\n});\nconst data = await response.json();\nconsole.log(data.choices[0].text);`,

        "js-stream": chat
          ? `const response = await fetch("${chatUrl}", {\n  method: "POST",\n  headers: { "Content-Type": "application/json" },\n  body: JSON.stringify({\n    model: "${m}",\n    messages: [{ role: "user", content: "Hello" }],\n    stream: true,\n  }),\n});\nconst reader = response.body.getReader();\nconst decoder = new TextDecoder();\nwhile (true) {\n  const { done, value } = await reader.read();\n  if (done) break;\n  for (const line of decoder.decode(value).split("\\n")) {\n    if (!line.startsWith("data: ") || line === "data: [DONE]") continue;\n    const chunk = JSON.parse(line.slice(6));\n    process.stdout.write(chunk.choices[0].delta.content ?? "");\n  }\n}`
          : `const response = await fetch("${compUrl}", {\n  method: "POST",\n  headers: { "Content-Type": "application/json" },\n  body: JSON.stringify({\n    model: "${m}",\n    prompt: "Hello",\n    stream: true,\n  }),\n});\nconst reader = response.body.getReader();\nconst decoder = new TextDecoder();\nwhile (true) {\n  const { done, value } = await reader.read();\n  if (done) break;\n  for (const line of decoder.decode(value).split("\\n")) {\n    if (!line.startsWith("data: ") || line === "data: [DONE]") continue;\n    const chunk = JSON.parse(line.slice(6));\n    process.stdout.write(chunk.choices[0].text ?? "");\n  }\n}`,
      };

      return snippets[this.activeTab] || "";
    },

    async copySnippet() {
      await navigator.clipboard.writeText(this.snippet());
    },

    // ---------------------------------------------------------- Test panel

    async runTest() {
      if (this.testRunning) return;
      this.testRunning = true;
      this.testOutput  = "";
      this.testStats   = "";

      const startTime = performance.now();
      let firstTokenTime = null;
      let tokenCount = 0;

      try {
        const body = this.supportsChat
          ? { model: this.currentModel, messages: [{ role: "user", content: this.testPrompt }], stream: true }
          : { model: this.currentModel, prompt: this.testPrompt, stream: true };

        const url = this.supportsChat
          ? `${this.endpoint}/chat/completions`
          : `${this.endpoint}/completions`;

        const response = await fetch(url, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        });

        const reader  = response.body.getReader();
        const decoder = new TextDecoder();

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          for (const line of decoder.decode(value).split("\n")) {
            if (!line.startsWith("data: ") || line === "data: [DONE]") continue;
            try {
              const chunk = JSON.parse(line.slice(6));
              const delta = this.supportsChat
                ? (chunk.choices[0].delta.content ?? "")
                : (chunk.choices[0].text ?? "");
              if (delta) {
                if (firstTokenTime === null) firstTokenTime = performance.now();
                tokenCount++;
                this.testOutput += delta;
              }
            } catch (_) {}
          }
        }
      } finally {
        this.testRunning = false;
        const totalMs = performance.now() - startTime;
        const ttft = firstTokenTime ? `${(firstTokenTime - startTime).toFixed(0)}ms` : "—";
        const tps  = tokenCount > 0 ? `${(tokenCount / (totalMs / 1000)).toFixed(1)} tok/s` : "—";
        this.testStats = `TTFT: ${ttft}  |  ${tps}  |  ${tokenCount} tokens  |  ${(totalMs / 1000).toFixed(2)}s`;
      }
    },
  };
}
