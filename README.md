# Aexoryn Agent

**Autonomous Multi-Model Coding Agent — OpenCode-Inspired, Terminal-Native**

> Copyright (c) 2026 Irfan Syazwan / Aexoryn — MIT License

---

## Architecture Flow

```mermaid
flowchart TD
    USER([👤 User Task]) -->|CLI / Terminal| AGENT[🤖 Aexoryn Agent\nmain.py]

    AGENT --> PLANNER[📋 Task Planner\ncore/planner.py]
    PLANNER -->|Decompose into steps| PLAN[Execution Plan\nREAD / WRITE / RUN / PATCH]

    PLAN -->|READ steps| FST[📁 FileSystemTool\ncore/tools.py]
    FST -->|File context| PLAN

    PLAN -->|Hybrid Router| ROUTER{🔀 Route Decision\ncore/registry.py}
    ROUTER -->|≤ 2K tokens\nno force tags| LOCAL[🏠 Local Ollama\nLlama / Qwen / Mistral]
    ROUTER -->|> 2K tokens\nor force tags| CLOUD[☁️ Cloud Registry\nvia OpenRouter]

    LOCAL & CLOUD --> JURY[⚖️ Jury — Parallel Inference\nasyncio.gather]

    subgraph JURY POOL
        GPT[GPT-4o / GPT-4.1]
        CLAUDE[Claude 3.7 / 4.7]
        GEMINI[Gemini 2.5 Pro]
        DEEPSEEK[DeepSeek R2]
        LLAMA[Llama 3.3 Local]
        QWEN[Qwen2.5 Coder Local]
    end

    JURY --> JUDGE[🧠 Super-Judge\nConsensus Engine\ncore/consensus.py]
    JUDGE -->|Conflict Detection\n+ Synthesis| MASTER[✅ Master Solution]

    MASTER --> EXEC[⚡ Executor\nWRITE / PATCH / RUN_COMMAND]
    EXEC --> FST2[📁 FileSystemTool\nWrite to Workspace]
    EXEC --> TERM[💻 TerminalTool\nnpm install / pytest / etc.]
    TERM -->|Error?| SELFCORRECT[🔄 Self-Correct Loop\nAsk Judge for fix → Retry]
    SELFCORRECT --> TERM

    FST2 & TERM --> UI[🖥️ Rich Terminal UI\nui/terminal.py]
    UI -->|Diffs / Summary\nToken Bar / MCP Status| USER
```

---

## Features

| Feature | Description |
|---|---|
| **Multi-Model Jury** | Parallel inference across GPT, Claude, Gemini, DeepSeek, Llama, Qwen |
| **Super-Judge Consensus** | A designated judge model synthesises all jury outputs into one Master Solution |
| **Hybrid Routing** | Routes simple tasks to local Ollama; complex tasks to cloud models |
| **Plug-and-Play Registry** | Add any model in `model_config.yaml` — no code changes needed |
| **MCP-Style Tools** | Create, read, edit, delete files & run terminal commands autonomously |
| **Self-Correction** | Failed commands are sent back to the judge model for auto-fix and retry |
| **Rich Terminal UI** | Live spinners, jury tables, visual diffs, token progress bar |

---

## Quick Start

```bash
# 1. Clone & install
git clone https://github.com/your-org/aexoryn-agent
cd aexoryn-agent
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# Edit .env → add your OPENROUTER_API_KEY

# 3. Run  (note: 'run' subcommand is required)
python main.py run "Refactor src/auth.py to use JWT and add refresh token support"

# 4. Options
python main.py run --workspace ./my-project "Add dark mode to App.tsx"
python main.py run --mode majority_vote "Fix failing unit tests"
python main.py list-models
python main.py reload-config
python main.py --help
python main.py run --help
```

---

## Adding a New Model

Open `model_config.yaml` and add an entry under `models:`:

```yaml
- id: gpt-5.2-preview          # Unique ID used internally
  name: GPT-5.2 Preview         # Display name
  provider: openrouter          # openrouter | litellm | ollama
  model_path: openai/gpt-5.2-preview   # Provider model string
  tier: cloud                   # cloud | local
  enabled: true                 # Toggle without removing the entry
  weight: 1.8                   # Jury consensus weight (0.0 – 2.0)
  strengths: [reasoning, architecture, meta_cognition]
  max_tokens: 32768
  temperature: 0.1
```

Then hot-reload:

```bash
python main.py --reload-config "your task"
# or
python main.py reload-config
```

No code changes required. The registry loads the YAML at runtime.

---

## Project Structure

```
aexoryn-agent/
├── main.py                  # CLI entry point (Typer)
├── model_config.yaml        # Plug-and-Play model registry
├── requirements.txt
├── .env.example
├── LICENSE                  # MIT 2026
│
├── core/
│   ├── registry.py          # Model Registry + hybrid router + client builder
│   ├── consensus.py         # Super-Judge / majority_vote / weighted_rank
│   ├── planner.py           # Task planner + execution loop
│   └── tools.py             # FileSystemTool + TerminalTool + DiffTool
│
└── ui/
    └── terminal.py          # Rich terminal UI (banner, logs, diffs, summary)
```

---

## Consensus Modes

| Mode | Description |
|---|---|
| `super_judge` | All jury responses are fed to the judge model, which writes the final Master Solution. Highest quality. |
| `majority_vote` | Most common response snippet wins. Fast and cost-efficient. |
| `weighted_rank` | Highest-weight model's response is returned directly. Fastest. |

Change mode in `model_config.yaml` under `consensus.mode`, or override per-run:

```bash
python main.py --mode majority_vote "task"
```

---

## Hybrid Routing

```yaml
hybrid_routing:
  local_threshold_tokens: 2000       # Tasks under 2K tokens → Ollama
  cloud_force_tags: [reasoning, architecture, security, consensus]
```

Tasks tagged with force tags **always** go to cloud, regardless of token count.
Local routing requires at least one enabled `tier: local` model.

---

## Environment Variables

| Variable | Description |
|---|---|
| `OPENROUTER_API_KEY` | Required for cloud models via OpenRouter |
| `OLLAMA_BASE_URL` | Ollama endpoint (default: `http://localhost:11434`) |
| `AEXORYN_WORKSPACE` | Default workspace path |
| `AEXORYN_LOG_LEVEL` | `DEBUG` / `INFO` / `WARNING` |
| `AEXORYN_SESSION_LOG` | Log file path |

---

## License

MIT License — Copyright (c) 2026 Irfan Syazwan / Aexoryn.
See [LICENSE](LICENSE) for full text.
