# AgentSmith

Small experiments with **local LLM agents**: tool-calling graphs, structured
("typed") decisions from a single forward pass, and MLX-backed local inference.
Each script is a standalone demo pointed at a local model server — there is no
application code yet.

## Requirements

- Python 3.14+
- A local model server, depending on the script (see the table below):
  - [Ollama](https://ollama.com) — default `http://localhost:11434`
  - [LM Studio](https://lmstudio.ai) — default `http://localhost:1234`
  - a SemIf-compatible endpoint — `http://127.0.0.1:1234`
- Apple Silicon for the MLX-based scripts (`laya-mlx`, `mlx`).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Scripts

| Script | What it shows | Needs |
| --- | --- | --- |
| `graph_001.py` | LangGraph agent with tool calling (`add` / `multiply` / `divide`) over `qwen3.8:27b-nvfp4`; renders the graph, then runs one arithmetic query. | Ollama, Jupyter/IPython |
| `ollama_demo_001.py` | TypeSafe `system_one` over Ollama: one ticket classified with `Choice` (team), `Noul` (refund?), `Score` (urgency). | Ollama |
| `laya_demo.py` | `laya_mlx` local inference with `choice` / `score` / `noul` questions. | Apple Silicon, model download |
| `laya_demo_002.py` | Minimal `laya_mlx` single-`choice` example. | Apple Silicon, model download |
| `test_001.py` | `lmstudio` Python SDK answering a riddle. | LM Studio |
| `test_002.py` | OpenAI-compatible client against LM Studio, reading `reasoning_content`. | LM Studio |
| `test_003.py` | TypeSafe `Choice` against a local SemIf endpoint. | SemIf endpoint |
| `test_004.py` | TypeSafe `Noul` (graded yes/no) against LM Studio. | LM Studio |
| `test_005.py` | TypeSafe `Choice` + `Noul` in a single call. | SemIf endpoint |

## Notes

- API keys in the scripts are local placeholders (`"lmstudio"`, `"local"`, `"your-key"`) — nothing here calls a paid API.
- The `test_00*.py` files are manual demos that print results; they are not pytest tests.
- Model names are hardcoded per script and must exist on your local server.

## License

MIT
