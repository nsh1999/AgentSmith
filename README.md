# AgentSmith

Local LLM experiments with an interactive front end: ask typed questions about a
support ticket, have several local models answer them, and aggregate the answers
into a single decision.

## What's here

| Path | Purpose |
| --- | --- |
| `decision_gui.py` | Interactive Gradio app: ticket, question builder, model picker, live results. |
| `decision_maker.py` | Shared engine: runs models through TypeSafe `system_one`, aggregates answers, renders HTML. |
| `jev_clone_demo.py` | Script version of the same flow, five models, results printed to the console. |
| `tests/` | Network-free pytest suite for the engine and the GUI helpers. |
| `smith/` | Standalone LangGraph starter project (its own `pyproject.toml` and venv). |
| `sandbox/` | Local scratch space for older demos and notebooks. Gitignored; not part of the project. |

The decision engine turns each question into a typed answer:

| Type | Asks | Aggregate |
| --- | --- | --- |
| `Choice` | Pick one label from a list | Label agreed by at least 2/3 of the models |
| `Noul` | Yes/no with a probability | Mean probability, `Yes` at 0.5 or above |
| `Score` | Rate on an ordered scale | Mean score mapped back to the scale |

## Requirements

- Python 3.14+
- A local model server that speaks the TypeSafe `system_one` API:
  - [Ollama](https://ollama.com) — default `http://localhost:11434`
  - [LM Studio](https://lmstudio.ai) — default `http://localhost:1234`
- Apple Silicon for the MLX-based demos in `sandbox/`.

## Setup

```bash
uv sync --extra dev          # preferred; uv.lock is the source of truth
```

Without `uv`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Interactive decision GUI

```bash
python decision_gui.py       # opens http://127.0.0.1:7860
```

1. Type or paste a **ticket** in the left panel.
2. Edit the **questions** table — one row per question: `name`, `type`
   (`choice` / `noul` / `score`), `instructions`, and `criteria`. Use
   **Add question** / **Remove last question** to change the list, and
   **Reset to demo questions** to restore the seeded example.
3. Pick **models** on the right (the dropdown fills in from the local server on
   load; **Refresh models** re-queries it, and a built-in list is used if
   discovery fails).
4. Press **Run**.

Results appear while the run is in progress: each finished model adds a column
to the result table, the status table shows `queued` → `running…` → `done` with
per-model runtime, and the summary line re-renders the aggregate after every
answer. **Stop** ends the run once the in-flight model returns.

Connection settings live in the **Connection** accordion and default to
`TYPESAFE_BASE_URL` / `TYPESAFE_API_KEY`, or `http://localhost:11434` and
`"ollamna"`.

### Criteria syntax

| Type | Criteria cell | Example |
| --- | --- | --- |
| `choice` | `label = description`, separated by `;` | `billing = Payments; technical = Bugs` |
| `score` | ordered levels separated by `,`, low to high | `Routine, Soon, Urgent` |
| `noul` | blank, or `true = ...; false = ...` | _(blank)_ |

A choice question needs at least two labels; a score question needs at least two
levels. Names must be identifiers, unique, and not `models`.

## Using the engine from code

```python
from typesafe_sdk import Choice, Noul, Score
from decision_maker import decision_maker

questions = {
    "team": Choice(
        instructions="Which team should handle this ticket?",
        criteria={"billing": "Payments and refunds", "technical": "Bugs", "other": "None"},
    ),
    "refund": Noul(instructions="Does the customer explicitly ask for a refund?"),
    "urgency": Score(
        instructions="How urgent is this ticket?",
        criteria=["Routine", "Soon", "Urgent"],
    ),
}
result_set = decision_maker(["nimble-v3:9b-nvfp4", "tev1:4b-mxfp8"], questions, ticket)
print(result_set["team"]["majority"])
```

`stream_decisions(...)` exposes the same run as a generator so a caller can show
each model's answer as it arrives; `decision_maker(...)` drains it and returns
the final result set. `resutlt_set_to_html(...)` renders a result set as an HTML
page (`result_set_to_html` is an alias; pass `escape=True` for user-supplied
text).

## Tests

```bash
.venv/bin/python -m pytest tests/ -q
.venv/bin/ruff check decision_maker.py decision_gui.py tests
```

The suite never touches the network — it swaps in a fake TypeSafe client and a
fake HTTP fetcher. Pytest is configured to collect `tests/` only, so the demo
scripts are never imported.

## Notes

- API keys are local placeholders (`"ollamna"`, `"lmstudio"`, `"local"`); nothing
  here calls a paid API.
- Model names must exist on your local server. The dropdown shows what the
  server reports; the fallback list is only a guess.
- A `choice` majority needs `len × 2/3` agreement, so five models need four
  agreeing. When no label reaches that, the summary says "no majority".
- `sandbox/` is ignored by git and may hold scratch copies of older demos.

## License

MIT
