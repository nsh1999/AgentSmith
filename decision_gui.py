"""Interactive Gradio GUI for the multi-model ticket decision engine.

Type a ticket, describe the questions to ask about it, pick models, and watch each
model's answer and the running aggregate appear as the run progresses.

Run with ``python decision_gui.py`` and open http://127.0.0.1:7860.
"""

import html as html_lib
import json
import os
import re
import threading
import uuid
from urllib.request import Request, urlopen

import gradio as gr
from typesafe_sdk import (
    Choice,
    Noul,
    RetryPolicy,
    Score,
    TypeSafeClient,
    TypeSafeError,
)

from decision_maker import resutlt_set_to_html, stream_decisions

DEFAULT_BASE_URL = os.environ.get("TYPESAFE_BASE_URL", "http://localhost:11434")
DEFAULT_API_KEY = os.environ.get("TYPESAFE_API_KEY", "ollamna")
DEFAULT_TICKET = "I was charged twice. Please refund the extra payment. this is urgent for me."

# Used when every discovery tier fails, so the app stays usable offline.
FALLBACK_MODELS = (
    "clef-flash:9b-mxfp8",
    "kev:9b-nvfp4",
    "nimble-v3:9b-nvfp4",
    "strands-decider:2b-mlx-bf16",
    "tev1:4b-mxfp8",
)

QUESTION_COLUMNS = ("name", "type", "instructions", "criteria")
DEMO_QUESTION_ROWS = [
    [
        "team",
        "choice",
        "Which team should handle this ticket?",
        "billing = Payments and refunds; technical = Bugs and integrations; other = None of the above",
    ],
    ["refund", "noul", "Does the customer explicitly ask for a refund?", ""],
    ["urgency", "score", "How urgent is this ticket?", "Routine, Soon, Urgent"],
]

CRITERIA_HELP = """
**Criteria syntax** — separate items with `;` or a newline.

- `choice`: `billing = Payments and refunds; technical = Bugs` (description optional, 2+ labels)
- `score`: `Routine, Soon, Urgent` (comma-separated, 2+ levels, low to high)
- `noul`: leave blank
"""

QUESTION_TYPES = {"choice", "noul", "score"}
RESERVED_QUESTION_NAMES = {"models"}
QUESTION_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

CSS = """
.decision-results table { border-collapse: collapse; width: 100%; font-size: 0.9rem; }
.decision-results th, .decision-results td { border: 1px solid #ccc; padding: 4px 8px; text-align: left; }
.decision-results h2 { font-size: 1.05rem; margin: 0.8rem 0 0.2rem; }
"""

# Live runs, keyed by the id held in each session's gr.State. A threading.Event
# cannot live inside gr.State because Gradio deep-copies state values.
_RUNS: dict[str, threading.Event] = {}
_RUNS_LOCK = threading.Lock()


class QuestionValidationError(ValueError):
    """Raised when GUI inputs cannot be turned into a runnable request."""


# --- question building ------------------------------------------------------


def parse_choice_criteria(text: str) -> dict[str, str | None]:
    """Parse ``label = description; label2`` into an ordered criteria mapping."""
    criteria: dict[str, str | None] = {}
    for item in re.split(r"[;\n]", text or ""):
        item = item.strip()
        if not item:
            continue
        label, _, description = item.partition("=")
        label, description = label.strip(), description.strip()
        if not label:
            raise QuestionValidationError("a choice criteria item has an empty label")
        if label in criteria:
            raise QuestionValidationError(f"duplicate choice label {label!r}")
        criteria[label] = description or None
    return criteria


def parse_score_criteria(text: str) -> list[str]:
    """Parse ``Routine, Soon, Urgent`` into an ordered list of levels."""
    return [item.strip() for item in re.split(r"[,;\n]", text or "") if item.strip()]


def parse_noul_criteria(text: str) -> dict[str, str] | None:
    """Parse optional ``true = ...; false = ...`` wording for a noul question."""
    criteria: dict[str, str] = {}
    for item in re.split(r"[;\n]", text or ""):
        item = item.strip()
        if not item:
            continue
        key, _, description = item.partition("=")
        key = key.strip().lower()
        if key not in {"true", "false"}:
            raise QuestionValidationError(
                f"noul criteria keys must be 'true' or 'false', got {key!r}"
            )
        criteria[key] = description.strip()
    return criteria or None


def build_questions(rows) -> dict[str, Choice | Noul | Score]:
    """Convert GUI table rows into a questions mapping.

    Blank rows are skipped. All problems are collected and reported together so
    the user can fix the table in one pass.
    """
    questions: dict[str, Choice | Noul | Score] = {}
    problems: list[str] = []

    for index, row in enumerate(rows or [], start=1):
        cells = [str(cell or "").strip() for cell in list(row) + [""] * 4]
        name, question_type, instructions, criteria_text = cells[:4]
        if not any(cells[:4]):
            continue

        label = name or f"row {index}"
        if not name:
            problems.append(f"Question {label!r}: name is required")
            continue
        if not QUESTION_NAME_PATTERN.match(name):
            problems.append(f"Question {label!r}: name must be an identifier (letters, digits, _)")
            continue
        if name in RESERVED_QUESTION_NAMES:
            problems.append(f"Question {label!r}: {name!r} is reserved")
            continue
        if name in questions:
            problems.append(f"Question {label!r}: duplicate name")
            continue
        if not instructions:
            problems.append(f"Question {label!r}: instructions are required")
            continue

        kind = question_type.lower()
        try:
            if kind == "choice":
                criteria = parse_choice_criteria(criteria_text)
                if len(criteria) < 2:
                    raise QuestionValidationError("a choice question needs at least 2 criteria")
                questions[name] = Choice(instructions=instructions, criteria=criteria)
            elif kind == "score":
                levels = parse_score_criteria(criteria_text)
                if len(levels) < 2:
                    raise QuestionValidationError("a score question needs at least 2 levels")
                questions[name] = Score(instructions=instructions, criteria=levels)
            elif kind == "noul":
                questions[name] = Noul(
                    instructions=instructions,
                    criteria=parse_noul_criteria(criteria_text),
                )
            else:
                raise QuestionValidationError(
                    f"unknown type {question_type!r}; use choice, noul or score"
                )
        except QuestionValidationError as error:
            problems.append(f"Question {label!r}: {error}")

    if problems:
        raise QuestionValidationError("\n".join(problems))
    if not questions:
        raise QuestionValidationError("Add at least one question.")
    return questions


def validate_run_inputs(ticket: str, models) -> None:
    """Raise if the run has no ticket text or no selected models."""
    problems = []
    if not (ticket or "").strip():
        problems.append("Enter a ticket to classify.")
    if not models:
        problems.append("Select at least one model.")
    if problems:
        raise QuestionValidationError("\n".join(problems))


# --- model discovery --------------------------------------------------------


def _fetch_json(url: str, api_key: str | None = None, timeout: float = 5.0):
    """GET ``url`` and decode its JSON body."""
    request = Request(url, headers={"Accept": "application/json"})
    if api_key:
        request.add_header("Authorization", f"Bearer {api_key}")
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _parse_model_names(payload) -> list[str]:
    """Extract model names from TypeSafe, OpenAI, or Ollama response shapes."""
    if not isinstance(payload, dict):
        return []
    for field, key in (("models", "name"), ("data", "id")):
        entries = payload.get(field)
        if not isinstance(entries, list):
            continue
        names = [
            entry[key].strip()
            for entry in entries
            if isinstance(entry, dict) and isinstance(entry.get(key), str) and entry[key].strip()
        ]
        if names:
            return list(dict.fromkeys(names))
    return []


def discover_models(
    base_url: str | None,
    api_key: str | None,
    timeout: float = 5.0,
) -> tuple[list[str], str]:
    """List models from the endpoint, falling back through three tiers.

    Returns the names and a human-readable note naming the tier that answered
    (or why every tier failed).
    """
    base = (base_url or DEFAULT_BASE_URL).rstrip("/")
    key = api_key or None
    failures: list[str] = []

    try:
        with TypeSafeClient(
            api_key=key,
            base_url=base,
            timeout=timeout,
            retry=RetryPolicy(max_retries=0),
        ) as client:
            names = [metadata.name for metadata in client.models.list().models]
        if names:
            return names, "TypeSafe `/v1/models`"
        failures.append("SDK: empty model list")
    except (TypeSafeError, OSError, ValueError) as error:
        failures.append(f"SDK: {error}")

    try:
        names = _parse_model_names(_fetch_json(f"{base}/v1/models", key, timeout))
        if names:
            return names, "OpenAI-compatible `/v1/models`"
        failures.append("/v1/models: no models found")
    except (OSError, ValueError) as error:
        failures.append(f"/v1/models: {error}")

    try:
        names = _parse_model_names(_fetch_json(f"{base}/api/tags", key, timeout))
        if names:
            return names, "Ollama `/api/tags`"
        failures.append("/api/tags: no models found")
    except (OSError, ValueError) as error:
        failures.append(f"/api/tags: {error}")

    return list(FALLBACK_MODELS), "hardcoded demo list (" + "; ".join(failures) + ")"


def discover_models_ui(base_url: str, api_key: str):
    """Refresh handler: repopulate the model dropdown and report the source."""
    names, source = discover_models(base_url, api_key)
    preselected = [model for model in FALLBACK_MODELS if model in names] or names[:3]
    return (
        gr.Dropdown(choices=names, value=preselected),
        f"Found **{len(names)}** models via {source}.",
    )


# --- rendering --------------------------------------------------------------


def _strip_page(page: str) -> str:
    """Drop the outer document wrapper that the shared renderer adds."""
    return page.removeprefix("<html><body>").removesuffix("</body></html>")


def results_html(result_set: dict | None) -> str:
    """Render a result-set snapshot as a table block."""
    if not result_set or not any(key != "models" for key in result_set):
        return "<p>No results yet.</p>"
    return (
        "<div class='decision-results'>"
        f"{_strip_page(resutlt_set_to_html(result_set, escape=True))}"
        "</div>"
    )


def summary_markdown(result_set: dict) -> str:
    """Render the aggregate line for every question answered so far."""
    lines = []
    for key, entry in result_set.items():
        if key == "models" or "explained" not in entry:
            continue
        text = entry["explained"]
        if entry["type"] == "choice" and entry.get("majority") is None:
            text += " — no majority"
        lines.append(f"**{entry['name']}** ({entry['type']}): {text}")
    return "\n\n".join(lines) or "_No answers yet._"


def status_rows(status: dict[str, dict]) -> list[list[str]]:
    """Render per-model progress as table rows."""
    return [
        [
            model,
            entry["status"],
            "" if entry["runtime"] is None else f"{entry['runtime']:.2f}",
        ]
        for model, entry in status.items()
    ]


def _error_html(message: str) -> str:
    """Render a message as an escaped error block."""
    escaped = html_lib.escape(message).replace("\n", "<br>")
    return f"<div class='decision-results'><p><strong>Cannot run</strong></p><p>{escaped}</p></div>"


# --- run / stop -------------------------------------------------------------


def stop_run(run_id: str) -> str:
    """Signal the run identified by ``run_id`` to stop after the current model."""
    with _RUNS_LOCK:
        stop_event = _RUNS.get(run_id or "")
    if stop_event is None:
        return "No run in progress."
    stop_event.set()
    return "Stopping after the current model finishes…"


def run_decision(ticket, rows, models, base_url, api_key, timeout, run_id):
    """Run the selected models, yielding a UI update after each step.

    Gradio drives this generator on a worker thread and streams each ``yield``,
    so the browser sees results appear model by model.
    """
    try:
        validate_run_inputs(ticket, models or [])
        questions = build_questions(rows or [])
    except QuestionValidationError as error:
        yield (_error_html(str(error)), [], "", run_id)
        return

    run_key = uuid.uuid4().hex
    stop_event = threading.Event()
    with _RUNS_LOCK:
        _RUNS[run_key] = stop_event

    status = {model: {"status": "queued", "runtime": None} for model in models}
    try:
        yield (
            results_html(None),
            status_rows(status),
            "Starting…",
            run_key,
        )
        for event in stream_decisions(
            models,
            questions,
            (ticket or "").strip(),
            timeout=float(timeout or 120.0),
            retry=RetryPolicy(max_retries=0),
            base_url=base_url or None,
            api_key=api_key or None,
            tolerate_errors=True,
            stop_event=stop_event,
        ):
            kind = event["event"]
            model = event.get("model")
            if kind == "model_start" and model:
                status[model] = {"status": "running…", "runtime": None}
            elif kind == "model_done" and model:
                status[model] = {"status": "done", "runtime": event["runtime"]}
            elif kind == "model_error" and model:
                status[model] = {"status": f"error: {event['error']}", "runtime": event["runtime"]}
            elif kind == "stopped":
                for remaining in event["remaining"]:
                    status[remaining] = {"status": "skipped (stopped)", "runtime": None}

            yield (
                results_html(event["result_set"]),
                status_rows(status),
                summary_markdown(event["result_set"]),
                run_key,
            )
    except TypeSafeError as error:
        yield (
            _error_html(f"{type(error).__name__}: {error}"),
            status_rows(status),
            "Run failed.",
            run_key,
        )
    finally:
        with _RUNS_LOCK:
            _RUNS.pop(run_key, None)


# --- layout -----------------------------------------------------------------


def build_demo() -> gr.Blocks:
    """Build the Gradio interface."""
    with gr.Blocks(title="AgentSmith decision maker") as demo:
        run_id = gr.State("")

        gr.Markdown(
            "# AgentSmith decision maker\n"
            "Ask typed questions about a ticket and watch several local models answer, "
            "with the aggregate updating as each model returns."
        )

        with gr.Row():
            with gr.Column(scale=3):
                ticket = gr.Textbox(label="Ticket", lines=6, value=DEFAULT_TICKET)
                questions = gr.Dataframe(
                    headers=list(QUESTION_COLUMNS),
                    datatype=["str"] * 4,
                    type="array",
                    interactive=True,
                    value=[list(row) for row in DEMO_QUESTION_ROWS],
                    row_count=len(DEMO_QUESTION_ROWS),
                    column_count=4,
                    label="Questions",
                )
                with gr.Row():
                    add_question = gr.Button("Add question")
                    remove_question = gr.Button("Remove last question")
                    reset_questions = gr.Button("Reset to demo questions")
                gr.Markdown(CRITERIA_HELP)

            with gr.Column(scale=2):
                with gr.Accordion("Connection", open=False):
                    base_url = gr.Textbox(label="Base URL", value=DEFAULT_BASE_URL)
                    api_key = gr.Textbox(label="API key", type="password", value=DEFAULT_API_KEY)
                    timeout = gr.Number(label="Per-model timeout (s)", value=120)
                models = gr.Dropdown(
                    label="Models",
                    multiselect=True,
                    choices=list(FALLBACK_MODELS),
                    value=list(FALLBACK_MODELS),
                    allow_custom_value=True,
                )
                with gr.Row():
                    refresh_models = gr.Button("Refresh models")
                    run_button = gr.Button("Run", variant="primary")
                    stop_button = gr.Button("Stop")
                discovery = gr.Markdown("Model discovery has not run yet.")

        with gr.Row():
            with gr.Column(scale=2):
                status = gr.Dataframe(
                    headers=["model", "status", "runtime (s)"],
                    datatype=["str"] * 3,
                    type="array",
                    interactive=False,
                    value=[],
                    label="Per-model status",
                )
                summary = gr.Markdown("")
            with gr.Column(scale=3):
                results = gr.HTML("<p>No results yet.</p>")

        run_button.click(
            run_decision,
            inputs=[ticket, questions, models, base_url, api_key, timeout, run_id],
            outputs=[results, status, summary, run_id],
        )
        stop_button.click(
            stop_run,
            inputs=[run_id],
            outputs=[summary],
            api_visibility="private",
        )
        add_question.click(
            lambda rows: list(rows or []) + [["", "choice", "", ""]],
            inputs=[questions],
            outputs=[questions],
        )
        remove_question.click(
            lambda rows: list(rows or [])[:-1],
            inputs=[questions],
            outputs=[questions],
        )
        reset_questions.click(
            lambda: [list(row) for row in DEMO_QUESTION_ROWS],
            outputs=[questions],
        )
        refresh_models.click(
            discover_models_ui,
            inputs=[base_url, api_key],
            outputs=[models, discovery],
        )
        demo.load(
            discover_models_ui,
            inputs=[base_url, api_key],
            outputs=[models, discovery],
        )

    return demo


def main() -> None:
    """Launch the GUI in the default browser."""
    build_demo().launch(
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=True,
        theme=gr.themes.Soft(),
        css=CSS,
    )


if __name__ == "__main__":
    main()
