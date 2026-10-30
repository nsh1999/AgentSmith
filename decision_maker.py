import copy
import html as html_lib
import os
import threading
import time
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from pprint import pprint
from statistics import mean

from typesafe_sdk import (
    Answer,
    Choice,
    Noul,
    Question,
    RetryPolicy,
    Score,
    TypeSafeClient,
    TypeSafeError,
)

os.environ.setdefault("TYPESAFE_API_KEY", "ollamna")
os.environ.setdefault("TYPESAFE_BASE_URL", "http://localhost:11434")
os.environ.setdefault("TYPESAFE_DEFAULT_MODEL", "nimble:latest")


def _init_result_set(models: Sequence[str], questions: Mapping[str, Question]) -> dict:
    """Build the empty result-set skeleton for a run.

    Holds one ``{"runtime": 0.0}`` entry per model plus, per question, the metadata
    needed for rendering. ``criteria`` is omitted for question types that have none.
    """
    result_set: dict = {"models": {}}
    for model in models:
        result_set["models"][model] = {"runtime": 0.0}
    for key, question in questions.items():
        criteria = getattr(question, "criteria", None)
        result_set[key] = {
            "name": key,
            "type": question.type,
            "instructions": question.instructions,
            "criteria": criteria,
            "model_answers": [],
        }
        if criteria is None:
            del result_set[key]["criteria"]
    return result_set


def _record_answer(
    result_set: dict,
    key: str,
    model: str,
    answer: Answer,
    question: Question,
) -> None:
    """Append one model's answer and recompute that question's aggregate.

    The aggregate is recomputed after every answer so a partial run is always
    displayable: choices use :func:`majority_2_3`, scores use the mean mapped onto
    the criteria legend, and nouls use the mean probability with a 0.5 threshold.
    """
    if answer.type == "choice":
        result_set[key]["model_answers"].append(
            {
                "model": model,
                "choice": answer.choice,
                "confidence": answer.confidence,
                "probabilities": answer.probabilities,
            }
        )
        final_choices = [entry["choice"] for entry in result_set[key]["model_answers"]]
        result_set[key]["majority"] = majority_2_3(final_choices)
        result_set[key]["explained"] = (
            f"{result_set[key]['instructions']}  "
            f"{question.criteria}, {result_set[key]['majority']}"
        )
        return

    if answer.type == "score":
        result_set[key]["model_answers"].append(
            {
                "model": model,
                "score": answer.score,
                "confidence": answer.confidence,
                "probabilities": answer.probabilities,
                "legend": answer.legend,
            }
        )
        score_list = [entry["score"] for entry in result_set[key]["model_answers"]]
        average_score = mean(score_list)
        result_set[key]["majority"] = average_score
        result_set[key]["explained"] = (
            f"{result_set[key]['instructions']} {question.criteria}, "
            f"{question.criteria[round(average_score)]}, avg score: {average_score}"
        )
        result_set[key]["score_list"] = score_list
        return

    if answer.type == "noul":
        result_set[key]["model_answers"].append({"model": model, "noul": answer.noul})
        noul_values = [entry["noul"] for entry in result_set[key]["model_answers"]]
        result_set[key]["majority"] = mean(noul_values)
        result_set[key]["explained"] = (
            f"{result_set[key]['instructions']}, "
            f"{'Yes' if result_set[key]['majority'] >= 0.5 else 'No'}, "
            f"model answers: {noul_values}"
        )


def stream_decisions(
    models: Sequence[str],
    questions: Mapping[str, Question],
    ticket: str,
    *,
    timeout: float = 120.0,
    retry: RetryPolicy | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    tolerate_errors: bool = False,
    stop_event: threading.Event | None = None,
) -> Iterator[dict]:
    """Run every model in turn, yielding a snapshot event after each step.

    Events carry ``event`` (one of ``model_start``, ``model_done``, ``model_error``,
    ``stopped``, ``done``), ``total`` model count, ``result_set`` (a deep copy, safe
    for a consumer to keep), and event-specific fields.

    Cancellation is per model: ``stop_event`` is checked between calls, so an
    in-flight request still runs to completion. With ``tolerate_errors`` a failing
    model is reported as a ``model_error`` event and the run continues; otherwise
    the :class:`TypeSafeError` propagates.
    """
    result_set = _init_result_set(models, questions)
    total = len(models)

    def snapshot(event: str, **fields) -> dict:
        return {
            "event": event,
            "total": total,
            **fields,
            "result_set": copy.deepcopy(result_set),
        }

    with TypeSafeClient(
        timeout=timeout,
        retry=retry,
        base_url=base_url,
        api_key=api_key,
    ) as client:
        for index, model in enumerate(models, start=1):
            if stop_event is not None and stop_event.is_set():
                yield snapshot("stopped", remaining=list(models[index - 1 :]))
                return

            yield snapshot("model_start", model=model, index=index)
            started_at = time.time()
            try:
                result = client.system_one(
                    model=model,
                    state={"ticket": ticket},
                    questions=questions,
                )
            except TypeSafeError as error:
                if not tolerate_errors:
                    raise
                runtime = time.time() - started_at
                result_set["models"][model]["runtime"] = runtime
                yield snapshot(
                    "model_error",
                    model=model,
                    runtime=runtime,
                    error=f"{type(error).__name__}: {error}",
                )
                continue

            runtime = time.time() - started_at
            result_set["models"][model]["runtime"] = runtime
            for key, answer in result.answers.items():
                _record_answer(result_set, key, model, answer, questions[key])
            yield snapshot("model_done", model=model, runtime=runtime)

    yield snapshot("done")


def decision_maker(models, questions, ticket):
    """Run every model and return the final aggregated result set.

    Thin synchronous wrapper over :func:`stream_decisions`; keeps the original
    script/notebook API, including raising on a model failure.
    """
    result_set = _init_result_set(models, questions)
    for event in stream_decisions(models, questions, ticket):
        result_set = event["result_set"]
    return result_set


# Maximum number of decimals shown for a floating-point answer value.
DECIMAL_PLACES = 4

# Question type -> ordered (row label, answer key) pairs. Each pair becomes
# one table row; every model gets its own column.
TABLE_ROWS = {
    "noul": [
        ("Noul", "noul"),
    ],
    "choice": [
        ("Choice", "choice"),
        ("Confidence", "confidence"),
        ("Probabilities", "probabilities"),
    ],
    "score": [
        ("Score", "score"),
        ("Confidence", "confidence"),
        ("Probabilities", "probabilities", "legend"),
    ],
}


def majority_2_3(lst):
    """Return the value that appears in at least 2/3 of the list, or None."""
    if not lst:
        return None

    threshold = len(lst) * 2 / 3
    counts = Counter(lst)

    for value, count in counts.items():
        if count >= threshold:
            return value
    return None


def _format_cell(value):
    """Format one answer value for display in a table cell.

    Floating-point values are rounded to at most ``DECIMAL_PLACES`` decimals;
    every other type is returned unchanged.
    """
    if not isinstance(value, float):
        return value
    return round(value, DECIMAL_PLACES)


def _escape(value, escape: bool):
    """HTML-escape ``value`` when ``escape`` is set, otherwise return it as is."""
    if not escape:
        return value
    return html_lib.escape(str(value))


def _model_answers_to_table(model_answers, rows, *, escape: bool = False):
    """Render model answers as an HTML table with one column per model.

    ``rows`` is an ordered sequence of ``(label, answer_key)`` pairs, each
    optionally extended to ``(label, answer_key, label_key)``. The model names
    form the header row, and each pair becomes one body row holding that key's
    value from every model's answer. When the value is a mapping, the pair
    expands into a heading row named after the label, followed by one row per
    mapping key. With ``label_key`` set, those rows are labelled by the first
    model's ``label_key`` mapping rather than by the key itself.

    A question with no answers yet renders as empty cells, which lets a caller
    display a run that is still in progress.
    """
    header_cells = "".join(
        f"<th>{_escape(answer['model'], escape)}</th>" for answer in model_answers
    )
    empty_cells = "".join("<td></td>" for _ in model_answers)
    html = f"<table border='1'><tr><th></th>{header_cells}</tr>"
    for row_spec in rows:
        label, answer_key = row_spec[0], row_spec[1]
        label_key = row_spec[2] if len(row_spec) > 2 else None
        values = [answer[answer_key] for answer in model_answers]
        if not values:
            html += f"<tr><td>{_escape(label, escape)}</td>{empty_cells}</tr>"
            continue
        if not isinstance(values[0], dict):
            value_cells = "".join(
                f"<td>{_escape(_format_cell(value), escape)}</td>" for value in values
            )
            html += f"<tr><td>{_escape(label, escape)}</td>{value_cells}</tr>"
            continue
        html += f"<tr><td>{_escape(label, escape)}</td>{empty_cells}</tr>"
        sub_labels = (model_answers[0].get(label_key) or {}) if label_key else {}
        sub_keys = list(dict.fromkeys(key for value in values for key in value))
        for sub_key in sub_keys:
            sub_cells = "".join(
                f"<td>{_escape(_format_cell(value.get(sub_key, '')), escape)}</td>"
                for value in values
            )
            sub_label = sub_labels.get(sub_key, sub_key)
            html += f"<tr><td>{_escape(sub_label, escape)}</td>{sub_cells}</tr>"
    return f"{html}</table>"


def resutlt_set_to_html(result_set, *, escape: bool = False):
    """Render a decision_maker result set as an HTML page.

    Question types listed in ``TABLE_ROWS`` (noul, choice, score) are rendered
    as a table: a leading label column, one column per model along the header
    row, and one row per value. Any other question type falls back to a
    bulleted list.

    Set ``escape=True`` when the result set contains user-supplied text that is
    rendered in a browser.
    """
    html = "<html><body>"
    for key in result_set:
        if key == "models":
            continue
        html += f"<h2>{_escape(result_set[key]['name'], escape)}</h2>"
        html += f"<p>Type: {_escape(result_set[key]['type'], escape)}</p>"
        html += f"<p>Instructions: {_escape(result_set[key]['instructions'], escape)}</p>"
        if "criteria" in result_set[key]:
            html += f"<p>Criteria: {_escape(result_set[key]['criteria'], escape)}</p>"
        model_answers = result_set[key]["model_answers"]
        rows = TABLE_ROWS.get(result_set[key]["type"])
        if rows is not None:
            html += _model_answers_to_table(model_answers, rows, escape=escape)
            continue
        html += "<ul>"
        for answer in model_answers:
            html += "<li>"
            html += f"Model: {_escape(answer['model'], escape)}<br>"
            if "choice" in answer:
                html += f"Choice: {_escape(answer['choice'], escape)}<br>"
                html += f"Confidence: {_escape(answer['confidence'], escape)}<br>"
                html += f"Probabilities: {_escape(answer['probabilities'], escape)}<br>"
            if "score" in answer:
                html += f"Score: {_escape(answer['score'], escape)}<br>"
                html += f"Confidence: {_escape(answer['confidence'], escape)}<br>"
                html += f"Probabilities: {_escape(answer['probabilities'], escape)}<br>"
                html += f"Legend: {_escape(answer['legend'], escape)}<br>"
            if "noul" in answer:
                html += f"Noul: {_escape(answer['noul'], escape)}<br>"
            html += "</li>"
        html += "</ul>"
    html += "</body></html>"
    return html


# Correctly spelled alias for new code; the misspelled name is kept because
# existing scripts and notebooks import it.
result_set_to_html = resutlt_set_to_html


def explain(result_set, questions):
    """Print the human-readable explanation for every question in the result set."""
    for key in result_set:
        if key == "models":
            continue
        if "explained" in result_set[key]:
            print(result_set[key]["explained"])


if __name__ == "__main__":
    models = [
        "clef-flash:9b-mxfp8",
        "tev1:4b-mxfp8",
        "laya:421m-typed-decisions-mlx-fp16",
    ]

    ticket = "I was charged twice. Please refund the extra payment. this is urgent for me"
    questions = {
        "team": Choice(
            instructions="Which team should handle this ticket?",
            criteria={
                "billing": "Payments and refunds",
                "technical": "Bugs and integrations",
                "other": "None of the above",
            },
        ),
        "refund": Noul(
            instructions="Does the customer explicitly ask for a refund?",
        ),
        "urgency": Score(
            instructions="How urgent is this ticket?",
            criteria=["Routine", "Soon", "Urgent"],
        ),
    }
    result_set = decision_maker(models, questions, ticket)
    pprint(result_set)
