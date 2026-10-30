"""Tests for the Gradio run generator and stop handler."""

import decision_gui as gui

DEMO_ROWS = [list(row) for row in gui.DEMO_QUESTION_ROWS]


def _fake_stream(events, captured):
    """Return a stream_decisions replacement yielding ``events``."""

    def stream(models, questions, ticket, **kwargs):
        captured["models"] = models
        captured["questions"] = questions
        captured["ticket"] = ticket
        captured["kwargs"] = kwargs
        yield from events

    return stream


def _result_set(models=("m1",)):
    """Build a minimal result set shaped like the engine's output."""
    result_set = {"models": {model: {"runtime": 0.1} for model in models}}
    result_set["team"] = {
        "name": "team",
        "type": "choice",
        "instructions": "Which team?",
        "criteria": {"billing": "Payments"},
        "model_answers": [],
        "majority": None,
        "explained": "Which team?  {'billing': 'Payments'}, None",
    }
    return result_set


def test_run_decision_streams_updates(monkeypatch):
    """Each engine event becomes one UI update and runtimes reach the status table."""
    captured = {}
    events = [
        {"event": "model_start", "total": 1, "model": "m1", "result_set": _result_set()},
        {
            "event": "model_done",
            "total": 1,
            "model": "m1",
            "runtime": 0.5,
            "result_set": _result_set(),
        },
        {"event": "done", "total": 1, "result_set": _result_set()},
    ]
    monkeypatch.setattr(gui, "stream_decisions", _fake_stream(events, captured))

    updates = list(gui.run_decision("ticket", DEMO_ROWS, ["m1"], "http://x", "key", 30, ""))

    assert len(updates) == 1 + len(events)
    assert captured["models"] == ["m1"]
    assert captured["ticket"] == "ticket"
    assert captured["kwargs"]["tolerate_errors"] is True
    assert captured["kwargs"]["timeout"] == 30.0
    assert captured["kwargs"]["stop_event"] is not None

    results, status, summary, run_id = updates[-1]
    assert "<table" in results
    assert status == [["m1", "done", "0.50"]]
    assert "no majority" in summary
    assert run_id


def test_run_decision_defaults_timeout(monkeypatch):
    """A missing timeout falls back to 120 seconds."""
    captured = {}
    monkeypatch.setattr(gui, "stream_decisions", _fake_stream([], captured))

    list(gui.run_decision("ticket", DEMO_ROWS, ["m1"], "", "", None, ""))

    assert captured["kwargs"]["timeout"] == 120.0


def test_run_decision_reports_validation_errors(monkeypatch):
    """Invalid inputs produce one error frame and never call the engine."""
    captured = {}
    monkeypatch.setattr(gui, "stream_decisions", _fake_stream([], captured))

    updates = list(gui.run_decision("", DEMO_ROWS, ["m1"], "", "", 120, ""))

    assert len(updates) == 1
    results, status, summary, run_id = updates[0]
    assert "Enter a ticket" in results
    assert status == []
    assert summary == ""
    assert run_id == ""
    assert captured == {}


def test_run_decision_reports_question_errors(monkeypatch):
    """A broken question table is reported without starting a run."""
    monkeypatch.setattr(gui, "stream_decisions", _fake_stream([], {}))

    updates = list(gui.run_decision("ticket", [["q", "score", "a?", "Only"]], ["m1"], "", "", 120, ""))

    assert "at least 2 levels" in updates[0][0]


def test_run_decision_reports_model_errors(monkeypatch):
    """A failing model is shown in the status table and the run continues."""
    events = [
        {
            "event": "model_error",
            "total": 1,
            "model": "m1",
            "runtime": 0.2,
            "error": "TypeSafeError: boom",
            "result_set": _result_set(),
        },
        {"event": "done", "total": 1, "result_set": _result_set()},
    ]
    monkeypatch.setattr(gui, "stream_decisions", _fake_stream(events, {}))

    updates = list(gui.run_decision("ticket", DEMO_ROWS, ["m1"], "", "", 120, ""))

    assert updates[-1][1] == [["m1", "error: TypeSafeError: boom", "0.20"]]


def test_run_decision_marks_remaining_models_stopped(monkeypatch):
    """A stopped run marks models it never reached."""
    events = [
        {"event": "stopped", "total": 2, "remaining": ["m2"], "result_set": _result_set()},
    ]
    monkeypatch.setattr(gui, "stream_decisions", _fake_stream(events, {}))

    updates = list(gui.run_decision("ticket", DEMO_ROWS, ["m1", "m2"], "", "", 120, ""))

    assert updates[-1][1] == [
        ["m1", "queued", ""],
        ["m2", "skipped (stopped)", ""],
    ]


def test_run_decision_cleans_up_run_registry(monkeypatch):
    """No stop flag survives a completed run."""
    monkeypatch.setattr(gui, "stream_decisions", _fake_stream([], {}))

    list(gui.run_decision("ticket", DEMO_ROWS, ["m1"], "", "", 120, ""))

    assert gui._RUNS == {}


def test_stop_run_unknown_id():
    """Stopping an unknown run reports that nothing is in progress."""
    assert gui.stop_run("missing") == "No run in progress."
    assert gui.stop_run("") == "No run in progress."
