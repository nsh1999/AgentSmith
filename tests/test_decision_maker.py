"""Tests for the decision engine's aggregation, streaming, and rendering."""

import threading
from types import SimpleNamespace

import pytest
from conftest import choice_answer, noul_answer, score_answer, system_one_response
from typesafe_sdk import Choice, Noul, Score, TypeSafeError

import decision_maker as dm

DEMO_QUESTIONS = {
    "team": Choice(
        instructions="Which team should handle this ticket?",
        criteria={"billing": "Payments and refunds", "technical": "Bugs"},
    ),
    "refund": Noul(instructions="Does the customer explicitly ask for a refund?"),
    "urgency": Score(instructions="How urgent is this ticket?", criteria=["Routine", "Soon"]),
}


def demo_answers(choice="billing", noul=0.9, score=1.0):
    """Build one model's answer set for the demo questions."""
    return {
        "team": choice_answer(choice),
        "refund": noul_answer(noul),
        "urgency": score_answer(score),
    }


# --- majority_2_3 -----------------------------------------------------------


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([], None),
        (["billing"], "billing"),
        (["billing", "billing"], "billing"),
        (["billing", "technical"], None),
        (["a", "a", "b", "b"], None),  # threshold 2.67, neither reaches it
        (["a", "a", "a", "b"], "a"),  # threshold 2.67, three agree
        (["a", "a", "a", "a", "b"], "a"),  # threshold 3.33, four agree
        (["a", "a", "a", "b", "b"], None),  # five models need four
    ],
)
def test_majority_2_3(values, expected):
    """Majority requires len*2/3 agreement."""
    assert dm.majority_2_3(values) == expected


# --- _init_result_set -------------------------------------------------------


def test_init_result_set_omits_criteria_for_noul():
    """Only questions with criteria keep the key."""
    result_set = dm._init_result_set(["m1"], DEMO_QUESTIONS)

    assert "criteria" not in result_set["refund"]
    assert result_set["team"]["criteria"] == {"billing": "Payments and refunds", "technical": "Bugs"}
    assert result_set["models"] == {"m1": {"runtime": 0.0}}
    assert result_set["team"]["model_answers"] == []


# --- _record_answer ---------------------------------------------------------


def test_record_answer_choice():
    """Choice answers append every field and use a 2/3 majority."""
    result_set = dm._init_result_set(["m1", "m2"], DEMO_QUESTIONS)

    for model in ("m1", "m2"):
        dm._record_answer(result_set, "team", model, choice_answer("billing", 0.9), DEMO_QUESTIONS["team"])

    entry = result_set["team"]
    assert entry["model_answers"][0] == {
        "model": "m1",
        "choice": "billing",
        "confidence": 0.9,
        "probabilities": {"billing": 0.9, "other": 0.1},
    }
    assert entry["majority"] == "billing"
    # Note the double space before the criteria repr: preserved from the original.
    assert entry["explained"] == (
        "Which team should handle this ticket?  "
        "{'billing': 'Payments and refunds', 'technical': 'Bugs'}, billing"
    )


def test_record_answer_score():
    """Score answers average and map onto the criteria legend."""
    result_set = dm._init_result_set(["m1", "m2"], DEMO_QUESTIONS)

    dm._record_answer(result_set, "urgency", "m1", score_answer(0.0), DEMO_QUESTIONS["urgency"])
    dm._record_answer(result_set, "urgency", "m2", score_answer(1.0), DEMO_QUESTIONS["urgency"])

    entry = result_set["urgency"]
    assert entry["score_list"] == [0.0, 1.0]
    assert entry["majority"] == 0.5
    # round(0.5) is 0 (banker's rounding), so the legend label is "Routine".
    assert entry["explained"] == (
        "How urgent is this ticket? ['Routine', 'Soon'], Routine, avg score: 0.5"
    )
    assert entry["model_answers"][1]["legend"] == {0: "Routine", 1: "Soon", 2: "Urgent"}


def test_record_answer_noul():
    """Noul answers average and convert to Yes/No at 0.5."""
    result_set = dm._init_result_set(["m1", "m2"], DEMO_QUESTIONS)

    dm._record_answer(result_set, "refund", "m1", noul_answer(0.2), DEMO_QUESTIONS["refund"])
    dm._record_answer(result_set, "refund", "m2", noul_answer(0.4), DEMO_QUESTIONS["refund"])

    entry = result_set["refund"]
    assert entry["majority"] == pytest.approx(0.30000000000000004)
    assert entry["explained"].endswith("No, model answers: [0.2, 0.4]")
    assert entry["model_answers"] == [{"model": "m1", "noul": 0.2}, {"model": "m2", "noul": 0.4}]


def test_record_answer_unknown_type_changes_nothing():
    """An unexpected answer type is ignored rather than crashing."""
    result_set = dm._init_result_set([], DEMO_QUESTIONS)

    dm._record_answer(result_set, "team", "m1", SimpleNamespace(type="future"), DEMO_QUESTIONS["team"])

    assert result_set["team"]["model_answers"] == []
    assert "majority" not in result_set["team"]


# --- stream_decisions -------------------------------------------------------


def test_stream_event_order(install_fake_client):
    """Each model gets a start and done event, then the run is done."""
    install_fake_client(
        {
            "m1": system_one_response("m1", demo_answers()),
            "m2": system_one_response("m2", demo_answers()),
        }
    )

    events = list(dm.stream_decisions(["m1", "m2"], DEMO_QUESTIONS, "ticket"))

    assert [event["event"] for event in events] == [
        "model_start",
        "model_done",
        "model_start",
        "model_done",
        "done",
    ]
    assert events[0]["model"] == "m1" and events[0]["index"] == 1
    assert events[1]["runtime"] >= 0
    assert events[-1]["total"] == 2


def test_stream_snapshots_are_independent(install_fake_client):
    """Mutating one snapshot must not affect later ones."""
    install_fake_client(
        {
            "m1": system_one_response("m1", demo_answers()),
            "m2": system_one_response("m2", demo_answers("technical")),
        }
    )
    events = list(dm.stream_decisions(["m1", "m2"], DEMO_QUESTIONS, "ticket"))

    first_snapshot = events[1]["result_set"]
    first_snapshot["team"]["model_answers"].clear()

    assert len(events[3]["result_set"]["team"]["model_answers"]) == 2


def test_stream_reuses_one_client(install_fake_client):
    """A single client serves every model in the run."""
    client = install_fake_client(
        {
            "m1": system_one_response("m1", demo_answers()),
            "m2": system_one_response("m2", demo_answers()),
        }
    )

    list(dm.stream_decisions(["m1", "m2"], DEMO_QUESTIONS, "ticket"))

    assert [call["model"] for call in client.calls] == ["m1", "m2"]
    assert client.calls[0]["state"] == {"ticket": "ticket"}
    assert client.closed is True


def test_stream_raises_by_default(install_fake_client):
    """Without tolerate_errors the SDK error propagates."""
    install_fake_client(error=TypeSafeError("boom"))

    with pytest.raises(TypeSafeError, match="boom"):
        list(dm.stream_decisions(["m1"], DEMO_QUESTIONS, "ticket"))


def test_stream_tolerates_errors(install_fake_client):
    """With tolerate_errors the failing model is reported and the run continues."""
    install_fake_client(error=TypeSafeError("boom"))

    events = list(dm.stream_decisions(["m1"], DEMO_QUESTIONS, "ticket", tolerate_errors=True))

    assert [event["event"] for event in events] == ["model_start", "model_error", "done"]
    assert "TypeSafeError: boom" in events[1]["error"]
    assert events[1]["result_set"]["team"]["model_answers"] == []


def test_stream_stop_event_skips_remaining(install_fake_client):
    """A set stop event ends the run before any model is called."""
    install_fake_client({})
    stop_event = threading.Event()
    stop_event.set()

    events = list(dm.stream_decisions(["m1", "m2"], DEMO_QUESTIONS, "ticket", stop_event=stop_event))

    assert [event["event"] for event in events] == ["stopped"]
    assert events[0]["remaining"] == ["m1", "m2"]


def test_decision_maker_matches_final_snapshot(install_fake_client):
    """The wrapper returns exactly the final streamed result set."""
    install_fake_client(
        {
            "m1": system_one_response("m1", demo_answers()),
            "m2": system_one_response("m2", demo_answers("technical")),
        }
    )

    result_set = dm.decision_maker(["m1", "m2"], DEMO_QUESTIONS, "ticket")
    final_snapshot = list(dm.stream_decisions(["m1", "m2"], DEMO_QUESTIONS, "ticket"))[-1]
    # Runtimes differ between runs; compare everything else.
    result_set["models"] = final_snapshot["result_set"]["models"] = {}

    assert result_set == final_snapshot["result_set"]
    assert len(result_set["team"]["model_answers"]) == 2
    assert result_set["team"]["majority"] is None  # one billing, one technical


# --- rendering --------------------------------------------------------------


def test_html_renders_question_with_no_answers():
    """Regression: an unanswered question renders empty cells, not IndexError."""
    result_set = dm._init_result_set(["m1", "m2"], DEMO_QUESTIONS)

    page = dm.resutlt_set_to_html(result_set)

    # Two models are selected but none has answered yet.
    assert "<tr><td>Choice</td></tr>" in page
    assert "Which team should handle this ticket?" in page


def test_html_escapes_when_requested():
    """escape=True neutralises markup supplied through instructions."""
    result_set = dm._init_result_set([], {"q": Noul(instructions="<script>x</script>")})

    escaped = dm.resutlt_set_to_html(result_set, escape=True)
    raw = dm.resutlt_set_to_html(result_set)

    assert "<script>" not in escaped
    assert "&lt;script&gt;" in escaped
    assert "<script>" in raw


def test_result_set_to_html_alias():
    """The correctly spelled alias points at the original function."""
    assert dm.result_set_to_html is dm.resutlt_set_to_html
