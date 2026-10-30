"""Tests for turning GUI table rows into typed questions."""

import pytest
from typesafe_sdk import Choice, Noul, Score

import decision_gui as gui

DEMO_ROWS = [list(row) for row in gui.DEMO_QUESTION_ROWS]


# --- criteria parsers -------------------------------------------------------


def test_parse_choice_criteria_with_descriptions():
    """Labels and descriptions are split on ';' and the first '='."""
    assert gui.parse_choice_criteria("billing = Payments, refunds; technical; other = None") == {
        "billing": "Payments, refunds",
        "technical": None,
        "other": "None",
    }


def test_parse_choice_criteria_newline_separator():
    """Newlines separate items just like semicolons."""
    assert gui.parse_choice_criteria("a = one\nb = two") == {"a": "one", "b": "two"}


def test_parse_choice_criteria_duplicate_label_raises():
    """Repeated labels are rejected."""
    with pytest.raises(gui.QuestionValidationError, match="duplicate choice label"):
        gui.parse_choice_criteria("billing; billing")


def test_parse_score_criteria_splits_commas():
    """Score levels are comma separated."""
    assert gui.parse_score_criteria("Routine, Soon, Urgent") == ["Routine", "Soon", "Urgent"]


def test_parse_noul_criteria_blank_returns_none():
    """A blank cell means no explicit wording."""
    assert gui.parse_noul_criteria("") is None


def test_parse_noul_criteria_true_false():
    """Only true/false keys are accepted."""
    assert gui.parse_noul_criteria("true = Yes it does; false = No") == {
        "true": "Yes it does",
        "false": "No",
    }


def test_parse_noul_criteria_rejects_other_keys():
    """Other keys are a validation error."""
    with pytest.raises(gui.QuestionValidationError, match="true' or 'false'"):
        gui.parse_noul_criteria("maybe = unsure")


# --- build_questions --------------------------------------------------------


def test_build_questions_demo_rows():
    """The seeded rows produce the three demo question types."""
    questions = gui.build_questions(DEMO_ROWS)

    assert list(questions) == ["team", "refund", "urgency"]
    assert isinstance(questions["team"], Choice)
    assert isinstance(questions["refund"], Noul)
    assert isinstance(questions["urgency"], Score)
    assert questions["team"].criteria["billing"] == "Payments and refunds"
    assert questions["urgency"].criteria == ["Routine", "Soon", "Urgent"]
    assert questions["refund"].criteria is None


def test_build_questions_ignores_blank_rows():
    """Fully blank rows are not questions."""
    questions = gui.build_questions(DEMO_ROWS + [["", "", "", ""], [None, None, None, None]])

    assert len(questions) == 3


def test_build_questions_duplicate_names_raises():
    """Names must be unique."""
    rows = [["q", "noul", "a?", ""], ["q", "noul", "b?", ""]]
    with pytest.raises(gui.QuestionValidationError, match="duplicate name"):
        gui.build_questions(rows)


def test_build_questions_reserved_name_raises():
    """'models' is reserved inside the result set."""
    with pytest.raises(gui.QuestionValidationError, match="reserved"):
        gui.build_questions([["models", "noul", "a?", ""]])


@pytest.mark.parametrize("name", ["", "1abc", "has space", "with-dash"])
def test_build_questions_invalid_name_raises(name):
    """Names must be identifiers."""
    with pytest.raises(gui.QuestionValidationError, match="name"):
        gui.build_questions([[name, "noul", "a?", ""]])


def test_build_questions_unknown_type_raises():
    """Only choice, noul and score are supported."""
    with pytest.raises(gui.QuestionValidationError, match="unknown type"):
        gui.build_questions([["q", "ranking", "a?", ""]])


def test_build_questions_requires_instructions():
    """Instructions cannot be blank."""
    with pytest.raises(gui.QuestionValidationError, match="instructions are required"):
        gui.build_questions([["q", "noul", "  ", ""]])


def test_build_questions_choice_needs_two_labels():
    """A single-label choice cannot be answered meaningfully."""
    with pytest.raises(gui.QuestionValidationError, match="at least 2 criteria"):
        gui.build_questions([["q", "choice", "a?", "only"]])


def test_build_questions_score_needs_two_levels():
    """A single-level score is rejected."""
    with pytest.raises(gui.QuestionValidationError, match="at least 2 levels"):
        gui.build_questions([["q", "score", "a?", "Only"]])


def test_build_questions_reports_every_problem():
    """All problems are collected into one error."""
    rows = [
        ["", "noul", "a?", ""],
        ["bad name", "noul", "b?", ""],
        ["q", "unknown", "c?", ""],
    ]
    with pytest.raises(gui.QuestionValidationError) as error_info:
        gui.build_questions(rows)

    message = str(error_info.value)
    assert message.count("\n") == 2


def test_build_questions_requires_at_least_one_question():
    """An all-blank table is an error."""
    with pytest.raises(gui.QuestionValidationError, match="at least one question"):
        gui.build_questions([["", "", "", ""]])


# --- run input validation ---------------------------------------------------


@pytest.mark.parametrize(
    ("ticket", "models", "message"),
    [
        ("", ["m1"], "Enter a ticket"),
        ("   ", ["m1"], "Enter a ticket"),
        ("ticket", [], "Select at least one model"),
        ("ticket", None, "Select at least one model"),
    ],
)
def test_validate_run_inputs_rejects(ticket, models, message):
    """Missing ticket or models is rejected."""
    with pytest.raises(gui.QuestionValidationError, match=message):
        gui.validate_run_inputs(ticket, models)


def test_validate_run_inputs_accepts():
    """A ticket plus one model is valid."""
    gui.validate_run_inputs("I was charged twice.", ["m1"])
