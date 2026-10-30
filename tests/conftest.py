"""Shared fixtures: a fake TypeSafe client so tests never touch the network."""

import pytest
from typesafe_sdk import (
    ChoiceAnswer,
    NoulAnswer,
    ScoreAnswer,
    SystemOneResponse,
    Usage,
)


def choice_answer(choice: str = "billing", confidence: float = 0.9) -> ChoiceAnswer:
    """Build a ChoiceAnswer with plausible probabilities."""
    return ChoiceAnswer(
        choice=choice,
        confidence=confidence,
        probabilities={choice: confidence, "other": round(1 - confidence, 4)},
    )


def score_answer(score: float = 1.0, confidence: float = 0.8) -> ScoreAnswer:
    """Build a ScoreAnswer over the demo's three-level legend."""
    return ScoreAnswer(
        score=score,
        confidence=confidence,
        legend={0: "Routine", 1: "Soon", 2: "Urgent"},
        probabilities={0: 0.1, 1: 0.1, 2: 0.8},
    )


def noul_answer(noul: float = 0.9) -> NoulAnswer:
    """Build a NoulAnswer with the given probability."""
    return NoulAnswer(noul=noul)


def system_one_response(model: str, answers: dict) -> SystemOneResponse:
    """Wrap answers in a SystemOneResponse for a given model."""
    return SystemOneResponse(
        model=model,
        usage=Usage(input_tokens=0, output_tokens=0),
        answers=answers,
    )


class FakeClient:
    """Stand-in for TypeSafeClient returning canned answers or raising."""

    def __init__(self, responses=None, error=None):
        """Store per-model responses (or an error to raise)."""
        self.responses = responses or {}
        self.error = error
        self.calls = []
        self.closed = False

    def __enter__(self):
        """Enter the context manager."""
        return self

    def __exit__(self, *exc_info):
        """Mark the client closed."""
        self.closed = True

    def system_one(self, *, model, state, questions):
        """Return the canned response for ``model`` or raise ``self.error``."""
        self.calls.append({"model": model, "state": state, "questions": questions})
        if self.error is not None:
            raise self.error
        return self.responses[model]


@pytest.fixture
def install_fake_client(monkeypatch):
    """Return a factory that installs a FakeClient into decision_maker."""

    def install(responses=None, error=None) -> FakeClient:
        client = FakeClient(responses=responses, error=error)
        monkeypatch.setattr("decision_maker.TypeSafeClient", lambda **kwargs: client)
        return client

    return install
