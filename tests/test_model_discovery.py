"""Tests for the model-discovery cascade, with every network seam faked."""

from types import SimpleNamespace

import pytest
from typesafe_sdk import TypeSafeError

import decision_gui as gui


def _mock_client(metadata):
    """Build a fake TypeSafeClient exposing the models accessor."""

    class FakeModels:
        def list(self):
            return SimpleNamespace(models=tuple(metadata))

    class FakeClient:
        def __init__(self, **kwargs):
            self.models = FakeModels()

        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            return False

    return FakeClient


# --- _parse_model_names -----------------------------------------------------


def test_parse_model_names_typesafe_shape():
    """TypeSafe responses use models[].name."""
    payload = {"models": [{"name": "a", "description": "", "release_date": ""}, {"name": "b"}]}
    assert gui._parse_model_names(payload) == ["a", "b"]


def test_parse_model_names_openai_shape():
    """OpenAI-compatible responses use data[].id."""
    assert gui._parse_model_names({"data": [{"id": "a"}, {"id": "b"}]}) == ["a", "b"]


def test_parse_model_names_ollama_shape():
    """Ollama tags responses also use models[].name."""
    assert gui._parse_model_names({"models": [{"name": "a"}, {"name": "b"}]}) == ["a", "b"]


def test_parse_model_names_deduplicates():
    """Repeated names collapse while preserving order."""
    assert gui._parse_model_names({"data": [{"id": "a"}, {"id": "a"}, {"id": "b"}]}) == ["a", "b"]


@pytest.mark.parametrize("payload", [None, [], "text", {}, {"data": "nope"}, {"models": [1, 2]}])
def test_parse_model_names_garbage_returns_empty(payload):
    """Unknown shapes yield an empty list rather than raising."""
    assert gui._parse_model_names(payload) == []


# --- discover_models cascade ------------------------------------------------


def test_discover_models_uses_sdk(monkeypatch):
    """The SDK tier wins when it returns metadata."""
    metadata = [SimpleNamespace(name="sdk-model")]
    monkeypatch.setattr(gui, "TypeSafeClient", _mock_client(metadata))

    names, source = gui.discover_models("http://localhost:11434", "key")

    assert names == ["sdk-model"]
    assert source == "TypeSafe `/v1/models`"


def test_discover_models_falls_back_to_openai(monkeypatch):
    """When the SDK cannot decode the response, /v1/models answers."""

    def failing_client(**kwargs):
        raise TypeSafeError("invalid shape")

    monkeypatch.setattr(gui, "TypeSafeClient", failing_client)
    monkeypatch.setattr(gui, "_fetch_json", lambda url, key=None, timeout=5.0: {"data": [{"id": "x"}]})

    names, source = gui.discover_models("http://localhost:11434", "key")

    assert names == ["x"]
    assert "OpenAI-compatible" in source


def test_discover_models_falls_back_to_ollama_tags(monkeypatch):
    """When /v1/models fails, the native tags endpoint answers."""
    monkeypatch.setattr(gui, "TypeSafeClient", _mock_client([]))

    def fake_fetch(url, key=None, timeout=5.0):
        if url.endswith("/v1/models"):
            raise OSError("connection refused")
        return {"models": [{"name": "ollama-model"}]}

    monkeypatch.setattr(gui, "_fetch_json", fake_fetch)

    names, source = gui.discover_models("http://localhost:11434", "key")

    assert names == ["ollama-model"]
    assert "Ollama" in source


def test_discover_models_falls_back_to_hardcoded(monkeypatch):
    """Every tier failing yields the demo list and a reason."""

    def failing_client(**kwargs):
        raise TypeSafeError("down")

    def failing_fetch(url, key=None, timeout=5.0):
        raise OSError("down")

    monkeypatch.setattr(gui, "TypeSafeClient", failing_client)
    monkeypatch.setattr(gui, "_fetch_json", failing_fetch)

    names, source = gui.discover_models("http://localhost:11434", "key")

    assert names == list(gui.FALLBACK_MODELS)
    assert "hardcoded demo list" in source
    assert "down" in source


def test_discover_models_strips_trailing_slash(monkeypatch):
    """A trailing slash on the base URL does not double up path separators."""
    seen = []

    def failing_client(**kwargs):
        raise TypeSafeError("skip")

    def recording_fetch(url, key=None, timeout=5.0):
        seen.append(url)
        raise OSError("skip")

    monkeypatch.setattr(gui, "TypeSafeClient", failing_client)
    monkeypatch.setattr(gui, "_fetch_json", recording_fetch)

    gui.discover_models("http://localhost:11434/", "key")

    assert seen == ["http://localhost:11434/v1/models", "http://localhost:11434/api/tags"]
