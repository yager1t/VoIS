"""Unit tests for ASR biasing utilities."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.config import Settings
from src.dictionary import ASRBias
from src.dictionary.base import ContextMode, DictionaryEntry, VocabularySource


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Return settings with a temporary vocab directory."""
    return Settings(
        vocab_dir=tmp_path / "vocab",
        dictionary_enabled=True,
    )


@pytest.fixture
def settings_disabled(tmp_path: Path) -> Settings:
    """Return settings with dictionary disabled."""
    return Settings(
        vocab_dir=tmp_path / "vocab",
        dictionary_enabled=False,
    )


def _make_manager(settings: Settings) -> MagicMock:
    """Create a mocked VocabularyManager with a few terms."""
    manager = MagicMock()
    manager.get_context_mode.return_value = ContextMode.email
    manager.get_terms.return_value = ["OAuth", "gRPC", "Kubernetes"]
    manager._static = {
        "OAuth": DictionaryEntry(
            term="OAuth",
            replacement="OAuth",
            source=VocabularySource.static,
            count=5,
        ),
    }
    manager._context = {
        "gRPC": DictionaryEntry(
            term="gRPC",
            replacement="gRPC",
            source=VocabularySource.context,
            count=10,
        ),
    }
    manager._user = {
        "Kubernetes": DictionaryEntry(
            term="Kubernetes",
            replacement="Kubernetes",
            source=VocabularySource.user,
            count=3,
        ),
    }
    return manager


def test_initial_prompt_includes_context_and_terms(settings: Settings) -> None:
    """The prompt should combine context description and vocabulary terms."""
    manager = _make_manager(settings)
    bias = ASRBias(manager, settings)

    prompt = bias.initial_prompt()

    assert "formal email" in prompt
    assert "OAuth" in prompt
    assert "gRPC" in prompt
    assert "Kubernetes" in prompt
    assert len(prompt) <= 250


def test_initial_prompt_general_context(settings: Settings) -> None:
    """General context has no description, so only terms are mentioned."""
    manager = _make_manager(settings)
    manager.get_context_mode.return_value = ContextMode.general
    bias = ASRBias(manager, settings)

    prompt = bias.initial_prompt()

    assert "Transcribe" not in prompt
    assert "Important terms:" in prompt


def test_hotwords_respects_limit_and_sorting(settings: Settings) -> None:
    """Hotwords should be limited and sorted by count then alphabetically."""
    manager = _make_manager(settings)
    manager.get_context_mode.return_value = ContextMode.general
    manager.get_terms.return_value = ["OAuth", "gRPC", "Kubernetes", "alpha"]
    manager._user["alpha"] = DictionaryEntry(
        term="alpha",
        replacement="alpha",
        source=VocabularySource.user,
        count=1,
    )
    bias = ASRBias(manager, settings)

    hotwords = bias.hotwords()

    assert hotwords[0] == "gRPC"
    assert hotwords[1] == "OAuth"
    assert hotwords[2] == "Kubernetes"
    assert hotwords[-1] == "alpha"


def test_disabled_dictionary_returns_empty_prompt(settings_disabled: Settings) -> None:
    """When disabled, initial_prompt should only return the context prompt."""
    manager = _make_manager(settings_disabled)
    bias = ASRBias(manager, settings_disabled)

    assert bias.initial_prompt() == "formal email"
    assert bias.hotwords() == []


def test_disabled_general_context_returns_empty(settings_disabled: Settings) -> None:
    """When disabled and context is general, both hints should be empty."""
    manager = _make_manager(settings_disabled)
    manager.get_context_mode.return_value = ContextMode.general
    bias = ASRBias(manager, settings_disabled)

    assert bias.initial_prompt() == ""
    assert bias.hotwords() == []


def test_truncate_does_not_break_words(settings: Settings) -> None:
    """Truncation should avoid cutting a word in half."""
    manager = _make_manager(settings)
    bias = ASRBias(manager, settings)

    long_text = "a " + "word " * 200
    truncated = bias._truncate(long_text, 20)

    assert len(truncated) <= 20
    assert truncated.endswith("…")
    assert "word" in truncated or truncated == "a …"
