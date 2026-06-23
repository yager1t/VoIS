"""Tests for the VocabularyManager."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config import Settings
from src.dictionary import VocabularyManager
from src.dictionary.base import ContextMode, DictionaryEntry, VocabularySource
from src.dictionary.context_modes import get_context_vocabulary


@pytest.fixture
def manager(tmp_path: Path) -> VocabularyManager:
    """Return a manager using an isolated vocabulary directory."""
    settings = Settings(
        vocab_dir=tmp_path / "vocab",
        context_mode="code",
    )
    return VocabularyManager(settings)


def test_load_all_loads_context_terms(manager: VocabularyManager) -> None:
    """load_all should populate context-specific starter terms."""
    manager.load_all()
    replacements = manager.get_replacements()
    for term in get_context_vocabulary(ContextMode.code):
        assert term in replacements


def test_get_replacements_returns_all_sources(manager: VocabularyManager) -> None:
    """Terms from static, context, and user sources should be merged."""
    storage = manager._storage
    storage.save(
        [DictionaryEntry(term="static_term", replacement="s", source=VocabularySource.static)],
        VocabularySource.static,
    )
    manager.load_all()
    manager.add_user_term("user_term", "u")

    replacements = manager.get_replacements()
    assert replacements["static_term"] == "s"
    assert replacements["user_term"] == "u"
    assert "function" in replacements


def test_add_and_remove_user_term(manager: VocabularyManager) -> None:
    """User terms should be persisted and removable."""
    manager.load_all()
    manager.add_user_term("acme", "ACME Corporation")
    assert manager.get_replacements()["acme"] == "ACME Corporation"

    manager.remove_user_term("acme")
    assert "acme" not in manager.get_replacements()

    # Removing a missing term should not raise.
    manager.remove_user_term("missing")


def test_context_mode_switch_reloads_vocabulary(manager: VocabularyManager) -> None:
    """Changing context mode should load the new context's starters."""
    manager.load_all()
    assert manager.get_context_mode() == ContextMode.code
    assert "function" in manager.get_replacements()

    manager.set_context_mode("email")
    assert manager.get_context_mode() == ContextMode.email
    replacements = manager.get_replacements()
    assert "Sincerely" in replacements
    assert "function" not in replacements


def test_override_priority_user_context_static(manager: VocabularyManager) -> None:
    """User entries should override context, which overrides static."""
    storage = manager._storage
    storage.save(
        [DictionaryEntry(term="foo", replacement="static", source=VocabularySource.static)],
        VocabularySource.static,
    )
    storage.save(
        [DictionaryEntry(term="foo", replacement="context", source=VocabularySource.context)],
        VocabularySource.context,
        ContextMode.code,
    )
    manager.load_all()

    # Context overrides static.
    assert manager.get_replacements()["foo"] == "context"

    # User overrides context.
    manager.add_user_term("foo", "user")
    assert manager.get_replacements()["foo"] == "user"

    # Removing user falls back to context.
    manager.remove_user_term("foo")
    assert manager.get_replacements()["foo"] == "context"


def test_dictionary_disabled_returns_empty(manager: VocabularyManager) -> None:
    """When disabled, get_replacements should be empty."""
    manager._dictionary_enabled = False
    manager.load_all()
    assert manager.get_replacements() == {}


def test_get_terms_filters_by_context(manager: VocabularyManager) -> None:
    """get_terms should support filtering by context value."""
    manager.load_all()
    code_terms = manager.get_terms(context=ContextMode.code)
    assert "function" in code_terms
    assert "Sincerely" not in code_terms
