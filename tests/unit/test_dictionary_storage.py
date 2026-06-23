"""Tests for dictionary JSON storage."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.dictionary.base import ContextMode, DictionaryEntry, VocabularySource
from src.dictionary.context_modes import get_context_vocabulary
from src.dictionary.storage import VocabularyStorage


@pytest.fixture
def storage(tmp_path: Path) -> VocabularyStorage:
    """Return an isolated vocabulary storage instance."""
    return VocabularyStorage(tmp_path / "vocab")


def test_load_creates_static_default(storage: VocabularyStorage) -> None:
    """Loading static vocabulary creates an empty default file."""
    entries = storage.load(VocabularySource.static)
    assert entries == []
    assert (storage.vocab_dir / "static.json").exists()


def test_load_creates_context_default_with_starters(storage: VocabularyStorage) -> None:
    """Loading context vocabulary creates starter terms."""
    entries = storage.load(VocabularySource.context, ContextMode.code)
    terms = {entry.term for entry in entries}
    assert terms == set(get_context_vocabulary(ContextMode.code))
    assert (storage.vocab_dir / "context_code.json").exists()


def test_save_and_load_roundtrip(storage: VocabularyStorage) -> None:
    """Saved entries should be returned unchanged on load."""
    entries = [
        DictionaryEntry(
            term="foo",
            replacement="bar",
            source=VocabularySource.user,
            count=3,
            context="chat",
        ),
        DictionaryEntry(
            term="hello",
            replacement="world",
            source=VocabularySource.user,
        ),
    ]
    storage.save(entries, VocabularySource.user)
    loaded = storage.load(VocabularySource.user)
    assert loaded == entries


def test_json_format_matches_expected_shape(storage: VocabularyStorage) -> None:
    """The JSON file should have the documented structure."""
    storage.save(
        [
            DictionaryEntry(
                term="foo",
                replacement="bar",
                source=VocabularySource.user,
                count=3,
            ),
        ],
        VocabularySource.user,
    )
    text = (storage.vocab_dir / "user.json").read_text(encoding="utf-8")
    assert '"term": "foo"' in text
    assert '"replacement": "bar"' in text
    assert '"count": 3' in text
    assert '"entries"' in text


def test_load_unknown_source_raises(storage: VocabularyStorage) -> None:
    """An invalid source should raise a clear error."""
    with pytest.raises(ValueError):
        storage.load(VocabularySource("unknown"))  # type: ignore[arg-type]
