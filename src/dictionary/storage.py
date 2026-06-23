"""Persistent JSON storage for vocabulary entries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.dictionary.base import ContextMode, DictionaryEntry, VocabularySource
from src.dictionary.context_modes import get_context_vocabulary


class VocabularyStorage:
    """Load and save vocabulary entries as JSON files."""

    def __init__(self, vocab_dir: Path) -> None:
        """Initialize storage, creating the directory if needed.

        Args:
            vocab_dir: Directory where vocabulary JSON files are stored.
        """
        self.vocab_dir = vocab_dir
        self.vocab_dir.mkdir(parents=True, exist_ok=True)

    def _path_for(
        self,
        source: VocabularySource,
        context: ContextMode | None = None,
    ) -> Path:
        """Resolve the JSON file path for a source and optional context."""
        if source == VocabularySource.static:
            return self.vocab_dir / "static.json"
        if source == VocabularySource.user:
            return self.vocab_dir / "user.json"
        if source == VocabularySource.context:
            if context is None:
                msg = "context is required for context vocabulary"
                raise ValueError(msg)
            return self.vocab_dir / f"context_{context.value}.json"
        msg = f"Unknown vocabulary source: {source!r}"
        raise ValueError(msg)

    def _ensure_default_file(
        self,
        path: Path,
        source: VocabularySource,
        context: ContextMode | None = None,
    ) -> None:
        """Create a default file when one is missing."""
        entries: list[DictionaryEntry] = []
        if source == VocabularySource.context and context is not None:
            entries = [
                DictionaryEntry(
                    term=term,
                    replacement=term,
                    source=source,
                    context=context.value,
                )
                for term in get_context_vocabulary(context)
            ]
        self.save(entries, source, context)

    def load(
        self,
        source: VocabularySource,
        context: ContextMode | None = None,
    ) -> list[DictionaryEntry]:
        """Load entries for the given source.

        Missing files are created with sensible defaults.
        """
        path = self._path_for(source, context)
        if not path.exists():
            self._ensure_default_file(path, source, context)

        data = json.loads(path.read_text(encoding="utf-8"))
        return [self._entry_from_dict(item, source) for item in data.get("entries", [])]

    def save(
        self,
        entries: list[DictionaryEntry],
        source: VocabularySource,
        context: ContextMode | None = None,
    ) -> None:
        """Persist entries for the given source."""
        path = self._path_for(source, context)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "entries": [self._entry_to_dict(entry) for entry in entries],
        }
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @staticmethod
    def _entry_to_dict(entry: DictionaryEntry) -> dict[str, object]:
        return {
            "term": entry.term,
            "replacement": entry.replacement,
            "source": entry.source.value,
            "count": entry.count,
            "context": entry.context,
        }

    @classmethod
    def _entry_from_dict(
        cls,
        item: dict[str, Any],
        source: VocabularySource,
    ) -> DictionaryEntry:
        return DictionaryEntry(
            term=str(item["term"]),
            replacement=str(item["replacement"]),
            source=source,
            count=int(item.get("count", 1)),
            context=item.get("context") if item.get("context") is not None else None,
        )
