"""High-level vocabulary manager used by the application."""

from __future__ import annotations

from src.config import Settings
from src.dictionary.base import ContextMode, DictionaryEntry, VocabularySource
from src.dictionary.context_modes import parse_context_mode
from src.dictionary.storage import VocabularyStorage


class VocabularyManager:
    """Manages static, context, and user vocabulary with override priority."""

    def __init__(self, settings: Settings) -> None:
        """Initialize the manager from application settings.

        Args:
            settings: Application settings containing ``vocab_dir`` and
                ``context_mode``.
        """
        self._storage = VocabularyStorage(settings.vocab_dir)
        self._context_mode = parse_context_mode(settings.context_mode)
        self._dictionary_enabled = settings.dictionary_enabled

        self._static: dict[str, DictionaryEntry] = {}
        self._context: dict[str, DictionaryEntry] = {}
        self._user: dict[str, DictionaryEntry] = {}

    def load_all(self) -> None:
        """Load static, user, and current-context vocabulary."""
        self._static = self._load_source(VocabularySource.static)
        self._user = self._load_source(VocabularySource.user)
        self._context = self._load_source(
            VocabularySource.context,
            self._context_mode,
        )

    def _load_source(
        self,
        source: VocabularySource,
        context: ContextMode | None = None,
    ) -> dict[str, DictionaryEntry]:
        """Load entries and index them by term."""
        entries = self._storage.load(source, context)
        return {entry.term: entry for entry in entries}

    def get_replacements(self) -> dict[str, str]:
        """Return term -> replacement for all loaded sources.

        Override priority: user > context > static.
        """
        if not self._dictionary_enabled:
            return {}

        merged: dict[str, str] = {}
        for entry in self._static.values():
            merged[entry.term] = entry.replacement
        for entry in self._context.values():
            merged[entry.term] = entry.replacement
        for entry in self._user.values():
            merged[entry.term] = entry.replacement
        return merged

    def get_terms(self, context: ContextMode | None = None) -> list[str]:
        """Return all loaded terms, optionally filtered by context."""
        terms: set[str] = set()
        for entry in (*self._static.values(), *self._context.values(), *self._user.values()):
            if context is None or entry.context == context.value:
                terms.add(entry.term)
        return sorted(terms)

    def add_user_term(self, term: str, replacement: str) -> None:
        """Add or update a user-level vocabulary entry."""
        self._user[term] = DictionaryEntry(
            term=term,
            replacement=replacement,
            source=VocabularySource.user,
        )
        self._save_user()

    def remove_user_term(self, term: str) -> None:
        """Remove a user-level vocabulary entry if it exists."""
        if term in self._user:
            del self._user[term]
            self._save_user()

    def _save_user(self) -> None:
        """Persist the current user vocabulary."""
        self._storage.save(list(self._user.values()), VocabularySource.user)

    def set_context_mode(self, mode: ContextMode | str) -> None:
        """Switch the active context mode and reload context vocabulary."""
        if isinstance(mode, str):
            mode = parse_context_mode(mode)
        self._context_mode = mode
        self._context = self._load_source(VocabularySource.context, mode)

    def get_context_mode(self) -> ContextMode:
        """Return the active context mode."""
        return self._context_mode
