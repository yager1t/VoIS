"""ASR biasing utilities using dictionary hotwords and initial prompts."""

from __future__ import annotations

from src.config import Settings
from src.dictionary.base import ContextMode
from src.dictionary.context_modes import get_context_prompt, get_context_vocabulary
from src.dictionary.vocab_manager import VocabularyManager

_DEFAULT_MAX_HOTWORDS = 30
_DEFAULT_MAX_PROMPT_LENGTH = 250


class ASRBias:
    """Builds Whisper-compatible ``initial_prompt`` and ``hotwords`` hints.

    The bias is derived from the active context mode and loaded vocabulary.
    When the dictionary is disabled, both hints are empty.
    """

    def __init__(
        self,
        vocab_manager: VocabularyManager,
        settings: Settings,
    ) -> None:
        """Initialize with a loaded vocabulary manager and settings.

        Args:
            vocab_manager: Manager that provides vocabulary terms.
            settings: Application settings, including ``dictionary_enabled``.
        """
        self._vocab_manager = vocab_manager
        self._settings = settings
        self._dictionary_enabled = settings.dictionary_enabled

    def _truncate(self, text: str, max_length: int) -> str:
        """Return ``text`` truncated to ``max_length`` without breaking words.

        Args:
            text: Input string.
            max_length: Maximum allowed length in characters.

        Returns:
            Truncated string ending with an ellipsis when trimmed.
        """
        if len(text) <= max_length:
            return text
        trimmed = text[:max_length]
        last_space = trimmed.rfind(" ")
        if last_space > 0:
            trimmed = trimmed[:last_space]
        return trimmed.rstrip() + "…"

    def initial_prompt(self, context: ContextMode | None = None) -> str:
        """Build a short initial prompt for the ASR model.

        Args:
            context: Optional context mode override. Defaults to the manager's
                active context mode.

        Returns:
            A prompt string combining the context description and important
            vocabulary terms, truncated to avoid overloading the model.
        """
        if context is None:
            context = self._vocab_manager.get_context_mode()

        context_prompt = get_context_prompt(context)
        if not self._dictionary_enabled:
            return context_prompt

        terms = self.hotwords(context=context)
        if not terms:
            return context_prompt

        terms_text = ", ".join(terms)
        if context_prompt:
            prompt = f"Transcribe a {context_prompt}. Important terms: {terms_text}."
        else:
            prompt = f"Important terms: {terms_text}."

        return self._truncate(prompt, _DEFAULT_MAX_PROMPT_LENGTH)

    def hotwords(self, context: ContextMode | None = None) -> list[str]:
        """Return a list of vocabulary terms for ASR biasing.

        Args:
            context: Optional context mode override.

        Returns:
            Up to ``_DEFAULT_MAX_HOTWORDS`` terms sorted by descending count and
            then alphabetically. Empty when the dictionary is disabled.
        """
        if not self._dictionary_enabled:
            return []

        if context is None:
            context = self._vocab_manager.get_context_mode()

        starter_terms = get_context_vocabulary(context)
        vocab_terms = self._vocab_manager.get_terms(context=context)

        all_terms = sorted(set(starter_terms) | set(vocab_terms))
        entries = {
            entry.term: entry
            for source in (
                self._vocab_manager._static,
                self._vocab_manager._context,
                self._vocab_manager._user,
            )
            for entry in source.values()
        }

        def _sort_key(term: str) -> tuple[int, str]:
            entry = entries.get(term)
            count = entry.count if entry is not None else 1
            return (-count, term.lower())

        ranked = sorted(all_terms, key=_sort_key)
        return ranked[:_DEFAULT_MAX_HOTWORDS]
