"""Text correction using loaded vocabulary."""

from __future__ import annotations

import re

from src.dictionary.base import ContextMode
from src.dictionary.vocab_manager import VocabularyManager


def _preserve_case(original: str, replacement: str) -> str:
    """Apply the original term's casing to the replacement.

    Args:
        original: The matched term as it appeared in the text.
        replacement: The dictionary replacement for the term.

    Returns:
        The replacement with casing preserved from ``original``.
    """
    if original.isupper():
        return replacement.upper()
    if original.istitle():
        return replacement.title()
    return replacement


class TextCorrector:
    """Apply vocabulary replacements to raw transcripts."""

    def __init__(self, vocab_manager: VocabularyManager) -> None:
        """Initialize the corrector from a vocabulary manager.

        Args:
            vocab_manager: Loaded vocabulary manager providing term replacements.
        """
        self._vocab_manager = vocab_manager

    def correct(self, text: str, context: ContextMode | None = None) -> str:
        """Replace known terms in ``text`` with their dictionary replacements.

        Replacements are applied longest-match first using word-boundary checks,
        so shorter terms do not corrupt longer words or phrases. The original
        casing of each matched term is preserved on the replacement.

        Args:
            text: Raw transcript text to correct.
            context: Optional context mode (reserved for future filtering).

        Returns:
            Corrected text.
        """
        del context  # Reserved for future context-specific correction.

        if not text:
            return text

        replacements = self._vocab_manager.get_replacements()
        # Skip no-op entries to avoid building an unnecessary regex.
        replacements = {term: repl for term, repl in replacements.items() if term != repl}
        if not replacements:
            return text

        # Longest-match first: sort by length descending so longer phrases win.
        terms = sorted(
            replacements.items(),
            key=lambda item: len(item[0]),
            reverse=True,
        )

        patterns: list[str] = []
        for term, _ in terms:
            escaped_words = [re.escape(word) for word in term.split()]
            phrase = r"\s+".join(escaped_words)
            patterns.append(r"(?<!\w)" + phrase + r"(?!\w)")

        regex = re.compile("|".join(patterns), flags=re.IGNORECASE)

        def _replace(match: re.Match[str]) -> str:
            matched = match.group(0)
            matched_lower = matched.lower()
            for term, repl in terms:
                if term.lower() == matched_lower:
                    return _preserve_case(matched, repl)
            return matched

        return regex.sub(_replace, text)
