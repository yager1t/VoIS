"""Deterministic text formatter used as the default post-processor."""

from __future__ import annotations

import re

from src.postprocess.base import PostProcessor


class TextFormatter(PostProcessor):
    """Simple deterministic formatter for ASR transcripts.

    Strips whitespace, collapses multiple spaces, capitalizes the first letter,
    and appends a period when the text does not already end with punctuation.
    """

    def process(self, text: str, context: str | None = None) -> str:  # noqa: ARG002
        """Format ``text`` deterministically.

        Args:
            text: Raw text from the ASR provider.
            context: Unused by this formatter.

        Returns:
            Trimmed, normalized, and lightly punctuated text.
        """
        text = text.strip()
        if not text:
            return text

        text = re.sub(r"\s+", " ", text)
        text = text[0].upper() + text[1:]

        if text[-1] not in {".", "!", "?"}:
            text = f"{text}."

        return text
