"""Abstract interface for post-processors."""

from __future__ import annotations

from abc import ABC, abstractmethod


class PostProcessor(ABC):
    """Abstract base for text post-processing.

    Implementations transform or improve raw ASR transcripts before they are
    injected at the cursor.
    """

    @abstractmethod
    def process(self, text: str, context: str | None = None) -> str:
        """Process the given text and return the improved result.

        Args:
            text: Raw text from the ASR provider.
            context: Optional context to guide the transformation.

        Returns:
            Transformed text ready for injection.
        """
