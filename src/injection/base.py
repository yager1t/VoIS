"""Abstract interface for text injectors."""

from __future__ import annotations

from abc import ABC, abstractmethod


class TextInjector(ABC):
    """Abstract base for cross-platform text injection.

    Implementations emulate keyboard input so that transcribed speech can be
    inserted at the current cursor position in the active application.
    """

    @abstractmethod
    def inject(self, text: str) -> None:
        """Type the given text at the current cursor position.

        Args:
            text: Unicode text to inject.
        """

    @abstractmethod
    def inject_with_delay(self, text: str, delay_ms: float = 0.0) -> None:
        """Type the given text with an optional per-character delay.

        Args:
            text: Unicode text to inject.
            delay_ms: Milliseconds to sleep between individual characters.
        """

    @classmethod
    @abstractmethod
    def is_available(cls) -> bool:
        """Return whether this injector works on the current platform."""
