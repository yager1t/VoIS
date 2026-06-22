"""Abstract interface and helpers for global hotkey managers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable


class HotkeyManager(ABC):
    """Abstract base for cross-platform global hotkey managers.

    Subclasses must implement the lifecycle methods and invoke
    :attr:`on_press` / :attr:`on_release` according to the configured
    ``push_to_talk`` mode.
    """

    def __init__(self, hotkey: str, push_to_talk: bool = True) -> None:
        """Initialize the manager with a hotkey and interaction mode.

        Args:
            hotkey: Human-readable hotkey such as ``"f9"`` or ``"<ctrl>+f9"``.
            push_to_talk: If ``True``, press/release events are emitted separately.
                If ``False``, only a press event is emitted and the caller toggles.
        """
        self.hotkey = hotkey.strip().lower()
        self.push_to_talk = push_to_talk
        self.on_press: Callable[[], None] | None = None
        self.on_release: Callable[[], None] | None = None
        self._modifiers, self._main_key = parse_hotkey(self.hotkey)

    @abstractmethod
    def start(self) -> None:
        """Start listening for the global hotkey."""

    @abstractmethod
    def stop(self) -> None:
        """Stop listening and release resources."""

    @abstractmethod
    def is_alive(self) -> bool:
        """Return whether the listener is currently active."""


def parse_hotkey(hotkey: str) -> tuple[set[str], str]:
    """Parse a hotkey string into modifiers and a main key.

    Supported forms::

        f9
        <ctrl>+f9
        ctrl+f9
        <ctrl>+<alt>+f9

    Args:
        hotkey: Hotkey string to parse.

    Returns:
        A tuple of ``(modifiers, main_key)`` where ``modifiers`` is a set of
        normalized modifier names and ``main_key`` is the normalized key name.

    Raises:
        ValueError: If ``hotkey`` is empty or contains no main key.
    """
    normalized = hotkey.strip().lower()
    if not normalized:
        raise ValueError("hotkey must not be empty")

    parts = [part.strip() for part in normalized.split("+")]
    parts = [part[1:-1] if part.startswith("<") and part.endswith(">") else part for part in parts]
    parts = [part for part in parts if part]

    if not parts:
        raise ValueError("hotkey must contain at least one key")

    main_key = parts[-1]
    modifiers = set(parts[:-1])

    valid_modifiers = {"ctrl", "alt", "shift", "cmd", "win", "super"}
    unknown = modifiers - valid_modifiers
    if unknown:
        raise ValueError(f"unknown modifier(s): {', '.join(sorted(unknown))}")

    return modifiers, main_key
