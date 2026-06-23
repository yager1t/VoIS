"""Context-specific prompts and starter vocabulary."""

from __future__ import annotations

from src.dictionary.base import ContextMode

_CONTEXT_PROMPTS: dict[ContextMode, str] = {
    ContextMode.general: "",
    ContextMode.chat: "casual chat message",
    ContextMode.email: "formal email",
    ContextMode.code: "code snippet or technical text",
}

_CONTEXT_STARTERS: dict[ContextMode, list[str]] = {
    ContextMode.general: [],
    ContextMode.chat: [
        "lol",
        "btw",
        "imo",
        "idk",
        "np",
    ],
    ContextMode.email: [
        "Best regards",
        "Sincerely",
        "Thank you",
        "Please find attached",
        "Looking forward",
    ],
    ContextMode.code: [
        "function",
        "return",
        "class",
        "import",
        "def",
        "async",
        "await",
        "lambda",
    ],
}


def get_context_prompt(mode: ContextMode) -> str:
    """Return a short prompt fragment describing the context."""
    return _CONTEXT_PROMPTS.get(mode, "")


def get_context_vocabulary(mode: ContextMode) -> list[str]:
    """Return starter terms for the given context mode."""
    return list(_CONTEXT_STARTERS.get(mode, []))


def parse_context_mode(value: str) -> ContextMode:
    """Convert a string to a ``ContextMode``.

    Args:
        value: One of the ``ContextMode`` values.

    Returns:
        The matching ``ContextMode``.

    Raises:
        ValueError: If ``value`` does not map to a known context mode.
    """
    try:
        return ContextMode(value.lower())
    except ValueError as exc:
        msg = f"Unknown context mode: {value!r}"
        raise ValueError(msg) from exc
