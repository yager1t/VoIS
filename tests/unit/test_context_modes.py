"""Tests for context mode helpers."""

from __future__ import annotations

import pytest

from src.dictionary.base import ContextMode
from src.dictionary.context_modes import (
    get_context_prompt,
    get_context_vocabulary,
    parse_context_mode,
)


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        (ContextMode.general, ""),
        (ContextMode.chat, "casual chat message"),
        (ContextMode.email, "formal email"),
        (ContextMode.code, "code snippet or technical text"),
    ],
)
def test_get_context_prompt(mode: ContextMode, expected: str) -> None:
    """Prompts should match the documented fragments."""
    assert get_context_prompt(mode) == expected


def test_get_context_vocabulary_has_starters() -> None:
    """Code, chat, and email modes should provide starter terms."""
    assert "function" in get_context_vocabulary(ContextMode.code)
    assert "lol" in get_context_vocabulary(ContextMode.chat)
    assert "Sincerely" in get_context_vocabulary(ContextMode.email)
    assert get_context_vocabulary(ContextMode.general) == []


def test_parse_context_mode_accepts_strings() -> None:
    """String values should parse to the matching enum member."""
    assert parse_context_mode("code") == ContextMode.code
    assert parse_context_mode("EMAIL") == ContextMode.email
    assert parse_context_mode("General") == ContextMode.general


def test_parse_context_mode_rejects_unknown() -> None:
    """Unknown strings should raise a ValueError."""
    with pytest.raises(ValueError, match="Unknown context mode"):
        parse_context_mode("unknown")
