"""Tests for the deterministic text formatter."""

import pytest

from src.postprocess.formatter import TextFormatter


@pytest.fixture
def formatter() -> TextFormatter:
    """Return a fresh TextFormatter instance."""
    return TextFormatter()


def test_strip_leading_and_trailing_whitespace(formatter: TextFormatter) -> None:
    """Leading and trailing whitespace should be removed."""
    assert formatter.process("  hello world  ") == "Hello world."


def test_collapse_multiple_spaces(formatter: TextFormatter) -> None:
    """Multiple consecutive spaces should collapse to a single space."""
    assert formatter.process("hello   world") == "Hello world."


def test_collapse_tabs_and_newlines(formatter: TextFormatter) -> None:
    """Tabs and newlines should be normalized to a single space."""
    assert formatter.process("hello\t\tworld\nnext") == "Hello world next."


def test_capitalize_first_letter(formatter: TextFormatter) -> None:
    """The first letter should be capitalized."""
    assert formatter.process("hello world") == "Hello world."


def test_add_period_when_missing(formatter: TextFormatter) -> None:
    """A period should be appended when no terminal punctuation exists."""
    assert formatter.process("hello world") == "Hello world."


def test_preserve_existing_period(formatter: TextFormatter) -> None:
    """Existing periods should not be duplicated."""
    assert formatter.process("hello world.") == "Hello world."


def test_preserve_exclamation_and_question(formatter: TextFormatter) -> None:
    """Exclamation and question marks should be preserved."""
    assert formatter.process("hello world!") == "Hello world!"
    assert formatter.process("hello world?") == "Hello world?"


def test_empty_string_returns_empty(formatter: TextFormatter) -> None:
    """Empty input should remain empty."""
    assert formatter.process("") == ""


def test_whitespace_only_returns_empty(formatter: TextFormatter) -> None:
    """Whitespace-only input should return an empty string."""
    assert formatter.process("   \t\n  ") == ""


def test_context_argument_is_ignored(formatter: TextFormatter) -> None:
    """TextFormatter should accept but ignore the optional context argument."""
    assert formatter.process("hello world", context="email") == "Hello world."
