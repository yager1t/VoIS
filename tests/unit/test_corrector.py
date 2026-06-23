"""Unit tests for the text corrector."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.dictionary.corrector import TextCorrector, _preserve_case


@pytest.fixture
def corrector() -> TextCorrector:
    """Return a TextCorrector with a mocked vocabulary manager."""
    vocab_manager = MagicMock()
    vocab_manager.get_replacements.return_value = {
        "acme": "ACME Corporation",
        "cat": "dog",
        "best regards": "Kind regards",
        "api": "API",
        "upper": "lower",
    }
    return TextCorrector(vocab_manager)


def test_basic_replacement(corrector: TextCorrector) -> None:
    """A known term should be replaced with its dictionary entry."""
    assert corrector.correct("I work at acme") == "I work at ACME Corporation"


def test_word_boundary_protection(corrector: TextCorrector) -> None:
    """Terms should not be replaced when they appear inside larger words."""
    assert corrector.correct("The category of cat") == "The category of dog"


def test_longest_match_priority(corrector: TextCorrector) -> None:
    """Longer phrases should win over shorter sub-phrases."""
    assert corrector.correct("best regards team") == "Kind regards team"


def test_case_preservation_lower(corrector: TextCorrector) -> None:
    """Lowercase matches should use the replacement as-is."""
    assert corrector.correct("the api is ready") == "the API is ready"


def test_case_preservation_title(corrector: TextCorrector) -> None:
    """Title-case matches should title-case the replacement."""
    assert corrector.correct("The Api is ready") == "The Api is ready"


def test_case_preservation_upper(corrector: TextCorrector) -> None:
    """Uppercase matches should uppercase the replacement."""
    assert corrector.correct("the UPPER limit") == "the LOWER limit"


def test_empty_text_returns_empty(corrector: TextCorrector) -> None:
    """An empty string should be returned unchanged."""
    assert corrector.correct("") == ""


def test_term_equals_replacement_is_no_op() -> None:
    """Entries where term equals replacement should be ignored."""
    vocab_manager = MagicMock()
    vocab_manager.get_replacements.return_value = {"same": "same"}
    corrector = TextCorrector(vocab_manager)

    assert corrector.correct("same") == "same"


def test_multiple_replacements(corrector: TextCorrector) -> None:
    """Several distinct terms should be replaced in one pass."""
    assert corrector.correct("acme has a cat") == "ACME Corporation has a dog"


def test_preserve_case_unchanged_for_mixed_case() -> None:
    """Mixed-case originals should return the replacement unchanged."""
    assert _preserve_case("aPi", "API") == "API"


def test_no_replacements_returns_text_unchanged() -> None:
    """An empty vocabulary should leave the text unchanged."""
    vocab_manager = MagicMock()
    vocab_manager.get_replacements.return_value = {}
    corrector = TextCorrector(vocab_manager)

    assert corrector.correct("hello world") == "hello world"
