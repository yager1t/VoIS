"""Unit tests for the adaptive vocabulary learner."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.config import Settings
from src.dictionary.base import VocabularySource
from src.dictionary.learning import VocabularyLearner
from src.dictionary.vocab_manager import VocabularyManager


@pytest.fixture
def vocab_manager(tmp_path: Path) -> VocabularyManager:
    """Return a loaded vocabulary manager using an isolated directory."""
    settings = Settings(vocab_dir=tmp_path / "vocab")
    manager = VocabularyManager(settings)
    manager.load_all()
    return manager


@pytest.fixture
def learner(vocab_manager: VocabularyManager, tmp_path: Path) -> VocabularyLearner:
    """Return a learner wired to the isolated vocabulary manager."""
    settings = Settings(vocab_dir=tmp_path / "vocab")
    return VocabularyLearner(vocab_manager, settings)


def test_record_correction_appends_to_jsonl(
    learner: VocabularyLearner,
    tmp_path: Path,
) -> None:
    """A correction should be appended as a JSON line with a timestamp."""
    learner.record_correction("kubertnetes", "Kubernetes")

    corrections_path = tmp_path / "vocab" / "corrections.jsonl"
    assert corrections_path.exists()
    lines = corrections_path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 1

    data = json.loads(lines[0])
    assert data["original"] == "kubertnetes"
    assert data["corrected"] == "Kubernetes"
    assert "timestamp" in data


def test_record_correction_skips_empty_or_equal(
    learner: VocabularyLearner,
    tmp_path: Path,
) -> None:
    """Empty or identity corrections should not be recorded."""
    learner.record_correction("foo", "")
    learner.record_correction("bar", "bar")

    corrections_path = tmp_path / "vocab" / "corrections.jsonl"
    assert not corrections_path.exists()


def test_learn_from_corrections_promotes_frequent_pairs(
    learner: VocabularyLearner,
    vocab_manager: VocabularyManager,
) -> None:
    """Corrections that meet the count threshold become user vocabulary."""
    for _ in range(3):
        learner.record_correction("kubertnetes", "Kubernetes")
    learner.record_correction("foo", "bar")  # below threshold

    promoted = learner.learn_from_corrections(min_count=3)

    assert len(promoted) == 1
    entry = promoted[0]
    assert entry.term == "kubertnetes"
    assert entry.replacement == "Kubernetes"
    assert entry.source == VocabularySource.user
    assert entry.count == 3

    user_entry = vocab_manager.get_user_term("kubertnetes")
    assert user_entry is not None
    assert user_entry.replacement == "Kubernetes"
    assert user_entry.count == 3


def test_learn_from_text_promotes_capitalized_terms(
    learner: VocabularyLearner,
    vocab_manager: VocabularyManager,
) -> None:
    """Capitalized terms are tracked and promoted once they reach min_frequency."""
    learner.learn_from_text("I use Kubernetes every day", min_frequency=2)
    assert vocab_manager.get_user_term("kubernetes") is None

    learner.learn_from_text("Kubernetes is great", min_frequency=2)

    entry = vocab_manager.get_user_term("kubernetes")
    assert entry is not None
    assert entry.replacement == "Kubernetes"
    assert entry.count == 2


def test_learn_from_text_promotes_mixed_case_and_digits(
    learner: VocabularyLearner,
    vocab_manager: VocabularyManager,
) -> None:
    """Mixed-case terms and terms with digits are also learned."""
    learner.learn_from_text("The gRPC endpoint uses PyQt6", min_frequency=2)
    learner.learn_from_text("gRPC and PyQt6 are used here", min_frequency=2)

    assert vocab_manager.get_user_term("grpc") is not None
    assert vocab_manager.get_user_term("pyqt6") is not None


def test_learn_from_text_ignores_common_words(
    learner: VocabularyLearner,
    vocab_manager: VocabularyManager,
) -> None:
    """Common lowercase words should not be promoted."""
    for _ in range(3):
        learner.learn_from_text("The quick brown fox", min_frequency=2)

    assert vocab_manager.get_user_term("the") is None
    assert vocab_manager.get_user_term("quick") is None


def test_learn_from_text_persists_term_counts(
    vocab_manager: VocabularyManager,
    tmp_path: Path,
) -> None:
    """Term counts should survive learner recreation."""
    settings = Settings(vocab_dir=tmp_path / "vocab")
    learner1 = VocabularyLearner(vocab_manager, settings)
    learner1.learn_from_text("Kubernetes is great", min_frequency=2)

    learner2 = VocabularyLearner(vocab_manager, settings)
    learner2.learn_from_text("I use Kubernetes today", min_frequency=2)

    entry = vocab_manager.get_user_term("kubernetes")
    assert entry is not None
    assert entry.count == 2

    counts_path = tmp_path / "vocab" / "term_counts.json"
    assert counts_path.exists()
    data = json.loads(counts_path.read_text(encoding="utf-8"))
    assert data["counts"]["kubernetes"] == 2
