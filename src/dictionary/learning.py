"""Adaptive vocabulary learning from corrections and dictated text."""

from __future__ import annotations

import datetime
import json
import re
import threading
from typing import Any

from loguru import logger

from src.config import Settings
from src.dictionary.base import DictionaryEntry, VocabularySource
from src.dictionary.vocab_manager import VocabularyManager


class VocabularyLearner:
    """Learn new vocabulary terms from user corrections and dictated text."""

    _TOKEN_PATTERN: re.Pattern[str] = re.compile(r"[a-zA-Z0-9']+")
    _COMMON_WORDS: frozenset[str] = frozenset(
        {
            "the",
            "and",
            "for",
            "are",
            "but",
            "not",
            "you",
            "all",
            "can",
            "had",
            "her",
            "was",
            "one",
            "our",
            "out",
            "day",
            "get",
            "has",
            "him",
            "his",
            "how",
            "man",
            "new",
            "now",
            "old",
            "see",
            "two",
            "way",
            "who",
            "boy",
            "did",
            "its",
            "let",
            "put",
            "say",
            "she",
            "too",
            "use",
            "with",
            "have",
            "this",
            "will",
            "your",
            "from",
            "they",
            "know",
            "want",
            "been",
            "good",
            "much",
            "some",
            "time",
            "very",
            "when",
            "come",
            "here",
            "just",
            "like",
            "long",
            "make",
            "many",
            "over",
            "such",
            "take",
            "than",
            "them",
            "well",
            "were",
            "what",
            "would",
            "there",
            "their",
            "where",
            "being",
            "every",
            "great",
            "might",
            "shall",
            "still",
            "those",
            "while",
            "about",
            "could",
            "other",
            "after",
            "first",
            "never",
            "these",
            "think",
            "under",
            "water",
            "hello",
            "thank",
            "please",
            "dear",
            "regards",
        }
    )

    def __init__(self, vocab_manager: VocabularyManager, settings: Settings) -> None:
        """Initialize the learner.

        Args:
            vocab_manager: Loaded vocabulary manager used to persist learned terms.
            settings: Application settings containing ``vocab_dir``.
        """
        self._vocab_manager = vocab_manager
        self._vocab_dir = settings.vocab_dir
        self._vocab_dir.mkdir(parents=True, exist_ok=True)
        self._corrections_path = self._vocab_dir / "corrections.jsonl"
        self._term_counts_path = self._vocab_dir / "term_counts.json"
        self._term_counts: dict[str, int] = {}
        self._term_casings: dict[str, str] = {}
        self._lock = threading.RLock()
        self._load_term_counts()

    def record_correction(self, original: str, corrected: str) -> None:
        """Persist a single correction pair.

        Args:
            original: The original (incorrect) text.
            corrected: The corrected text.
        """
        if not corrected or corrected == original:
            return
        entry = {
            "original": original,
            "corrected": corrected,
            "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
        }
        with self._lock, self._corrections_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(entry, ensure_ascii=False) + "\n")
        logger.debug("Recorded correction: {} -> {}", original, corrected)

    def learn_from_corrections(self, min_count: int = 3) -> list[DictionaryEntry]:
        """Aggregate frequent corrections and promote them to user vocabulary.

        Args:
            min_count: Minimum occurrences before a correction is promoted.

        Returns:
            List of added or updated dictionary entries.
        """
        if not self._corrections_path.exists():
            return []

        counts: dict[tuple[str, str], int] = {}
        with self._lock, self._corrections_path.open("r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                original = self._normalize_term(str(data.get("original", "")))
                corrected = str(data.get("corrected", ""))
                if not original or not corrected:
                    continue
                key = (original, corrected)
                counts[key] = counts.get(key, 0) + 1

        promoted: list[DictionaryEntry] = []
        for (original, corrected), count in counts.items():
            if count >= min_count:
                entry = self._add_or_update_user_term(original, corrected, count)
                promoted.append(entry)

        if promoted:
            logger.info("Promoted {} correction(s) to user vocabulary", len(promoted))
        return promoted

    def learn_from_text(
        self,
        text: str,
        min_length: int = 4,
        min_frequency: int = 2,
    ) -> list[DictionaryEntry]:
        """Extract candidate terms from text and promote frequent ones.

        Args:
            text: Raw dictated text.
            min_length: Minimum term length to consider.
            min_frequency: Minimum total occurrences before promotion.

        Returns:
            List of newly promoted dictionary entries.
        """
        tokens = self._TOKEN_PATTERN.findall(text)
        new_promotions: list[DictionaryEntry] = []

        with self._lock:
            for token in tokens:
                if not self._is_candidate(token, min_length):
                    continue
                term = self._normalize_term(token)
                self._term_counts[term] = self._term_counts.get(term, 0) + 1
                self._term_casings[term] = token
                count = self._term_counts[term]
                if count == min_frequency:
                    entry = self._add_or_update_user_term(
                        term,
                        self._term_casings[term],
                        count,
                    )
                    new_promotions.append(entry)

            self._save_term_counts()

        if new_promotions:
            logger.info("Learned {} new term(s) from text", len(new_promotions))
        return new_promotions

    def _add_or_update_user_term(
        self,
        term: str,
        replacement: str,
        count: int,
    ) -> DictionaryEntry:
        """Add or update a user vocabulary entry preserving existing replacement.

        Args:
            term: Normalized term.
            replacement: Replacement text.
            count: Observed occurrence count.

        Returns:
            The resulting dictionary entry.
        """
        existing = self._vocab_manager.get_user_term(term)
        if existing is not None:
            replacement = existing.replacement
        self._vocab_manager.add_user_term(term, replacement, count=count)
        return DictionaryEntry(
            term=term,
            replacement=replacement,
            source=VocabularySource.user,
            count=count,
        )

    def _normalize_term(self, term: str) -> str:
        """Normalize a term for consistent counting and storage.

        Args:
            term: Raw term.

        Returns:
            Lowercase, stripped term.
        """
        return term.lower().strip()

    def _is_likely_proper_noun(self, term: str) -> bool:
        """Heuristic for proper nouns.

        Args:
            term: Raw term.

        Returns:
            True when the term looks like a proper noun.
        """
        if not term or not term[0].isupper():
            return False
        if term.isupper():
            return False
        return len(term) >= 4 or any(char.isupper() for char in term[1:])

    def _is_candidate(self, term: str, min_length: int) -> bool:
        """Decide whether a token should be tracked for learning.

        Args:
            term: Raw token.
            min_length: Minimum term length.

        Returns:
            True if the term is a learning candidate.
        """
        normalized = self._normalize_term(term)
        if len(normalized) < min_length:
            return False
        if normalized in self._COMMON_WORDS:
            return False
        return (
            self._is_likely_proper_noun(term)
            or any(char.isdigit() for char in term)
            or any(not (char.isalnum() or char == "'") for char in term)
            or any(char.isupper() for char in term[1:])
        )

    def _load_term_counts(self) -> None:
        """Load persisted term counts into memory."""
        if not self._term_counts_path.exists():
            self._term_counts = {}
            self._term_casings = {}
            return
        try:
            with self._lock:
                data = json.loads(self._term_counts_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            self._term_counts = {}
            self._term_casings = {}
            return
        self._term_counts = {str(key): int(value) for key, value in data.get("counts", {}).items()}
        self._term_casings = {
            str(key): str(value) for key, value in data.get("casings", {}).items()
        }

    def _save_term_counts(self) -> None:
        """Persist current in-memory term counts."""
        payload: dict[str, Any] = {
            "counts": self._term_counts,
            "casings": self._term_casings,
        }
        with self._lock:
            self._term_counts_path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
