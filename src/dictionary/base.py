"""Core vocabulary data models."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class VocabularySource(StrEnum):
    """Origin of a vocabulary entry."""

    static = "static"
    user = "user"
    context = "context"


class ContextMode(StrEnum):
    """Dictation context that influences vocabulary and post-processing."""

    general = "general"
    chat = "chat"
    email = "email"
    code = "code"


@dataclass
class DictionaryEntry:
    """A single term/replacement pair with metadata."""

    term: str
    replacement: str
    source: VocabularySource
    count: int = 1
    context: str | None = None
