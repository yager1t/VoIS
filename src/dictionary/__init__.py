"""Dictionary and vocabulary management for Voice-to-Cursor."""

from src.dictionary.base import ContextMode, DictionaryEntry, VocabularySource
from src.dictionary.context_modes import (
    get_context_prompt,
    get_context_vocabulary,
    parse_context_mode,
)
from src.dictionary.vocab_manager import VocabularyManager

__all__ = [
    "ContextMode",
    "DictionaryEntry",
    "VocabularyManager",
    "VocabularySource",
    "get_context_prompt",
    "get_context_vocabulary",
    "parse_context_mode",
]
