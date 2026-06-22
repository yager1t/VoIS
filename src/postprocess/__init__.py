"""Post-processing factory and exports."""

from __future__ import annotations

from src.config import Settings
from src.postprocess.base import PostProcessor
from src.postprocess.formatter import TextFormatter
from src.postprocess.llm_client import LLMPostProcessor, OllamaClient


def create_post_processor(settings: Settings) -> PostProcessor:
    """Create a post-processor from application settings.

    Args:
        settings: Parsed application configuration.

    Returns:
        An ``LLMPostProcessor`` when LLM is enabled, otherwise a ``TextFormatter``.
    """
    if settings.llm_enabled:
        return LLMPostProcessor(settings)
    return TextFormatter()


__all__ = [
    "PostProcessor",
    "TextFormatter",
    "OllamaClient",
    "LLMPostProcessor",
    "create_post_processor",
]
