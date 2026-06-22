"""Tests for the post-processor factory."""

from src.config import Settings
from src.postprocess import create_post_processor
from src.postprocess.formatter import TextFormatter
from src.postprocess.llm_client import LLMPostProcessor


def test_factory_returns_text_formatter_by_default() -> None:
    """When LLM is disabled the factory should return a TextFormatter."""
    settings = Settings(llm_enabled=False)

    processor = create_post_processor(settings)

    assert isinstance(processor, TextFormatter)


def test_factory_returns_llm_post_processor_when_enabled() -> None:
    """When LLM is enabled the factory should return an LLMPostProcessor."""
    settings = Settings(llm_enabled=True)

    processor = create_post_processor(settings)

    assert isinstance(processor, LLMPostProcessor)
