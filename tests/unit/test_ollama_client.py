"""Tests for the Ollama-backed LLM client and post-processor."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest

from src.config import Settings
from src.postprocess.llm_client import LLMPostProcessor, OllamaClient


@pytest.fixture
def client() -> OllamaClient:
    """Return an OllamaClient configured for fast unit tests."""
    return OllamaClient(base_url="http://test-ollama", model="test-model", timeout=1.0)


def _make_response(content: str, status_code: int = 200) -> MagicMock:
    """Return a mock httpx.Response with the given JSON body."""
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = {"message": {"content": content}}
    response.raise_for_status.return_value = None
    return response


def test_improve_returns_llm_content(client: OllamaClient) -> None:
    """A successful response should return the LLM content."""
    response = _make_response("Improved text.")

    with patch("httpx.post", return_value=response) as mock_post:
        result = client.improve("raw text")

    assert result == "Improved text."
    mock_post.assert_called_once()
    call_kwargs = mock_post.call_args.kwargs
    assert call_kwargs["json"]["model"] == "test-model"
    assert call_kwargs["json"]["stream"] is False
    assert call_kwargs["timeout"] == 1.0


def test_improve_uses_custom_instruction(client: OllamaClient) -> None:
    """A custom instruction should override the default user prompt."""
    response = _make_response("Formal text.")

    with patch("httpx.post", return_value=response) as mock_post:
        result = client.improve("raw text", instruction="Make it formal")

    assert result == "Formal text."
    messages = mock_post.call_args.kwargs["json"]["messages"]
    assert messages[1]["content"] == "Make it formal"


def test_improve_timeout_returns_original(client: OllamaClient) -> None:
    """A timeout should fall back to the original text."""
    with patch("httpx.post", side_effect=httpx.TimeoutException("timed out")):
        result = client.improve("keep me")

    assert result == "keep me"


def test_improve_connection_error_returns_original(client: OllamaClient) -> None:
    """A connection error should fall back to the original text."""
    with patch("httpx.post", side_effect=httpx.ConnectError("refused")):
        result = client.improve("keep me")

    assert result == "keep me"


def test_improve_http_error_returns_original(client: OllamaClient) -> None:
    """An HTTP error response should fall back to the original text."""
    response = MagicMock()
    response.raise_for_status.side_effect = httpx.HTTPError("500")

    with patch("httpx.post", return_value=response):
        result = client.improve("keep me")

    assert result == "keep me"


def test_improve_invalid_json_returns_original(client: OllamaClient) -> None:
    """Invalid JSON in the response should fall back to the original text."""
    response = _make_response("ok")
    response.json.side_effect = ValueError("not json")

    with patch("httpx.post", return_value=response):
        result = client.improve("keep me")

    assert result == "keep me"


def test_improve_missing_message_returns_original(client: OllamaClient) -> None:
    """A valid JSON without the expected message key should fall back."""
    response = _make_response("irrelevant")
    response.json.return_value = {"unexpected": "shape"}

    with patch("httpx.post", return_value=response):
        result = client.improve("keep me")

    assert result == "keep me"


def test_llm_post_processor_uses_settings() -> None:
    """LLMPostProcessor should configure its client from Settings."""
    settings = Settings(
        llm_enabled=True,
        llm_url="http://custom-ollama",
        llm_model="custom-model",
        llm_timeout=2.0,
        llm_prompt="Custom prompt.",
    )
    processor = LLMPostProcessor(settings)

    assert processor.client.base_url == "http://custom-ollama"
    assert processor.client.model == "custom-model"
    assert processor.client.timeout == 2.0
    assert processor.client.system_prompt == "Custom prompt."


def test_llm_post_processor_process_calls_improve() -> None:
    """process() should delegate to the underlying client."""
    processor = LLMPostProcessor(Settings(llm_enabled=True))

    with patch.object(processor.client, "improve", return_value="improved") as mock_improve:
        result = processor.process("raw")

    assert result == "improved"
    mock_improve.assert_called_once_with("raw")
