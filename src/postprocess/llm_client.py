"""Ollama-backed LLM post-processing client."""

from __future__ import annotations

import httpx
from loguru import logger

from src.config import Settings
from src.postprocess.base import PostProcessor

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant. Fix grammar, punctuation, and formatting "
    "of the user's text. Keep the original language. Do not add explanations. "
    "Return only the improved text."
)


class OllamaClient:
    """Simple synchronous client for the Ollama chat API."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3",
        timeout: float = 5.0,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ) -> None:
        """Initialize the Ollama client.

        Args:
            base_url: Base URL of the Ollama server.
            model: Model name to use for chat completions.
            timeout: Request timeout in seconds.
            system_prompt: System message sent with every request.
        """
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.system_prompt = system_prompt

    def improve(self, text: str, instruction: str | None = None) -> str:
        """Ask the LLM to improve ``text``.

        On any error the original ``text`` is returned unchanged.

        Args:
            text: Raw text to improve.
            instruction: Optional override for the user prompt.

        Returns:
            Improved text from the LLM, or the original text on failure.
        """
        prompt = instruction if instruction is not None else f"Improve this text:\n{text}"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
        }

        try:
            response = httpx.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            return str(response.json()["message"]["content"])
        except (httpx.TimeoutException, httpx.ConnectError, httpx.HTTPError) as exc:
            logger.warning("Ollama request failed ({}); falling back to raw text", exc)
            return text
        except Exception as exc:  # pragma: no cover - broad safety net
            logger.warning("Ollama post-processing failed ({}); falling back to raw text", exc)
            return text


class LLMPostProcessor(PostProcessor):
    """Post-processor that improves transcripts via an Ollama LLM."""

    def __init__(self, settings: Settings) -> None:
        """Initialize the LLM post-processor from settings.

        Args:
            settings: Application settings containing LLM configuration.
        """
        self.client = OllamaClient(
            base_url=settings.llm_url,
            model=settings.llm_model,
            timeout=settings.llm_timeout,
            system_prompt=settings.llm_prompt,
        )

    def process(self, text: str, context: str | None = None) -> str:  # noqa: ARG002
        """Improve ``text`` using the configured LLM.

        Args:
            text: Raw text from the ASR provider.
            context: Unused by this post-processor.

        Returns:
            Improved text, or the original text when the LLM is unavailable.
        """
        return self.client.improve(text)
