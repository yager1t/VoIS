"""Abstract interface and shared types for ASR providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from src.config import Settings


@dataclass
class TranscriptionResult:
    """Result of a transcription request.

    Attributes:
        text: Recognized text. Empty string when no speech is detected.
        is_final: Whether this result represents a final (committed) transcript.
        confidence: Optional aggregated confidence score between 0.0 and 1.0.
        language: Detected or configured language code (e.g. ``en``).
    """

    text: str
    is_final: bool = True
    confidence: float | None = None
    language: str | None = None


class ASRProvider(ABC):
    """Abstract base class for automatic speech recognition backends."""

    def __init__(self, settings: Settings) -> None:
        """Initialize the provider with validated application settings.

        Args:
            settings: Parsed application configuration.
        """
        self.settings = settings

    @abstractmethod
    def load_model(self) -> None:
        """Load or warm up the underlying ASR model.

        Implementations are encouraged to perform this lazily, but callers may
        invoke this method to preload a model before recording starts.
        """

    @abstractmethod
    def warmup(self) -> None:
        """Force model load without real audio.

        The default implementation is a no-op; providers that need eager model
        loading can override it.
        """

    @abstractmethod
    def transcribe(self, audio: np.ndarray, sample_rate: int) -> TranscriptionResult:
        """Transcribe a complete audio clip into text.

        Args:
            audio: One-dimensional ``float32`` audio samples in the range ``[-1, 1]``.
            sample_rate: Sample rate of ``audio`` in Hz.

        Returns:
            A ``TranscriptionResult`` containing the recognized text.
        """

    @abstractmethod
    def transcribe_streaming(
        self,
        audio_chunk: np.ndarray,
        sample_rate: int,
    ) -> TranscriptionResult:
        """Process a streaming audio chunk and return an incremental result.

        The default implementation returns a final transcription for the chunk.
        Streaming partial transcripts will be implemented in a future version.

        Args:
            audio_chunk: One-dimensional ``float32`` audio samples.
            sample_rate: Sample rate of ``audio_chunk`` in Hz.

        Returns:
            A ``TranscriptionResult`` with incremental or final text.
        """
