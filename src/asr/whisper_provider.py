"""faster-whisper based ASR provider implementation."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import soundfile as sf
from loguru import logger

from src.asr.base import ASRProvider, TranscriptionResult
from src.asr.model_manager import ModelManager
from src.config import Settings

if TYPE_CHECKING:  # pragma: no cover
    from faster_whisper import WhisperModel


class FasterWhisperProvider(ASRProvider):
    """ASR provider using ``faster-whisper.WhisperModel``.

    The model is loaded lazily on first transcription unless :meth:`load_model`
    is called explicitly.
    """

    def __init__(self, settings: Settings) -> None:
        """Initialize the provider with settings.

        Args:
            settings: Parsed application configuration.
        """
        super().__init__(settings)
        self._model: WhisperModel | None = None
        self._model_manager = ModelManager(settings)
        self._model_name = settings.asr_model

    def load_model(self) -> None:
        """Load the faster-whisper model into memory."""
        if self._model is not None:
            return

        logger.info(
            "Loading faster-whisper model '{}' on {} ({})",
            self._model_name,
            self.settings.asr_device,
            self.settings.asr_compute_type,
        )
        self._model = self._model_manager.load_whisper_model(
            self._model_name,
            device=self.settings.asr_device,
            compute_type=self.settings.asr_compute_type,
        )

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> TranscriptionResult:
        """Transcribe a complete audio clip into text.

        The audio is written to a temporary WAV file and passed to the model.

        Args:
            audio: One-dimensional ``float32`` audio samples in the range ``[-1, 1]``.
            sample_rate: Sample rate of ``audio`` in Hz.

        Returns:
            A ``TranscriptionResult`` with the recognized text.
        """
        self.load_model()
        if self._model is None:  # pragma: no cover - defensive
            raise RuntimeError("Failed to load faster-whisper model")

        samples = np.asarray(audio, dtype=np.float32).reshape(-1)
        if samples.size == 0:
            return TranscriptionResult(text="")

        language = (
            self.settings.asr_language
            if self.settings.asr_language.lower() != "auto"
            else None
        )

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            sf.write(tmp_path, samples, sample_rate)
            segments, info = self._model.transcribe(
                str(tmp_path),
                language=language,
                beam_size=self.settings.asr_beam_size,
                condition_on_previous_text=True,
            )

            text_parts: list[str] = []
            confidences: list[float] = []
            for segment in segments:
                text_parts.append(segment.text)
                if segment.avg_logprob is not None:
                    confidences.append(segment.avg_logprob)

            text = " ".join(part.strip() for part in text_parts).strip()
            detected_language = info.language if info.language else self.settings.asr_language
            confidence = float(np.mean(confidences)) if confidences else None

            return TranscriptionResult(
                text=text,
                is_final=True,
                confidence=confidence,
                language=detected_language,
            )
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError as exc:  # pragma: no cover - defensive
                logger.warning("Could not remove temporary WAV file: {}", exc)

    def transcribe_streaming(
        self,
        audio_chunk: np.ndarray,
        sample_rate: int,
    ) -> TranscriptionResult:
        """Return a final transcription for a streaming chunk.

        Streaming partial transcripts will be implemented in a future version;
        for now this delegates to :meth:`transcribe`.

        Args:
            audio_chunk: One-dimensional ``float32`` audio samples.
            sample_rate: Sample rate of ``audio_chunk`` in Hz.

        Returns:
            A ``TranscriptionResult`` with final text for the chunk.
        """
        result = self.transcribe(audio_chunk, sample_rate)
        return TranscriptionResult(
            text=result.text,
            is_final=True,
            confidence=result.confidence,
            language=result.language,
        )
