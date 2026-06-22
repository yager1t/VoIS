"""Integration tests for the audio/VAD/ASR pipeline."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import sounddevice as sd

from src.app import trim_silence
from src.asr.whisper_provider import FasterWhisperProvider
from src.audio.buffer import AudioBuffer
from src.audio.capture import AudioCapture
from src.audio.vad import WebRTCVADProvider
from src.config import Settings


@pytest.mark.integration
def test_audio_buffer_to_vad(synthetic_mixed_audio: np.ndarray, sample_rate: int) -> None:
    """Audio should survive buffering, then VAD should trim surrounding silence."""
    buffer = AudioBuffer(max_seconds=10.0, sample_rate=sample_rate)
    buffer.append(synthetic_mixed_audio)

    vad = WebRTCVADProvider(aggressiveness=1)
    trimmed = trim_silence(
        buffer.get(),
        sample_rate,
        vad,
        trim_seconds=0.3,
        frame_ms=30,
    )

    assert trimmed.size > 0
    assert trimmed.size <= synthetic_mixed_audio.size


@pytest.mark.integration
def test_capture_callback_fills_buffer(sample_rate: int) -> None:
    """Simulated sounddevice callbacks should accumulate in an AudioBuffer."""
    buffer = AudioBuffer(max_seconds=10.0, sample_rate=sample_rate)
    capture = AudioCapture(
        sample_rate=sample_rate,
        channels=1,
        push_to_talk=True,
        block_size=512,
    )
    capture.set_callback(buffer.append)

    chunk = np.sin(np.linspace(0.0, 2.0 * np.pi, 512)).astype(np.float32) * 0.1
    mock_stream = MagicMock()

    with patch("src.audio.capture.sd.InputStream", return_value=mock_stream):
        capture.start()
        # Simulate two callback invocations while recording.
        capture._on_audio(chunk, chunk.shape[0], None, sd.CallbackFlags())
        capture._on_audio(chunk, chunk.shape[0], None, sd.CallbackFlags())
        capture.stop()

    assert buffer.get().size == chunk.size * 2


@pytest.mark.integration
def test_asr_pipeline_with_mock_model(
    settings: Settings,
    synthetic_speech: np.ndarray,
    sample_rate: int,
    mock_whisper_model: MagicMock,
) -> None:
    """FasterWhisperProvider should transcribe synthetic audio via patched wrappers."""
    provider = FasterWhisperProvider(settings)

    with (
        patch("src.asr.model_manager._download_model") as mock_download,
        patch(
            "src.asr.model_manager._create_whisper_model",
            return_value=mock_whisper_model,
        ) as mock_create,
    ):
        result = provider.transcribe(synthetic_speech, sample_rate)

    assert result.text == "Hello, world"
    assert result.language == "en"
    assert result.is_final is True
    expected_model_path = settings.models_dir / settings.asr_model
    mock_download.assert_called_once_with(
        settings.asr_model,
        output_dir=str(expected_model_path),
    )
    mock_create.assert_called_once()
