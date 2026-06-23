"""Unit tests for the FinalTranscriber class."""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.asr.base import TranscriptionResult
from src.asr.final_transcriber import FinalTranscriber

SAMPLE_RATE = 16000


@pytest.fixture
def mock_asr() -> MagicMock:
    """Return a mock ASR provider that returns deterministic text."""
    provider = MagicMock()
    provider.transcribe.return_value = TranscriptionResult(
        text="final text",
        is_final=True,
        confidence=0.95,
        language="en",
    )
    return provider


@pytest.fixture
def audio() -> np.ndarray:
    """Return a short float32 audio array."""
    return np.ones(SAMPLE_RATE, dtype=np.float32)


def test_final_transcriber_invokes_callback_with_text(
    mock_asr: MagicMock,
    audio: np.ndarray,
) -> None:
    """The callback should receive the transcribed text."""
    callback = MagicMock()
    transcriber = FinalTranscriber(mock_asr, audio, SAMPLE_RATE, callback)

    transcriber.start()
    time.sleep(0.05)
    transcriber.stop()

    callback.assert_called_once_with("final text")
    mock_asr.transcribe.assert_called_once_with(audio, SAMPLE_RATE)


def test_final_transcriber_skips_callback_for_empty_text(
    mock_asr: MagicMock,
    audio: np.ndarray,
) -> None:
    """An empty transcription result should still invoke the callback with ''."""
    mock_asr.transcribe.return_value = TranscriptionResult(text="")
    callback = MagicMock()
    transcriber = FinalTranscriber(mock_asr, audio, SAMPLE_RATE, callback)

    transcriber.start()
    time.sleep(0.05)
    transcriber.stop()

    callback.assert_called_once_with("")


def test_final_transcriber_stop_prevents_callback(
    mock_asr: MagicMock,
    audio: np.ndarray,
) -> None:
    """Stopping before transcription starts should prevent the callback."""
    callback = MagicMock()
    transcriber = FinalTranscriber(mock_asr, audio, SAMPLE_RATE, callback)

    transcriber.stop()
    time.sleep(0.05)

    callback.assert_not_called()
    mock_asr.transcribe.assert_not_called()


def test_final_transcriber_stop_is_idempotent(
    mock_asr: MagicMock,
    audio: np.ndarray,
) -> None:
    """Calling stop multiple times should not raise or deadlock."""
    transcriber = FinalTranscriber(mock_asr, audio, SAMPLE_RATE, MagicMock())
    transcriber.start()
    transcriber.stop()
    transcriber.stop()

    assert not transcriber.is_running()


def test_final_transcriber_start_is_idempotent(audio: np.ndarray) -> None:
    """Starting an already running transcriber should not spawn a second thread."""

    def _slow_transcribe(*_args: object, **_kwargs: object) -> TranscriptionResult:
        time.sleep(0.2)
        return TranscriptionResult(text="final text")

    slow_asr = MagicMock()
    slow_asr.transcribe.side_effect = _slow_transcribe

    callback = MagicMock()
    transcriber = FinalTranscriber(slow_asr, audio, SAMPLE_RATE, callback)

    transcriber.start()
    transcriber.start()  # Second start while the first thread is still running
    transcriber.stop()

    slow_asr.transcribe.assert_called_once_with(audio, SAMPLE_RATE)


def test_final_transcriber_logs_transcription_error(
    mock_asr: MagicMock,
    audio: np.ndarray,
) -> None:
    """ASR failures should be logged and not propagate to the caller."""
    mock_asr.transcribe.side_effect = RuntimeError("asr failed")
    callback = MagicMock()
    transcriber = FinalTranscriber(mock_asr, audio, SAMPLE_RATE, callback)

    transcriber.start()
    time.sleep(0.05)
    transcriber.stop()

    callback.assert_not_called()
