"""Unit tests for the StreamingTranscriber class."""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.asr.base import TranscriptionResult
from src.asr.streaming import StreamingTranscriber
from src.config import Settings

SAMPLE_RATE = 16000
CHUNK_SAMPLES = int(SAMPLE_RATE * 1.0)


@pytest.fixture
def settings(tmp_path) -> Settings:
    """Return minimal settings for the streaming transcriber."""
    return Settings(
        data_dir=tmp_path / "data",
        models_dir=tmp_path / "models",
        audio_sample_rate=SAMPLE_RATE,
    )


@pytest.fixture
def mock_asr() -> MagicMock:
    """Return a mock ASR provider that returns deterministic text."""
    provider = MagicMock()
    provider.transcribe_streaming.return_value = TranscriptionResult(
        text="hello",
        is_final=True,
        confidence=0.9,
        language="en",
    )
    return provider


@pytest.fixture
def mock_vad() -> MagicMock:
    """Return a mock VAD provider that detects speech on every frame."""
    provider = MagicMock()
    provider.is_speech.return_value = True
    return provider


@pytest.fixture
def transcriber(
    settings: Settings,
    mock_asr: MagicMock,
    mock_vad: MagicMock,
) -> StreamingTranscriber:
    """Return a configured streaming transcriber with mocked providers."""
    return StreamingTranscriber(settings, mock_asr, mock_vad)


def test_audio_chunks_accumulate_and_transcribe(
    transcriber: StreamingTranscriber,
    mock_asr: MagicMock,
) -> None:
    """Chunks smaller than the chunk size should accumulate and then transcribe."""
    transcriber.streaming_chunk_seconds = 1.0
    transcriber.loop_interval_seconds = 0.01
    transcriber.start()

    try:
        for _ in range(5):
            transcriber.add_audio(np.ones(CHUNK_SAMPLES // 5, dtype=np.float32))

        time.sleep(0.15)
    finally:
        transcriber.stop()

    results = transcriber.get_results()
    assert len(results) >= 1
    assert results[-1].text == "hello"
    mock_asr.transcribe_streaming.assert_called()


def test_results_queued_with_is_final_false(
    transcriber: StreamingTranscriber,
) -> None:
    """A normal chunk transcription should queue a non-final result."""
    transcriber.streaming_chunk_seconds = 1.0
    transcriber.loop_interval_seconds = 0.01
    transcriber.start()

    try:
        transcriber.add_audio(np.ones(CHUNK_SAMPLES, dtype=np.float32))
        time.sleep(0.15)
        results = transcriber.get_results()
    finally:
        transcriber.stop()

    assert any(not result.is_final for result in results)


def test_stop_flushes_remaining_audio(
    transcriber: StreamingTranscriber,
    mock_asr: MagicMock,
) -> None:
    """Stopping should transcribe any unprocessed audio as a final result."""
    transcriber.streaming_chunk_seconds = 5.0
    transcriber.loop_interval_seconds = 0.01
    transcriber.start()

    try:
        transcriber.add_audio(np.ones(CHUNK_SAMPLES // 2, dtype=np.float32))
        time.sleep(0.05)
    finally:
        transcriber.stop()

    results = transcriber.get_results()
    assert len(results) >= 1
    assert results[-1].is_final is True
    mock_asr.transcribe_streaming.assert_called()


def test_silence_pause_marks_result_final(
    transcriber: StreamingTranscriber,
    mock_vad: MagicMock,
) -> None:
    """A long silence after speech should mark the last partial result final."""
    mock_vad.is_speech.side_effect = [True, False, False, False, False]
    transcriber.streaming_chunk_seconds = 0.5
    transcriber.silence_pause_seconds = 0.3
    transcriber.loop_interval_seconds = 0.05
    transcriber.start()

    try:
        transcriber.add_audio(np.ones(CHUNK_SAMPLES // 2, dtype=np.float32))
        time.sleep(0.2)
        transcriber.add_audio(np.zeros(CHUNK_SAMPLES, dtype=np.float32))
        time.sleep(0.5)
    finally:
        transcriber.stop()

    results = transcriber.get_results()
    assert any(result.is_final for result in results)


def test_no_speech_does_not_queue_result(
    transcriber: StreamingTranscriber,
    mock_vad: MagicMock,
    mock_asr: MagicMock,
) -> None:
    """When VAD reports no speech, ASR should not be called."""
    mock_vad.is_speech.return_value = False
    transcriber.streaming_chunk_seconds = 0.5
    transcriber.loop_interval_seconds = 0.05
    transcriber.start()

    try:
        transcriber.add_audio(np.ones(CHUNK_SAMPLES, dtype=np.float32))
        time.sleep(0.15)
    finally:
        transcriber.stop()

    results = transcriber.get_results()
    assert len(results) == 0
    mock_asr.transcribe_streaming.assert_not_called()


def test_add_audio_before_start(
    transcriber: StreamingTranscriber,
) -> None:
    """Audio can be added before the background thread starts."""
    transcriber.streaming_chunk_seconds = 1.0
    transcriber.loop_interval_seconds = 0.05
    transcriber.add_audio(np.ones(CHUNK_SAMPLES, dtype=np.float32))
    transcriber.start()

    try:
        time.sleep(0.15)
    finally:
        transcriber.stop()

    results = transcriber.get_results()
    assert len(results) >= 1


def test_stop_is_idempotent(transcriber: StreamingTranscriber) -> None:
    """Calling stop multiple times should not raise or deadlock."""
    transcriber.start()
    transcriber.stop()
    transcriber.stop()

    assert transcriber.get_results() == []


def test_streaming_uses_streaming_beam_size(
    settings: Settings,
    mock_asr: MagicMock,
    mock_vad: MagicMock,
) -> None:
    """Chunk transcription should use the configured streaming beam size."""
    settings.asr_streaming_beam_size = 3
    transcriber = StreamingTranscriber(settings, mock_asr, mock_vad)
    transcriber.streaming_chunk_seconds = 1.0
    transcriber.loop_interval_seconds = 0.01
    transcriber.start()

    try:
        transcriber.add_audio(np.ones(CHUNK_SAMPLES, dtype=np.float32))
        time.sleep(0.15)
    finally:
        transcriber.stop()

    assert mock_asr.transcribe_streaming.call_count >= 1
    audio, sample_rate = mock_asr.transcribe_streaming.call_args_list[0].args
    assert audio.shape == (CHUNK_SAMPLES,)
    assert sample_rate == SAMPLE_RATE
    assert mock_asr.transcribe_streaming.call_args_list[0].kwargs["beam_size"] == 3


def test_stop_flush_uses_streaming_beam_size(
    settings: Settings,
    mock_asr: MagicMock,
    mock_vad: MagicMock,
) -> None:
    """Flush transcription should use the configured streaming beam size."""
    settings.asr_streaming_beam_size = 4
    transcriber = StreamingTranscriber(settings, mock_asr, mock_vad)
    transcriber.streaming_chunk_seconds = 5.0

    transcriber.add_audio(np.ones(CHUNK_SAMPLES // 2, dtype=np.float32))
    transcriber.stop()

    mock_asr.transcribe_streaming.assert_called_once()
    audio, sample_rate = mock_asr.transcribe_streaming.call_args.args
    assert audio.shape == (CHUNK_SAMPLES // 2,)
    assert sample_rate == SAMPLE_RATE
    assert mock_asr.transcribe_streaming.call_args.kwargs["beam_size"] == 4
