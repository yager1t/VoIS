"""Unit tests for the StreamingAudioBuffer class."""

from __future__ import annotations

import threading

import numpy as np
import pytest

from src.audio.streaming_buffer import StreamingAudioBuffer

SAMPLE_RATE = 16000


def test_empty_buffer_returns_empty_arrays() -> None:
    """A fresh buffer should return empty arrays and zero durations."""
    buffer = StreamingAudioBuffer(SAMPLE_RATE)

    assert buffer.get_unprocessed_audio().size == 0
    assert buffer.get_all().size == 0
    assert buffer.duration() == 0.0
    assert buffer.unprocessed_duration() == 0.0


def test_append_and_get_all() -> None:
    """Appended chunks should concatenate in order for get_all."""
    buffer = StreamingAudioBuffer(SAMPLE_RATE)
    chunk_a = np.ones(100, dtype=np.float32)
    chunk_b = np.full(50, -1.0, dtype=np.float32)

    buffer.append(chunk_a)
    buffer.append(chunk_b)

    result = buffer.get_all()
    assert result.shape == (150,)
    assert np.array_equal(result[:100], chunk_a)
    assert np.array_equal(result[100:], chunk_b)


def test_get_unprocessed_audio_after_commit() -> None:
    """Committing samples reduces the unprocessed audio returned."""
    buffer = StreamingAudioBuffer(SAMPLE_RATE)
    chunk = np.arange(1000, dtype=np.float32)
    buffer.append(chunk)

    assert buffer.get_unprocessed_audio().shape == (1000,)
    buffer.commit(400)
    assert buffer.get_unprocessed_audio().shape == (600,)
    assert np.array_equal(buffer.get_unprocessed_audio(), chunk[400:])


def test_commit_all_makes_unprocessed_empty() -> None:
    """Committing all samples leaves no unprocessed audio."""
    buffer = StreamingAudioBuffer(SAMPLE_RATE)
    buffer.append(np.ones(500, dtype=np.float32))
    buffer.commit(500)

    assert buffer.get_unprocessed_audio().size == 0
    assert buffer.get_all().shape == (500,)


def test_commit_negative_or_excess_raises() -> None:
    """Commit must reject negative values and commits beyond unprocessed data."""
    buffer = StreamingAudioBuffer(SAMPLE_RATE)
    buffer.append(np.ones(100, dtype=np.float32))

    with pytest.raises(ValueError):
        buffer.commit(-1)

    with pytest.raises(ValueError):
        buffer.commit(200)


def test_clear_resets_buffer() -> None:
    """Clear should remove all audio and reset the commit marker."""
    buffer = StreamingAudioBuffer(SAMPLE_RATE)
    buffer.append(np.ones(1000, dtype=np.float32))
    buffer.commit(500)

    buffer.clear()

    assert buffer.get_all().size == 0
    assert buffer.get_unprocessed_audio().size == 0
    assert buffer.duration() == 0.0
    assert buffer.unprocessed_duration() == 0.0


def test_duration_and_unprocessed_duration() -> None:
    """Durations should reflect total and unprocessed sample counts."""
    buffer = StreamingAudioBuffer(SAMPLE_RATE)
    buffer.append(np.ones(SAMPLE_RATE, dtype=np.float32))
    buffer.append(np.ones(SAMPLE_RATE // 2, dtype=np.float32))

    assert buffer.duration() == pytest.approx(1.5)
    assert buffer.unprocessed_duration() == pytest.approx(1.5)

    buffer.commit(SAMPLE_RATE)
    assert buffer.duration() == pytest.approx(1.5)
    assert buffer.unprocessed_duration() == pytest.approx(0.5)


def test_append_ignores_empty_chunk() -> None:
    """Empty chunks should not affect the buffer."""
    buffer = StreamingAudioBuffer(SAMPLE_RATE)
    buffer.append(np.array([], dtype=np.float32))

    assert buffer.get_all().size == 0


def test_thread_safety_concurrent_appends_and_commits() -> None:
    """Concurrent appends and commits should not corrupt the buffer."""
    buffer = StreamingAudioBuffer(SAMPLE_RATE)
    errors: list[Exception] = []

    def append_worker() -> None:
        try:
            for _ in range(1000):
                buffer.append(np.ones(16, dtype=np.float32))
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    def commit_worker() -> None:
        try:
            for _ in range(1000):
                if buffer.unprocessed_duration() > 0:
                    buffer.commit(8)
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    threads = [
        threading.Thread(target=append_worker),
        threading.Thread(target=append_worker),
        threading.Thread(target=commit_worker),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors
    total = buffer.get_all().shape[0]
    unprocessed = buffer.get_unprocessed_audio().shape[0]
    assert total == 32000
    assert 0 <= unprocessed <= total


def test_invalid_sample_rate_returns_zero_duration() -> None:
    """A non-positive sample rate should yield zero durations."""
    buffer = StreamingAudioBuffer(0)
    buffer.append(np.ones(1000, dtype=np.float32))

    assert buffer.duration() == 0.0
    assert buffer.unprocessed_duration() == 0.0
