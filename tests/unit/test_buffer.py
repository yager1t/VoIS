"""Unit tests for the AudioBuffer class."""

import numpy as np
import pytest

from src.audio.buffer import AudioBuffer

SAMPLE_RATE = 16000


def test_empty_buffer() -> None:
    """A fresh buffer should return empty arrays and zero duration."""
    buffer = AudioBuffer()
    assert buffer.get().size == 0
    assert buffer.duration(SAMPLE_RATE) == 0.0


def test_append_and_get() -> None:
    """Appended chunks should concatenate in order."""
    buffer = AudioBuffer()
    chunk_a = np.ones(100, dtype=np.float32)
    chunk_b = np.full(50, -1.0, dtype=np.float32)

    buffer.append(chunk_a)
    buffer.append(chunk_b)

    result = buffer.get()
    assert result.shape == (150,)
    assert np.array_equal(result[:100], chunk_a)
    assert np.array_equal(result[100:], chunk_b)


def test_append_reshapes_multidimensional() -> None:
    """Multi-dimensional input should be flattened to a mono stream."""
    buffer = AudioBuffer()
    buffer.append(np.ones((20, 2), dtype=np.float32))

    assert buffer.get().shape == (40,)


def test_clear() -> None:
    """Clear should remove all buffered audio."""
    buffer = AudioBuffer()
    buffer.append(np.ones(100, dtype=np.float32))
    buffer.clear()

    assert buffer.get().size == 0
    assert buffer.duration(SAMPLE_RATE) == 0.0


def test_duration() -> None:
    """Duration should equal total samples divided by sample rate."""
    buffer = AudioBuffer()
    buffer.append(np.ones(SAMPLE_RATE, dtype=np.float32))
    buffer.append(np.ones(SAMPLE_RATE // 2, dtype=np.float32))

    assert buffer.duration(SAMPLE_RATE) == pytest.approx(1.5)


def test_get_recent_seconds() -> None:
    """get_recent_seconds should return the trailing window."""
    buffer = AudioBuffer()
    total = SAMPLE_RATE * 2  # 2 seconds
    audio = np.arange(total, dtype=np.float32)
    buffer.append(audio)

    recent = buffer.get_recent_seconds(0.5, SAMPLE_RATE)
    assert recent.shape == (SAMPLE_RATE // 2,)
    assert np.array_equal(recent, audio[-SAMPLE_RATE // 2 :])


def test_get_recent_seconds_not_exceed_buffer() -> None:
    """When the requested window exceeds the buffer, return everything."""
    buffer = AudioBuffer()
    chunk = np.ones(SAMPLE_RATE, dtype=np.float32)
    buffer.append(chunk)

    recent = buffer.get_recent_seconds(5.0, SAMPLE_RATE)
    assert recent.shape == (SAMPLE_RATE,)


def test_get_recent_seconds_invalid_arguments() -> None:
    """Non-positive seconds or sample rate should yield an empty array."""
    buffer = AudioBuffer()
    buffer.append(np.ones(100, dtype=np.float32))

    assert buffer.get_recent_seconds(-1.0, SAMPLE_RATE).size == 0
    assert buffer.get_recent_seconds(1.0, 0).size == 0


def test_max_seconds_discards_oldest_samples() -> None:
    """A bounded buffer should keep only the most recent samples."""
    buffer = AudioBuffer(max_seconds=1.0, sample_rate=10)

    buffer.append(np.arange(8, dtype=np.float32))
    buffer.append(np.arange(8, 15, dtype=np.float32))

    result = buffer.get()
    assert result.shape == (10,)
    assert np.array_equal(result, np.arange(5, 15, dtype=np.float32))
