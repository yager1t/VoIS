"""Thread-safe audio buffer for streaming capture."""

from __future__ import annotations

import threading
from collections import deque

import numpy as np


class AudioBuffer:
    """Thread-safe accumulator for audio chunks.

    Stores one-dimensional mono audio frames as ``np.float32`` samples and provides
    helpers for retrieving the full stream or a recent trailing window.
    """

    def __init__(self, max_seconds: float | None = None, sample_rate: int | None = None) -> None:
        """Initialize an empty buffer.

        Args:
            max_seconds: Optional maximum duration to retain, in seconds.
            sample_rate: Sample rate used to convert ``max_seconds`` into a sample
                limit. Required for ``max_seconds`` to take effect.
        """
        self._lock = threading.Lock()
        self._chunks: deque[np.ndarray] = deque()
        self._max_samples = (
            int(max_seconds * sample_rate)
            if max_seconds is not None and sample_rate is not None and sample_rate > 0
            else None
        )

    def append(self, chunk: np.ndarray) -> None:
        """Append a new audio chunk to the buffer.

        Args:
            chunk: One-dimensional ``float32`` audio samples.
        """
        if chunk.size == 0:
            return
        flat = np.asarray(chunk, dtype=np.float32).reshape(-1)
        with self._lock:
            self._chunks.append(flat)
            self._trim_to_limit()

    def get(self) -> np.ndarray:
        """Return a concatenated copy of all buffered audio.

        Returns:
            Concatenated ``float32`` samples. Empty buffers return a zero-length array.
        """
        with self._lock:
            if not self._chunks:
                return np.array([], dtype=np.float32)
            return np.concatenate(list(self._chunks), dtype=np.float32)

    def get_recent_seconds(self, seconds: float, sample_rate: int) -> np.ndarray:
        """Return the last ``seconds`` of buffered audio.

        Args:
            seconds: Duration to retain from the end of the buffer.
            sample_rate: Sample rate in Hz.

        Returns:
            Trailing ``float32`` samples up to ``seconds * sample_rate`` frames.
        """
        if seconds <= 0 or sample_rate <= 0:
            return np.array([], dtype=np.float32)

        max_samples = int(seconds * sample_rate)
        with self._lock:
            if not self._chunks:
                return np.array([], dtype=np.float32)

            combined = np.concatenate(list(self._chunks), dtype=np.float32)
        return combined[-max_samples:] if combined.size > max_samples else combined

    def clear(self) -> None:
        """Remove all audio from the buffer."""
        with self._lock:
            self._chunks.clear()

    def _trim_to_limit(self) -> None:
        """Drop oldest samples when the configured buffer limit is exceeded."""
        if self._max_samples is None:
            return

        total = sum(int(chunk.shape[0]) for chunk in self._chunks)
        while self._chunks and total > self._max_samples:
            excess = total - self._max_samples
            first = self._chunks[0]
            if first.shape[0] <= excess:
                total -= int(first.shape[0])
                self._chunks.popleft()
            else:
                self._chunks[0] = first[excess:]
                total = self._max_samples

    def duration(self, sample_rate: int) -> float:
        """Return the buffered duration in seconds.

        Args:
            sample_rate: Sample rate in Hz.

        Returns:
            Buffered duration in seconds.
        """
        if sample_rate <= 0:
            return 0.0
        with self._lock:
            total = sum(int(chunk.shape[0]) for chunk in self._chunks)
        return float(total / sample_rate)
