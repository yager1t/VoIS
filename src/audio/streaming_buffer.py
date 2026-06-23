"""Streaming audio buffer with processed/unprocessed tracking."""

from __future__ import annotations

import threading

import numpy as np


class StreamingAudioBuffer:
    """Thread-safe audio buffer that tracks processed and unprocessed samples.

    Audio is stored as one-dimensional ``np.float32`` samples. Callers append
    chunks, read unprocessed audio for incremental transcription, then ``commit``
    the samples they have consumed.
    """

    def __init__(self, sample_rate: int) -> None:
        """Initialize an empty streaming buffer.

        Args:
            sample_rate: Sample rate of the audio in Hz.
        """
        self._sample_rate = sample_rate
        self._lock = threading.Lock()
        self._buffer = np.array([], dtype=np.float32)
        self._committed_samples = 0

    def append(self, chunk: np.ndarray) -> None:
        """Append float32 audio to the buffer.

        Args:
            chunk: One-dimensional ``float32`` audio samples.
        """
        if chunk.size == 0:
            return
        flat = np.asarray(chunk, dtype=np.float32).reshape(-1)
        with self._lock:
            self._buffer = np.concatenate([self._buffer, flat], dtype=np.float32)

    def get_unprocessed_audio(self) -> np.ndarray:
        """Return all samples that have not yet been committed.

        Returns:
            A copy of the unprocessed ``float32`` samples.
        """
        with self._lock:
            if self._committed_samples >= self._buffer.shape[0]:
                return np.array([], dtype=np.float32)
            return self._buffer[self._committed_samples :].copy()

    def commit(self, n_samples: int) -> None:
        """Mark ``n_samples`` additional samples as processed.

        Args:
            n_samples: Number of unprocessed samples to commit.

        Raises:
            ValueError: If ``n_samples`` is negative or exceeds unprocessed data.
        """
        if n_samples < 0:
            raise ValueError("n_samples must be non-negative")
        with self._lock:
            unprocessed = self._buffer.shape[0] - self._committed_samples
            if n_samples > unprocessed:
                raise ValueError(
                    f"Cannot commit {n_samples} samples; only {unprocessed} unprocessed"
                )
            self._committed_samples += n_samples

    def processed_samples(self) -> int:
        """Return the number of samples that have been committed."""
        with self._lock:
            return int(self._committed_samples)

    def get_all(self) -> np.ndarray:
        """Return a copy of the entire buffer.

        Returns:
            Concatenated ``float32`` samples. Empty buffers return a zero-length array.
        """
        with self._lock:
            return self._buffer.copy()

    def clear(self) -> None:
        """Reset the buffer and processed marker."""
        with self._lock:
            self._buffer = np.array([], dtype=np.float32)
            self._committed_samples = 0

    def duration(self) -> float:
        """Return the total buffered duration in seconds."""
        if self._sample_rate <= 0:
            return 0.0
        with self._lock:
            return float(self._buffer.shape[0] / self._sample_rate)

    def unprocessed_duration(self) -> float:
        """Return the duration of unprocessed samples in seconds."""
        if self._sample_rate <= 0:
            return 0.0
        with self._lock:
            unprocessed = self._buffer.shape[0] - self._committed_samples
            return float(unprocessed / self._sample_rate)
