"""Background final transcription helper."""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

if TYPE_CHECKING:  # pragma: no cover
    from src.asr.base import ASRProvider


class FinalTranscriber:
    """Run a full ASR transcription in a background thread.

    The helper accepts prepared audio, calls ``ASRProvider.transcribe`` on a
    daemon thread, and delivers the recognized text through a callback. It is
    stoppable in the sense that ``stop()`` prevents the callback from being
    invoked when transcription has not yet started, and waits for the worker to
    finish with a timeout.
    """

    def __init__(
        self,
        asr_provider: ASRProvider,
        audio: np.ndarray,
        sample_rate: int,
        on_result: Callable[[str], None],
    ) -> None:
        """Initialize the final transcriber.

        Args:
            asr_provider: Provider used to transcribe the complete audio clip.
            audio: One-dimensional ``float32`` audio samples.
            sample_rate: Sample rate of ``audio`` in Hz.
            on_result: Callback invoked with the recognized text on success.
        """
        self.asr_provider = asr_provider
        self.audio = audio
        self.sample_rate = sample_rate
        self.on_result = on_result

        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._started = False

    def start(self) -> None:
        """Start the background transcription thread.

        Has no effect if a worker thread is already running.
        """
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._started = True
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def stop(self) -> None:
        """Signal cancellation and wait for the worker to finish."""
        self._stop_event.set()
        with self._lock:
            thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=5.0)

    def is_running(self) -> bool:
        """Return whether the worker thread is currently active."""
        with self._lock:
            return self._thread is not None and self._thread.is_alive()

    def _run(self) -> None:
        """Transcribe the audio and deliver the result."""
        if self._stop_event.is_set():
            return

        try:
            result = self.asr_provider.transcribe(self.audio, self.sample_rate)
        except Exception:
            logger.exception("Final transcription failed")
            return

        if self._stop_event.is_set():
            return

        try:
            self.on_result(result.text)
        except Exception:
            logger.exception("Final transcription callback failed")
