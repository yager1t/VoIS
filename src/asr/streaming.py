"""Streaming transcriber built on a background thread."""

from __future__ import annotations

import queue
import threading
from typing import TYPE_CHECKING

import numpy as np

from src.asr.base import TranscriptionResult
from src.audio.streaming_buffer import StreamingAudioBuffer

if TYPE_CHECKING:  # pragma: no cover
    from src.asr.base import ASRProvider
    from src.audio.vad import VADProvider
    from src.config import Settings


class StreamingTranscriber:
    """Background transcriber that consumes audio chunks and emits results.

    The transcriber accumulates audio in a thread-safe ``StreamingAudioBuffer``.
    Every ``loop_interval_seconds`` it checks whether enough unprocessed audio is
    available to run a streaming chunk transcription. When speech is detected by
    the VAD provider, the chunk is sent to the ASR provider and a result with
    ``is_final=False`` is queued. A long enough silence pause marks the most
    recent partial result as final. On ``stop()`` any remaining audio is flushed
    and transcribed as a final result.
    """

    def __init__(
        self,
        settings: Settings,
        asr_provider: ASRProvider,
        vad_provider: VADProvider,
    ) -> None:
        """Initialize the streaming transcriber.

        Args:
            settings: Parsed application settings.
            asr_provider: Provider used to transcribe audio chunks.
            vad_provider: Provider used to detect speech activity.
        """
        self.settings = settings
        self.asr_provider = asr_provider
        self.vad_provider = vad_provider

        self.sample_rate = settings.audio_sample_rate
        self.streaming_chunk_seconds = getattr(settings, "streaming_chunk_seconds", 1.0)
        self.silence_pause_seconds = getattr(settings, "streaming_silence_pause_seconds", 0.5)
        self.loop_interval_seconds = 0.1

        self._buffer = StreamingAudioBuffer(self.sample_rate)
        self._results: queue.Queue[TranscriptionResult] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        self._lock = threading.Lock()
        self._last_speech_committed_samples = 0
        self._pending_finalization = False

    def start(self) -> None:
        """Start the background transcription thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Signal the background thread to stop and wait for it to finish.

        Any remaining unprocessed audio is transcribed and queued as a final
        result before the thread exits.
        """
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=5.0)
        self._flush()

    def add_audio(self, chunk: np.ndarray) -> None:
        """Add an audio chunk to the internal streaming buffer.

        Args:
            chunk: One-dimensional ``float32`` audio samples.
        """
        self._buffer.append(chunk)

    def get_results(self) -> list[TranscriptionResult]:
        """Return all collected transcription results.

        Returns:
            A list of results in the order they were produced.
        """
        results: list[TranscriptionResult] = []
        while not self._results.empty():
            try:
                results.append(self._results.get_nowait())
            except queue.Empty:
                break
        return results

    def _run(self) -> None:
        """Background loop that processes audio chunks."""
        while not self._stop_event.is_set():
            self._process_once()
            self._check_silence_pause()
            self._stop_event.wait(timeout=self.loop_interval_seconds)

    def _process_once(self) -> None:
        """Transcribe one chunk of unprocessed audio if enough is available."""
        if self._buffer.unprocessed_duration() < self.streaming_chunk_seconds:
            return

        audio = self._buffer.get_unprocessed_audio()
        if audio.size == 0:
            return

        chunk_samples = audio.shape[0]
        is_speech = self.vad_provider.is_speech(audio, self.sample_rate)

        self._buffer.commit(chunk_samples)

        if is_speech:
            result = self.asr_provider.transcribe(audio, self.sample_rate)
            result.is_final = False
            self._results.put(result)
            with self._lock:
                self._last_speech_committed_samples = self._buffer.processed_samples()
                self._pending_finalization = True

    def _check_silence_pause(self) -> None:
        """Mark the last partial result final if enough silence has elapsed."""
        total_samples = self._buffer.get_all().shape[0]
        with self._lock:
            silence_samples = total_samples - self._last_speech_committed_samples
            silence_seconds = silence_samples / self.sample_rate if self.sample_rate > 0 else 0.0
            pending = self._pending_finalization

        if pending and silence_seconds >= self.silence_pause_seconds:
            self._finalize_last_result()

    def _finalize_last_result(self) -> None:
        """Convert the most recent partial result to a final result."""
        with self._lock:
            self._pending_finalization = False

        existing = self.get_results()
        if not existing:
            return

        last = existing[-1]
        if not last.is_final:
            existing[-1] = TranscriptionResult(
                text=last.text,
                is_final=True,
                confidence=last.confidence,
                language=last.language,
            )
        for result in existing:
            self._results.put(result)

    def _flush(self) -> None:
        """Transcribe any remaining audio and queue it as a final result."""
        audio = self._buffer.get_unprocessed_audio()
        self._buffer.commit(audio.shape[0])

        if audio.size == 0:
            self._finalize_last_result()
            return

        result = self.asr_provider.transcribe(audio, self.sample_rate)
        result.is_final = True
        self._results.put(result)
