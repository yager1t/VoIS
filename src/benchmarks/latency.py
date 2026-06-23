"""Latency benchmark harness for streaming vs batch ASR.

The harness measures wall-clock latency of the transcription pipeline in two
modes:

* ``batch`` -- transcribe the entire prepared audio clip with a single call to
  the ASR provider.
* ``streaming`` -- feed the audio to :class:`src.asr.streaming.StreamingTranscriber`
  chunk-by-chunk and record the time until the first partial result and the
  final result.

A mock ASR provider is included so the harness can be exercised in CI or unit
runs without downloading a real model.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from src.asr.base import ASRProvider, TranscriptionResult
from src.asr.streaming import StreamingTranscriber
from src.audio.vad import VADProvider
from src.config import Settings

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Sequence


@dataclass
class LatencyResult:
    """Latency metrics for one benchmark run.

    Attributes:
        mode: Benchmark mode, either ``streaming`` or ``batch``.
        time_to_first_partial: Seconds until the first non-final (partial)
            streaming result. For batch mode this equals ``time_to_final``.
        time_to_final: Seconds until the final result is available.
        total_audio_seconds: Duration of the input audio in seconds.
        rtf: Real-time factor, ``time_to_final / total_audio_seconds``.
        transcript: Recognized text produced by the run.
    """

    mode: str
    time_to_first_partial: float | None
    time_to_final: float
    total_audio_seconds: float
    rtf: float
    transcript: str

    def summary(self) -> dict[str, object]:
        """Return a JSON-serializable dictionary of the metrics."""
        return {
            "mode": self.mode,
            "time_to_first_partial": self.time_to_first_partial,
            "time_to_final": self.time_to_final,
            "total_audio_seconds": self.total_audio_seconds,
            "rtf": self.rtf,
            "transcript": self.transcript,
        }


class AlwaysSpeechVAD(VADProvider):
    """VAD provider that treats every chunk as speech.

    Used by the benchmark harness so that streaming runs exercise the ASR path
    for every chunk without depending on a real VAD implementation.
    """

    def is_speech(self, audio: np.ndarray, sample_rate: int) -> bool:  # noqa: ARG002
        """Always return ``True``."""
        return True

    def process_stream(
        self,
        chunks: list[np.ndarray],
        sample_rate: int,  # noqa: ARG002
    ) -> list[tuple[np.ndarray, bool]]:
        """Return every chunk paired with ``True``."""
        return [(chunk, True) for chunk in chunks]


class MockASRProvider(ASRProvider):
    """Fake ASR provider that sleeps a configurable amount per transcription.

    This provider lets benchmarks run without downloading or loading a real
    Whisper model. Each call to :meth:`transcribe` increments ``call_count`` and
    sleeps for ``sleep_per_chunk`` seconds.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        sleep_per_chunk: float = 0.001,
        text: str = "benchmark transcript",
    ) -> None:
        """Initialize the mock provider.

        Args:
            settings: Parsed application settings.
            sleep_per_chunk: Seconds to sleep on each transcription call.
            text: Text returned by every transcription.
        """
        super().__init__(settings)
        self.sleep_per_chunk = sleep_per_chunk
        self.text = text
        self.call_count = 0

    def load_model(self) -> None:
        """No-op for the mock provider."""

    def warmup(self) -> None:
        """No-op for the mock provider."""

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int,
        beam_size: int | None = None,  # noqa: ARG002
    ) -> TranscriptionResult:  # noqa: ARG002
        """Return deterministic text after a short sleep."""
        self.call_count += 1
        time.sleep(self.sleep_per_chunk)
        return TranscriptionResult(
            text=self.text,
            is_final=True,
            confidence=0.9,
            language="en",
        )

    def transcribe_streaming(
        self,
        audio_chunk: np.ndarray,
        sample_rate: int,
        beam_size: int | None = None,
    ) -> TranscriptionResult:
        """Delegate to :meth:`transcribe`."""
        return self.transcribe(audio_chunk, sample_rate, beam_size=beam_size)


class _TimedASRProvider(ASRProvider):
    """Wraps an ASR provider and records the first ``transcribe`` call time."""

    def __init__(
        self,
        delegate: ASRProvider,
        start_time: float,
    ) -> None:
        """Initialize the wrapper.

        Args:
            delegate: The real ASR provider to delegate transcription to.
            start_time: ``time.perf_counter()`` value for the benchmark start.
        """
        super().__init__(delegate.settings)
        self._delegate = delegate
        self._start = start_time
        self.first_call_elapsed: float | None = None

    def load_model(self) -> None:
        """Delegate model loading."""
        self._delegate.load_model()

    def warmup(self) -> None:
        """Delegate warmup."""
        self._delegate.warmup()

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int,
        beam_size: int | None = None,
    ) -> TranscriptionResult:
        """Record first-call latency and delegate transcription."""
        if self.first_call_elapsed is None:
            self.first_call_elapsed = time.perf_counter() - self._start
        return self._delegate.transcribe(audio, sample_rate, beam_size=beam_size)

    def transcribe_streaming(
        self,
        audio_chunk: np.ndarray,
        sample_rate: int,
        beam_size: int | None = None,
    ) -> TranscriptionResult:
        """Delegate streaming transcription."""
        if self.first_call_elapsed is None:
            self.first_call_elapsed = time.perf_counter() - self._start
        return self._delegate.transcribe_streaming(
            audio_chunk,
            sample_rate,
            beam_size=beam_size,
        )


class LatencyBenchmark:
    """Measure and compare transcription latency for streaming and batch modes."""

    def __init__(
        self,
        settings: Settings,
        asr_provider: ASRProvider,
        audio: np.ndarray,
        mode: str,
        *,
        chunk_size: int | None = None,
    ) -> None:
        """Initialize the benchmark.

        Args:
            settings: Parsed application settings.
            asr_provider: ASR provider to benchmark.
            audio: One-dimensional ``float32`` audio samples.
            mode: Either ``streaming`` or ``batch``.
            chunk_size: Number of samples per streaming chunk. When ``None``,
                ``settings.audio_sample_rate * settings.streaming_chunk_seconds``
                is used.

        Raises:
            ValueError: If ``mode`` is not ``streaming`` or ``batch``.
        """
        if mode not in {"streaming", "batch"}:
            raise ValueError(f"mode must be 'streaming' or 'batch', got {mode!r}")

        self.settings = settings
        self.asr_provider = asr_provider
        self.audio = np.asarray(audio, dtype=np.float32)
        self.mode = mode
        self.sample_rate = settings.audio_sample_rate

        if chunk_size is None:
            chunk_seconds = getattr(settings, "streaming_chunk_seconds", 1.0)
            self.chunk_size = max(1, int(self.sample_rate * chunk_seconds))
        else:
            self.chunk_size = max(1, int(chunk_size))

    def run(self) -> LatencyResult:
        """Run the benchmark and return latency metrics."""
        if self.mode == "batch":
            return self._run_batch()
        return self._run_streaming()

    def _audio_duration(self) -> float:
        """Return the duration of the input audio in seconds."""
        return self.audio.shape[0] / self.sample_rate if self.sample_rate > 0 else 0.0

    def _compute_rtf(self, total_time: float) -> float:
        """Compute the real-time factor for a measured total time."""
        duration = self._audio_duration()
        return total_time / duration if duration > 0 else 0.0

    def _run_batch(self) -> LatencyResult:
        """Run a single full-file transcription and measure its latency."""
        start = time.perf_counter()
        result = self.asr_provider.transcribe(self.audio, self.sample_rate)
        elapsed = time.perf_counter() - start

        return LatencyResult(
            mode="batch",
            time_to_first_partial=elapsed,
            time_to_final=elapsed,
            total_audio_seconds=self._audio_duration(),
            rtf=self._compute_rtf(elapsed),
            transcript=result.text,
        )

    def _run_streaming(self) -> LatencyResult:
        """Run chunk-by-chunk streaming transcription and measure latency."""
        vad = AlwaysSpeechVAD()
        start = time.perf_counter()
        timed_asr = _TimedASRProvider(self.asr_provider, start)
        transcriber = StreamingTranscriber(self.settings, timed_asr, vad)
        transcriber.loop_interval_seconds = 0.005
        transcriber.streaming_chunk_seconds = self.chunk_size / self.sample_rate

        chunks = self._chunk_audio()
        collected: list[TranscriptionResult] = []

        transcriber.start()
        try:
            for chunk in chunks:
                transcriber.add_audio(chunk)
                # Give the background thread a chance to process the new audio.
                time.sleep(0.005)
                self._drain_results(transcriber, collected)

            # Wait for the background thread to finish processing queued chunks.
            deadline = time.perf_counter() + 2.0
            chunk_seconds = self.chunk_size / self.sample_rate
            while (
                transcriber._buffer.unprocessed_duration() >= chunk_seconds
                and time.perf_counter() < deadline
            ):
                time.sleep(0.005)
                self._drain_results(transcriber, collected)
        finally:
            transcriber.stop()
            final_time = time.perf_counter() - start
            self._drain_results(transcriber, collected)

        transcript = self._build_transcript(collected)
        return LatencyResult(
            mode="streaming",
            time_to_first_partial=timed_asr.first_call_elapsed,
            time_to_final=final_time,
            total_audio_seconds=self._audio_duration(),
            rtf=self._compute_rtf(final_time),
            transcript=transcript,
        )

    def _chunk_audio(self) -> Sequence[np.ndarray]:
        """Split the input audio into chunk-sized arrays."""
        if self.audio.size == 0:
            return [self.audio]

        return [
            self.audio[i : i + self.chunk_size]
            for i in range(0, self.audio.shape[0], self.chunk_size)
        ]

    def _drain_results(
        self,
        transcriber: StreamingTranscriber,
        collected: list[TranscriptionResult],
    ) -> None:
        """Move available results from the transcriber queue into ``collected``."""
        collected.extend(transcriber.get_results())

    def _build_transcript(self, collected: list[TranscriptionResult]) -> str:
        """Build a single transcript string from collected results."""
        parts = [result.text for result in collected if result.text]
        return " ".join(parts).strip()
