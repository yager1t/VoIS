"""Unit tests for the streaming vs batch latency benchmark harness."""

from __future__ import annotations

import numpy as np
import pytest

from src.benchmarks.latency import (
    AlwaysSpeechVAD,
    LatencyBenchmark,
    LatencyResult,
    MockASRProvider,
)
from src.config import Settings

SAMPLE_RATE = 16000


@pytest.fixture
def settings(tmp_path) -> Settings:
    """Return isolated settings for latency benchmarks."""
    return Settings(
        data_dir=tmp_path / "data",
        models_dir=tmp_path / "models",
        audio_sample_rate=SAMPLE_RATE,
        streaming_chunk_seconds=0.5,
    )


@pytest.fixture
def one_second_audio() -> np.ndarray:
    """Return one second of float32 audio."""
    return np.ones(SAMPLE_RATE, dtype=np.float32)


@pytest.fixture
def mock_asr(settings) -> MockASRProvider:
    """Return a mock ASR provider with a tiny per-chunk sleep."""
    return MockASRProvider(settings, sleep_per_chunk=0.001, text="benchmark")


def test_latency_result_summary() -> None:
    """``LatencyResult.summary`` must expose all metrics."""
    result = LatencyResult(
        mode="batch",
        time_to_first_partial=0.1,
        time_to_final=0.1,
        total_audio_seconds=1.0,
        rtf=0.1,
        transcript="hello",
    )
    summary = result.summary()
    assert summary["mode"] == "batch"
    assert summary["time_to_first_partial"] == 0.1
    assert summary["time_to_final"] == 0.1
    assert summary["total_audio_seconds"] == 1.0
    assert summary["rtf"] == 0.1
    assert summary["transcript"] == "hello"


def test_always_speech_vad() -> None:
    """``AlwaysSpeechVAD`` must report every frame as speech."""
    vad = AlwaysSpeechVAD()
    audio = np.zeros(160, dtype=np.float32)
    assert vad.is_speech(audio, SAMPLE_RATE) is True
    chunks = [audio, audio]
    assert vad.process_stream(chunks, SAMPLE_RATE) == [(audio, True), (audio, True)]


def test_mock_asr_counts_calls(settings) -> None:
    """``MockASRProvider`` must track how many times it was called."""
    asr = MockASRProvider(settings, text="hello")
    audio = np.zeros(160, dtype=np.float32)
    result = asr.transcribe(audio, SAMPLE_RATE)
    assert result.text == "hello"
    assert asr.call_count == 1


def test_batch_benchmark(settings, one_second_audio, mock_asr) -> None:
    """Batch mode must produce a result with sensible metrics."""
    benchmark = LatencyBenchmark(settings, mock_asr, one_second_audio, "batch")
    result = benchmark.run()

    assert result.mode == "batch"
    assert result.time_to_first_partial == pytest.approx(result.time_to_final)
    assert result.time_to_final > 0
    assert result.total_audio_seconds == pytest.approx(1.0)
    assert result.rtf > 0
    assert result.transcript == "benchmark"


def test_streaming_benchmark(settings, one_second_audio, mock_asr) -> None:
    """Streaming mode must produce results and a plausible RTF."""
    benchmark = LatencyBenchmark(
        settings,
        mock_asr,
        one_second_audio,
        "streaming",
        chunk_size=SAMPLE_RATE // 2,
    )
    result = benchmark.run()

    assert result.mode == "streaming"
    assert result.time_to_final > 0
    assert result.total_audio_seconds == pytest.approx(1.0)
    assert result.rtf > 0
    assert "benchmark" in result.transcript


def test_streaming_reports_first_partial(settings, one_second_audio, mock_asr) -> None:
    """Streaming mode should report a first partial time for speech audio."""
    benchmark = LatencyBenchmark(
        settings,
        mock_asr,
        one_second_audio,
        "streaming",
        chunk_size=SAMPLE_RATE // 2,
    )
    result = benchmark.run()

    assert result.time_to_first_partial is not None
    assert 0 < result.time_to_first_partial <= result.time_to_final
    assert "benchmark" in result.transcript


def test_invalid_mode_raises(settings, one_second_audio, mock_asr) -> None:
    """An unsupported mode must raise ``ValueError``."""
    with pytest.raises(ValueError, match="mode must be 'streaming' or 'batch'"):
        LatencyBenchmark(settings, mock_asr, one_second_audio, "realtime")


def test_empty_audio_streaming(settings, mock_asr) -> None:
    """Streaming mode must handle empty audio without crashing."""
    audio = np.array([], dtype=np.float32)
    benchmark = LatencyBenchmark(
        settings,
        mock_asr,
        audio,
        "streaming",
        chunk_size=SAMPLE_RATE // 2,
    )
    result = benchmark.run()

    assert result.mode == "streaming"
    assert result.total_audio_seconds == 0.0
    assert result.rtf == 0.0


def test_empty_audio_batch(settings, mock_asr) -> None:
    """Batch mode must handle empty audio without crashing."""
    audio = np.array([], dtype=np.float32)
    benchmark = LatencyBenchmark(settings, mock_asr, audio, "batch")
    result = benchmark.run()

    assert result.mode == "batch"
    assert result.total_audio_seconds == 0.0
    assert result.rtf == 0.0
    assert result.transcript == "benchmark"


@pytest.mark.slow
@pytest.mark.requires_model
def test_real_model_batch_benchmark() -> None:
    """Real-model benchmark path (excluded from default test runs).

    This test exists to verify the benchmark can accept a real
    :class:`src.asr.whisper_provider.FasterWhisperProvider`. It is skipped
    unless pytest is invoked with ``--run-slow`` or the markers are explicitly
    selected, because it downloads or loads a Whisper model.
    """
    settings = Settings(audio_sample_rate=SAMPLE_RATE)
    from src.asr.whisper_provider import FasterWhisperProvider

    asr = FasterWhisperProvider(settings)
    audio = np.zeros(SAMPLE_RATE // 2, dtype=np.float32)
    benchmark = LatencyBenchmark(settings, asr, audio, "batch")
    benchmark.run()
