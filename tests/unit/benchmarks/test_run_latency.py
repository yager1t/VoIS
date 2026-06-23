"""Unit tests for the benchmark runner CLI."""

from __future__ import annotations

import json
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf

from benchmarks import run_latency
from src.config import Settings


@pytest.fixture
def wav_file(tmp_path: Path) -> Path:
    """Return the path to a short synthetic WAV file."""
    path = tmp_path / "test.wav"
    audio = np.zeros(16000, dtype=np.float32)
    sf.write(path, audio, 16000)
    return path


def test_generate_silence() -> None:
    """``generate_silence`` must produce the expected number of samples."""
    audio = run_latency.generate_silence(duration=0.5, sample_rate=16000)
    assert audio.shape == (8000,)
    assert audio.dtype == np.float32
    assert np.all(audio == 0)


def test_load_audio_valid(wav_file: Path) -> None:
    """``load_audio`` must load a mono WAV file."""
    audio = run_latency.load_audio(wav_file, sample_rate=16000)
    assert audio.shape == (16000,)
    assert audio.dtype == np.float32


def test_load_audio_wrong_sample_rate(wav_file: Path) -> None:
    """``load_audio`` must reject a sample rate mismatch."""
    with pytest.raises(ValueError, match="Sample rate mismatch"):
        run_latency.load_audio(wav_file, sample_rate=8000)


def test_load_or_generate_audio_uses_wav(wav_file: Path) -> None:
    """When a WAV path is supplied, synthetic audio is not generated."""
    audio = run_latency.load_or_generate_audio(wav_file, duration=10.0, sample_rate=16000)
    assert audio.shape == (16000,)


def test_load_or_generate_audio_generates_silence() -> None:
    """When no WAV path is supplied, silence is generated."""
    audio = run_latency.load_or_generate_audio(None, duration=0.1, sample_rate=16000)
    assert audio.shape == (1600,)


def test_create_mock_provider(settings: Settings) -> None:
    """``create_asr_provider`` returns a mock provider by default."""
    provider = run_latency.create_asr_provider(settings, real_model=False)
    assert provider.__class__.__name__ == "MockASRProvider"


def test_parse_args_defaults() -> None:
    """Default CLI arguments must be sensible."""
    args = run_latency.parse_args([])
    assert args.mode == "both"
    assert args.duration == 2.0
    assert args.chunk_seconds == 1.0
    assert args.sample_rate == 16000
    assert args.real_model is False


def test_main_batch_mode() -> None:
    """``main`` must print a JSON summary for batch mode."""
    with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
        code = run_latency.main(["--mode", "batch", "--duration", "0.2"])

    assert code == 0
    output = mock_stdout.getvalue()
    summary = json.loads(output)
    assert summary["settings"]["mode"] == "batch"
    assert "batch" in summary["results"]
    assert summary["results"]["batch"]["transcript"] == "benchmark transcript"


def test_main_streaming_mode() -> None:
    """``main`` must print a JSON summary for streaming mode."""
    with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
        code = run_latency.main([
            "--mode",
            "streaming",
            "--duration",
            "0.2",
            "--chunk-seconds",
            "0.1",
        ])

    assert code == 0
    summary = json.loads(mock_stdout.getvalue())
    assert "streaming" in summary["results"]
    assert summary["results"]["streaming"]["total_audio_seconds"] == pytest.approx(0.2)


def test_main_both_modes() -> None:
    """``main`` must run both modes when requested."""
    with patch("sys.stdout", new_callable=StringIO) as mock_stdout:
        code = run_latency.main(["--mode", "both", "--duration", "0.2"])

    assert code == 0
    summary = json.loads(mock_stdout.getvalue())
    assert "batch" in summary["results"]
    assert "streaming" in summary["results"]
