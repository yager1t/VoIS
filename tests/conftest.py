"""Shared pytest fixtures for Voice-to-Cursor tests."""

from __future__ import annotations

from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from pytest import Config, Item, Parser

from src.config import Settings

SAMPLE_RATE = 16000
DEFAULT_DURATION = 0.5


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Return settings isolated to a temporary data/models directory."""
    return Settings(
        data_dir=tmp_path / "data",
        models_dir=tmp_path / "models",
    )


@pytest.fixture
def sample_rate() -> int:
    """Default audio sample rate used across tests."""
    return SAMPLE_RATE


def _generate_sine(
    frequency: float,
    duration: float,
    sample_rate: int,
    amplitude: float = 0.5,
) -> np.ndarray:
    """Generate a mono sine wave as float32 samples."""
    t = np.linspace(0.0, duration, int(sample_rate * duration), endpoint=False)
    return (amplitude * np.sin(2.0 * np.pi * frequency * t)).astype(np.float32)


@pytest.fixture
def synthetic_speech(sample_rate: int) -> np.ndarray:
    """Return a sine wave simulating voiced audio."""
    return _generate_sine(frequency=400.0, duration=DEFAULT_DURATION, sample_rate=sample_rate)


@pytest.fixture
def synthetic_silence(sample_rate: int) -> np.ndarray:
    """Return a silent float32 audio segment."""
    return np.zeros(int(sample_rate * DEFAULT_DURATION), dtype=np.float32)


@pytest.fixture
def synthetic_mixed_audio(sample_rate: int) -> np.ndarray:
    """Return silence + sine + silence, at least 1.5 seconds long."""
    silence = np.zeros(int(sample_rate * DEFAULT_DURATION), dtype=np.float32)
    speech = _generate_sine(frequency=400.0, duration=DEFAULT_DURATION, sample_rate=sample_rate)
    return np.concatenate([silence, speech, silence], dtype=np.float32)


@dataclass
class FakeWhisperSegment:
    """Minimal stand-in for a faster-whisper segment."""

    text: str
    avg_logprob: float


@dataclass
class FakeWhisperInfo:
    """Minimal stand-in for faster-whisper transcription info."""

    language: str


@pytest.fixture
def mock_whisper_model() -> MagicMock:
    """Return a MagicMock that mimics a faster-whisper.WhisperModel."""
    model = MagicMock()
    model.transcribe.return_value = (
        [FakeWhisperSegment(text="Hello, world", avg_logprob=-0.1)],
        FakeWhisperInfo(language="en"),
    )
    return model


@pytest.fixture
def patched_model_manager() -> Generator[MagicMock, None, None]:
    """Patch faster-whisper download/create wrappers in src.asr.model_manager."""
    mock_model = MagicMock()
    with (
        patch("src.asr.model_manager._download_model") as mock_download,
        patch(
            "src.asr.model_manager._create_whisper_model",
            return_value=mock_model,
        ) as mock_create,
    ):
        manager_mock = MagicMock()
        manager_mock.download = mock_download
        manager_mock.create = mock_create
        manager_mock.model = mock_model
        yield manager_mock


def pytest_addoption(parser: Parser) -> None:
    """Register the ``--run-smoke`` CLI option for smoke tests."""
    parser.addoption(
        "--run-smoke",
        action="store_true",
        default=False,
        help="run smoke tests",
    )


def pytest_configure(config: Config) -> None:
    """Register custom markers used by the test suite."""
    config.addinivalue_line(
        "markers",
        "smoke: marks tests requiring real hardware/OS interaction",
    )


def pytest_collection_modifyitems(config: Config, items: list[Item]) -> None:
    """Skip smoke tests unless ``--run-smoke`` was requested."""
    if not config.getoption("--run-smoke"):
        skip_smoke = pytest.mark.skip(reason="need --run-smoke option to run")
        for item in items:
            if "smoke" in item.keywords:
                item.add_marker(skip_smoke)
