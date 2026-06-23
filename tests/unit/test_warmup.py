"""Unit tests for ASR provider warmup."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.asr.whisper_provider import FasterWhisperProvider
from src.config import Settings

SAMPLE_RATE = 16000


@pytest.fixture
def settings(tmp_path) -> Settings:
    """Return minimal settings for warmup tests."""
    return Settings(
        asr_model="tiny",
        asr_language="en",
        models_dir=tmp_path / "models",
    )


def test_warmup_transcribes_one_second_of_silence(settings: Settings) -> None:
    """warmup() should call transcribe with one second of zeros."""
    provider = FasterWhisperProvider(settings)
    transcribe_mock = MagicMock(return_value=MagicMock(text=""))

    with patch.object(provider, "transcribe", transcribe_mock):
        provider.warmup()

    transcribe_mock.assert_called_once()
    audio_arg, sample_rate_arg = transcribe_mock.call_args.args
    assert sample_rate_arg == settings.audio_sample_rate
    assert audio_arg.dtype == np.float32
    assert audio_arg.shape == (settings.audio_sample_rate,)
    assert np.all(audio_arg == 0.0)
