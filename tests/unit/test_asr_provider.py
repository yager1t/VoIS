"""Unit tests for the faster-whisper ASR provider."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.asr.whisper_provider import FasterWhisperProvider
from src.config import Settings

SAMPLE_RATE = 16000


@dataclass
class _FakeSegment:
    """Minimal stand-in for a faster-whisper segment."""

    text: str
    avg_logprob: float


@dataclass
class _FakeInfo:
    """Minimal stand-in for faster-whisper transcription info."""

    language: str


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Return settings that keep models inside a temporary directory."""
    return Settings(
        asr_model="tiny",
        asr_language="en",
        models_dir=tmp_path / "models",
    )


@pytest.fixture
def fake_model() -> MagicMock:
    """Return a mocked WhisperModel with deterministic transcription output."""
    model = MagicMock()
    model.transcribe.return_value = (
        [
            _FakeSegment(text="Hello,", avg_logprob=-0.1),
            _FakeSegment(text=" world", avg_logprob=-0.2),
        ],
        _FakeInfo(language="en"),
    )
    return model


@pytest.fixture
def mock_bias() -> MagicMock:
    """Return a mock ASRBias that provides non-empty prompt and hotwords."""
    bias = MagicMock()
    bias.initial_prompt.return_value = "Transcribe a formal email. Important terms: OAuth."
    bias.hotwords.return_value = ["OAuth", "gRPC"]
    return bias


def test_transcribe_returns_expected_text(
    settings: Settings,
    fake_model: MagicMock,
) -> None:
    """The provider should concatenate segment text into the result."""
    provider = FasterWhisperProvider(settings)

    with patch.object(provider, "_model", fake_model):
        audio = np.zeros(SAMPLE_RATE, dtype=np.float32)
        result = provider.transcribe(audio, SAMPLE_RATE)

    assert result.text == "Hello, world"
    assert result.is_final is True
    assert result.language == "en"
    assert result.confidence is not None
    assert result.confidence == pytest.approx(-0.15)


def test_transcribe_uses_language_from_settings(settings: Settings) -> None:
    """When asr_language is not auto, it should be passed to the model."""
    provider = FasterWhisperProvider(settings)
    model = MagicMock()
    model.transcribe.return_value = ([_FakeSegment("Bonjour", -0.1)], _FakeInfo("fr"))

    with patch.object(provider, "_model", model):
        provider.transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32), SAMPLE_RATE)

    _, kwargs = model.transcribe.call_args
    assert kwargs["language"] == "en"


def test_transcribe_auto_language_passes_none(settings: Settings) -> None:
    """When asr_language is auto, no language hint should be passed."""
    settings.asr_language = "auto"
    provider = FasterWhisperProvider(settings)
    model = MagicMock()
    model.transcribe.return_value = ([_FakeSegment("hi", -0.1)], _FakeInfo("en"))

    with patch.object(provider, "_model", model):
        provider.transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32), SAMPLE_RATE)

    _, kwargs = model.transcribe.call_args
    assert kwargs["language"] is None


def test_transcribe_empty_audio_returns_empty_result(settings: Settings) -> None:
    """Empty audio should short-circuit and return an empty result."""
    provider = FasterWhisperProvider(settings)
    model = MagicMock()

    with patch.object(provider, "_model", model):
        result = provider.transcribe(np.array([], dtype=np.float32), SAMPLE_RATE)

    assert result.text == ""
    assert result.confidence is None
    model.transcribe.assert_not_called()


def test_streaming_delegates_to_transcribe(settings: Settings) -> None:
    """Streaming currently returns a final transcription for the chunk."""
    provider = FasterWhisperProvider(settings)
    model = MagicMock()
    model.transcribe.return_value = ([_FakeSegment("chunk", -0.1)], _FakeInfo("en"))

    with patch.object(provider, "_model", model):
        result = provider.transcribe_streaming(
            np.zeros(SAMPLE_RATE // 2, dtype=np.float32),
            SAMPLE_RATE,
        )

    assert result.text == "chunk"
    assert result.is_final is True


def test_load_model_caches_model(settings: Settings) -> None:
    """load_model should only instantiate the model once."""
    provider = FasterWhisperProvider(settings)

    with patch.object(
        provider._model_manager,
        "load_whisper_model",
        return_value=MagicMock(),
    ) as mock_load:
        provider.load_model()
        provider.load_model()

    mock_load.assert_called_once()


def test_load_model_calls_model_manager_with_correct_args(settings: Settings) -> None:
    """load_model should delegate to ModelManager with settings-derived arguments."""
    provider = FasterWhisperProvider(settings)

    with patch.object(
        provider._model_manager,
        "load_whisper_model",
        return_value=MagicMock(),
    ) as mock_load:
        provider.load_model()

    mock_load.assert_called_once_with(
        settings.asr_model,
        device=settings.asr_device,
        compute_type=settings.asr_compute_type,
    )


def test_transcribe_cleans_up_temp_wav(settings: Settings) -> None:
    """Transcribe should remove the temporary WAV file after transcription."""
    provider = FasterWhisperProvider(settings)
    model = MagicMock()
    model.transcribe.return_value = ([_FakeSegment("hello", -0.1)], _FakeInfo("en"))

    written_path = None
    original_write = __import__("soundfile").write

    def capture_write(path, *args, **kwargs):
        nonlocal written_path
        written_path = Path(path)
        original_write(path, *args, **kwargs)

    with (
        patch.object(provider, "_model", model),
        patch("soundfile.write", side_effect=capture_write),
    ):
        provider.transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32), SAMPLE_RATE)

    assert written_path is not None
    assert not written_path.exists()


def test_transcribe_handles_exception_and_cleans_up(settings: Settings) -> None:
    """Transcribe should propagate exceptions and still remove the temp file."""
    provider = FasterWhisperProvider(settings)
    model = MagicMock()
    model.transcribe.side_effect = RuntimeError("transcription failed")

    written_path = None
    original_write = __import__("soundfile").write

    def capture_write(path, *args, **kwargs):
        nonlocal written_path
        written_path = Path(path)
        original_write(path, *args, **kwargs)

    with (
        patch.object(provider, "_model", model),
        patch("soundfile.write", side_effect=capture_write),
        pytest.raises(RuntimeError, match="transcription failed"),
    ):
        provider.transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32), SAMPLE_RATE)

    assert written_path is not None
    assert not written_path.exists()


def test_transcribe_passes_bias_when_dictionary_enabled(
    settings: Settings,
    fake_model: MagicMock,
    mock_bias: MagicMock,
) -> None:
    """When dictionary is enabled, bias hints are passed to the model."""
    settings.dictionary_enabled = True
    provider = FasterWhisperProvider(settings)

    with (
        patch.object(provider, "_model", fake_model),
        patch.object(provider, "_get_bias", return_value=mock_bias),
    ):
        provider.transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32), SAMPLE_RATE)

    _, kwargs = fake_model.transcribe.call_args
    assert kwargs["initial_prompt"] == "Transcribe a formal email. Important terms: OAuth."
    assert kwargs["hotwords"] == "OAuth,gRPC"


def test_transcribe_does_not_pass_bias_when_dictionary_disabled(
    settings: Settings,
    fake_model: MagicMock,
    mock_bias: MagicMock,
) -> None:
    """When dictionary is disabled, bias hints are not passed to the model."""
    settings.dictionary_enabled = False
    provider = FasterWhisperProvider(settings)

    with (
        patch.object(provider, "_model", fake_model),
        patch.object(provider, "_get_bias", return_value=mock_bias),
    ):
        provider.transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32), SAMPLE_RATE)

    _, kwargs = fake_model.transcribe.call_args
    assert "initial_prompt" not in kwargs
    assert "hotwords" not in kwargs
