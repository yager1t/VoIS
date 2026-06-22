"""Unit tests for voice activity detection providers."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.audio import vad as vad_module
from src.audio.vad import VADProvider, WebRTCVADProvider, float_to_int16, frame_audio

SAMPLE_RATE = 16000


def generate_sine(
    frequency: float,
    duration: float,
    sample_rate: int,
    amplitude: float = 0.5,
) -> np.ndarray:
    """Generate a mono sine wave as float32 samples."""
    t = np.linspace(0.0, duration, int(sample_rate * duration), endpoint=False)
    return (amplitude * np.sin(2.0 * np.pi * frequency * t)).astype(np.float32)


def test_float_to_int16_clips() -> None:
    """Conversion must clip to int16 range."""
    audio = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    result = float_to_int16(audio)

    expected = np.array([-32767, -32767, 0, 32767, 32767], dtype=np.int16)
    assert np.array_equal(result, expected)


def test_frame_audio_sizes() -> None:
    """frame_audio should split into 30 ms frames by default."""
    frame_size = int(SAMPLE_RATE * 0.03)
    audio = np.ones(frame_size * 4 + 10, dtype=np.float32)
    frames = frame_audio(audio, SAMPLE_RATE)

    assert len(frames) == 4
    assert all(frame.shape == (frame_size,) for frame in frames)


def test_frame_audio_rejects_invalid_duration() -> None:
    """Only 10, 20, and 30 ms frames are allowed."""
    with pytest.raises(ValueError):
        frame_audio(np.ones(100, dtype=np.float32), SAMPLE_RATE, frame_ms=15)


def test_abstract_methods() -> None:
    """A concrete subclass must implement both VADProvider methods."""

    class DummyVAD(VADProvider):
        def is_speech(self, audio: np.ndarray, sample_rate: int) -> bool:
            return True

        def process_stream(
            self,
            chunks: list[np.ndarray],
            sample_rate: int,
        ) -> list[tuple[np.ndarray, bool]]:
            return [(chunk, True) for chunk in chunks]

    dummy = DummyVAD()
    assert dummy.is_speech(np.zeros(10), SAMPLE_RATE) is True


def test_webrtcvad_not_installed_raises() -> None:
    """WebRTCVADProvider should raise if webrtcvad is unavailable."""
    with patch.object(vad_module, "webrtcvad", None), pytest.raises(RuntimeError):
        WebRTCVADProvider()


def test_webrtcvad_silence(mock_webrtcvad: MagicMock) -> None:
    """Silence frames should be classified as non-speech."""
    mock_instance = mock_webrtcvad.Vad.return_value
    mock_instance.is_speech.return_value = False

    provider = WebRTCVADProvider(aggressiveness=2)
    silence = np.zeros(int(SAMPLE_RATE * 0.03), dtype=np.float32)

    assert provider.is_speech(silence, SAMPLE_RATE) is False


def test_webrtcvad_sine_speech(mock_webrtcvad: MagicMock) -> None:
    """A sine wave frame should be classified as speech when mocked so."""
    mock_instance = mock_webrtcvad.Vad.return_value
    mock_instance.is_speech.return_value = True

    provider = WebRTCVADProvider(aggressiveness=1)
    speech = generate_sine(frequency=400.0, duration=0.03, sample_rate=SAMPLE_RATE)
    assert speech.size == int(SAMPLE_RATE * 0.03)

    assert provider.is_speech(speech, SAMPLE_RATE) is True


def test_process_stream_pairs(mock_webrtcvad: MagicMock) -> None:
    """process_stream should return each chunk paired with its decision."""
    mock_instance = mock_webrtcvad.Vad.return_value
    mock_instance.is_speech.side_effect = [True, False, True]

    provider = WebRTCVADProvider()
    chunks = [
        generate_sine(400.0, 0.03, SAMPLE_RATE),
        np.zeros(int(SAMPLE_RATE * 0.03), dtype=np.float32),
        generate_sine(800.0, 0.03, SAMPLE_RATE),
    ]
    result = provider.process_stream(chunks, SAMPLE_RATE)

    assert len(result) == 3
    assert result[0][1] is True
    assert result[1][1] is False
    assert result[2][1] is True


def test_split_on_silence_different_trim_seconds() -> None:
    """split_on_silence should keep context according to keep_chunks."""

    class KeepAllVAD(VADProvider):
        def is_speech(self, audio: np.ndarray, sample_rate: int) -> bool:
            return True

        def process_stream(
            self,
            chunks: list[np.ndarray],
            sample_rate: int,
        ) -> list[tuple[np.ndarray, bool]]:
            return [(chunk, True) for chunk in chunks]

    frame_size = int(SAMPLE_RATE * 0.03)
    audio = np.arange(frame_size * 5, dtype=np.float32)
    vad = KeepAllVAD()

    segments = vad.split_on_silence(audio, SAMPLE_RATE, frame_ms=30, keep_chunks=1)
    assert len(segments) == 1
    assert segments[0].size == frame_size * 5


def test_split_on_silence_keeps_actual_context_frames() -> None:
    """Context around speech should use neighboring silence, not duplicate speech."""

    class PatternVAD(VADProvider):
        def __init__(self) -> None:
            self._decisions = iter([False, True, False])

        def is_speech(self, audio: np.ndarray, sample_rate: int) -> bool:
            return next(self._decisions)

        def process_stream(
            self,
            chunks: list[np.ndarray],
            sample_rate: int,
        ) -> list[tuple[np.ndarray, bool]]:
            return [(chunk, True) for chunk in chunks]

    frame_size = int(SAMPLE_RATE * 0.03)
    silence_before = np.full(frame_size, 1, dtype=np.float32)
    speech = np.full(frame_size, 2, dtype=np.float32)
    silence_after = np.full(frame_size, 3, dtype=np.float32)
    audio = np.concatenate([silence_before, speech, silence_after])

    segments = PatternVAD().split_on_silence(audio, SAMPLE_RATE, frame_ms=30, keep_chunks=1)

    assert len(segments) == 1
    assert np.array_equal(segments[0], audio)


def test_split_on_silence_short_audio() -> None:
    """Audio shorter than one frame should produce no segments."""

    class KeepAllVAD(VADProvider):
        def is_speech(self, audio: np.ndarray, sample_rate: int) -> bool:
            return True

        def process_stream(
            self,
            chunks: list[np.ndarray],
            sample_rate: int,
        ) -> list[tuple[np.ndarray, bool]]:
            return [(chunk, True) for chunk in chunks]

    audio = np.zeros(10, dtype=np.float32)
    vad = KeepAllVAD()
    segments = vad.split_on_silence(audio, SAMPLE_RATE, frame_ms=30)
    assert segments == []


def test_is_speech_rejects_invalid_sample_rate(mock_webrtcvad: MagicMock) -> None:
    """WebRTCVADProvider.is_speech should raise for unsupported sample rates."""
    mock_instance = mock_webrtcvad.Vad.return_value
    mock_instance.is_speech.side_effect = ValueError("invalid sample rate")

    provider = WebRTCVADProvider()
    with pytest.raises(ValueError, match="invalid sample rate"):
        provider.is_speech(np.zeros(int(SAMPLE_RATE * 0.03), dtype=np.float32), 12345)


@pytest.fixture
def mock_webrtcvad() -> MagicMock:
    """Replace the webrtcvad module with a MagicMock for the test function."""
    with patch.object(vad_module, "webrtcvad", new_callable=MagicMock) as mocked:
        yield mocked
