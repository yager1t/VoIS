"""Unit tests for audio capture."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import sounddevice as sd

from src.audio.capture import AudioCapture
from src.config import Settings


@pytest.fixture
def settings() -> Settings:
    """Return capture-friendly settings."""
    return Settings(audio_sample_rate=16000, audio_channels=1, push_to_talk=True)


def test_init_from_settings(settings: Settings) -> None:
    """Capture should derive values from Settings when no kwargs are given."""
    capture = AudioCapture(settings)

    assert capture.sample_rate == settings.audio_sample_rate
    assert capture.channels == settings.audio_channels
    assert capture.push_to_talk is settings.push_to_talk
    assert capture.device is None
    assert capture.block_size == 1024


def test_init_kwargs_override_settings(settings: Settings) -> None:
    """Explicit kwargs should override Settings values."""
    capture = AudioCapture(
        settings,
        sample_rate=44100,
        channels=2,
        device="default",
        push_to_talk=False,
        block_size=512,
    )

    assert capture.sample_rate == 44100
    assert capture.channels == 2
    assert capture.device == "default"
    assert capture.push_to_talk is False
    assert capture.block_size == 512


def test_init_without_settings() -> None:
    """Default values should be used when Settings is omitted."""
    capture = AudioCapture()

    assert capture.sample_rate == 16000
    assert capture.channels == 1
    assert capture.push_to_talk is True
    assert capture.device is None


def test_set_callback_stores_callback() -> None:
    """set_callback should store the provided callable."""
    capture = AudioCapture()
    callback = MagicMock()

    capture.set_callback(callback)

    assert capture._callback is callback


def test_on_audio_returns_when_not_recording() -> None:
    """_on_audio should do nothing while not recording."""
    capture = AudioCapture()
    capture._recording = False
    callback = MagicMock()
    capture.set_callback(callback)

    capture._on_audio(
        np.ones(100, dtype=np.float32),
        100,
        None,
        sd.CallbackFlags(),
    )

    callback.assert_not_called()


def test_on_audio_converts_stereo_to_mono() -> None:
    """_on_audio should average stereo channels into mono float32."""
    capture = AudioCapture(channels=2)
    capture._recording = True
    callback = MagicMock()
    capture.set_callback(callback)

    stereo = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    capture._on_audio(stereo, stereo.shape[0], None, sd.CallbackFlags())

    callback.assert_called_once()
    chunk = callback.call_args[0][0]
    expected = np.array([1.5, 3.5, 5.5], dtype=np.float32)
    assert np.array_equal(chunk, expected)
    assert chunk.dtype == np.float32


def test_on_audio_forwards_1d_float32() -> None:
    """_on_audio should forward one-dimensional float32 audio unchanged."""
    capture = AudioCapture()
    capture._recording = True
    callback = MagicMock()
    capture.set_callback(callback)

    audio = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    capture._on_audio(audio, audio.size, None, sd.CallbackFlags())

    callback.assert_called_once()
    chunk = callback.call_args[0][0]
    assert np.array_equal(chunk, audio)
    assert chunk.dtype == np.float32


def test_on_audio_logs_status_flags(caplog: pytest.LogCaptureFixture) -> None:
    """_on_audio should log a warning when status flags are present."""
    capture = AudioCapture()
    capture._recording = True
    callback = MagicMock()
    capture.set_callback(callback)

    status = sd.CallbackFlags()
    status.input_overflow = True
    capture._on_audio(np.zeros(10, dtype=np.float32), 10, None, status)

    callback.assert_called_once()


def test_on_audio_catches_callback_exception() -> None:
    """_on_audio should swallow exceptions raised by the callback."""
    capture = AudioCapture()
    capture._recording = True
    capture.set_callback(MagicMock(side_effect=RuntimeError("boom")))

    capture._on_audio(np.zeros(10, dtype=np.float32), 10, None, sd.CallbackFlags())


def test_start_push_to_talk_opens_stream() -> None:
    """start() in push-to-talk mode should open the input stream."""
    capture = AudioCapture(push_to_talk=True)
    mock_stream = MagicMock()

    with patch("src.audio.capture.sd.InputStream", return_value=mock_stream) as mock_input:
        capture.start()

    assert capture.is_recording() is True
    mock_input.assert_called_once()
    mock_stream.start.assert_called_once()


def test_start_push_to_talk_repeated_is_noop() -> None:
    """start() in push-to-talk mode should be a no-op while already recording."""
    capture = AudioCapture(push_to_talk=True)
    mock_stream = MagicMock()

    with patch("src.audio.capture.sd.InputStream", return_value=mock_stream):
        capture.start()
        capture.start()

    assert capture.is_recording() is True
    assert capture._stream is mock_stream


def test_start_toggle_opens_and_closes_stream() -> None:
    """start() in toggle mode should open and close the stream on successive calls."""
    capture = AudioCapture(push_to_talk=False)
    first_stream = MagicMock()
    second_stream = MagicMock()

    with patch("src.audio.capture.sd.InputStream", side_effect=[first_stream, second_stream]):
        capture.start()
        assert capture.is_recording() is True
        capture.start()
        assert capture.is_recording() is False

    first_stream.stop.assert_called_once()
    first_stream.close.assert_called_once()


def test_stop_closes_stream_and_resets_flag() -> None:
    """stop() should close the stream and clear the recording flag."""
    capture = AudioCapture(push_to_talk=True)
    mock_stream = MagicMock()

    with patch("src.audio.capture.sd.InputStream", return_value=mock_stream):
        capture.start()
        capture.stop()

    assert capture.is_recording() is False
    assert capture._stream is None
    mock_stream.stop.assert_called_once()
    mock_stream.close.assert_called_once()


def test_open_stream_invokes_sounddevice() -> None:
    """_open_stream should create an sd.InputStream with the configured parameters."""
    capture = AudioCapture(
        sample_rate=22050,
        channels=2,
        device=3,
        block_size=512,
    )
    mock_stream = MagicMock()

    with patch("src.audio.capture.sd.InputStream", return_value=mock_stream) as mock_input:
        capture._open_stream()

    mock_input.assert_called_once_with(
        samplerate=22050,
        channels=2,
        device=3,
        blocksize=512,
        dtype=np.float32,
        callback=capture._on_audio,
    )
    mock_stream.start.assert_called_once()


def test_close_stream_handles_none() -> None:
    """_close_stream should be safe when no stream is open."""
    capture = AudioCapture()
    capture._close_stream()
    assert capture._stream is None


def test_open_stream_error_handling() -> None:
    """_open_stream should log and re-raise when sd.InputStream fails."""
    capture = AudioCapture()

    with (
        patch("src.audio.capture.sd.InputStream", side_effect=RuntimeError("device failed")),
        pytest.raises(RuntimeError, match="device failed"),
    ):
        capture._open_stream()

    assert capture._stream is None
