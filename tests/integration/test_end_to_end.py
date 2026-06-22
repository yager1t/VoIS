"""End-to-end integration tests for the Voice-to-Cursor flow."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.app import App
from src.config import Settings


@pytest.fixture
def app(settings: Settings) -> App:
    """Return an App wired with fully mocked platform dependencies."""
    event_mock = MagicMock()
    event_mock.is_set.return_value = True

    with (
        patch("src.app.WebRTCVADProvider") as mock_vad_cls,
        patch("src.app.threading.Event", return_value=event_mock),
        patch("src.app.create_hotkey_manager") as mock_hotkey_factory,
        patch("src.app.create_text_injector") as mock_injector_factory,
        patch("src.app.AudioCapture") as mock_capture_cls,
        patch("src.app.AudioBuffer") as mock_buffer_cls,
    ):
        vad_instance = MagicMock()
        mock_vad_cls.return_value = vad_instance

        hotkey_instance = MagicMock()
        mock_hotkey_factory.return_value = hotkey_instance

        injector_instance = MagicMock()
        mock_injector_factory.return_value = injector_instance

        capture_instance = MagicMock()
        mock_capture_cls.return_value = capture_instance

        buffer_instance = MagicMock()
        mock_buffer_cls.return_value = buffer_instance

        app = App(settings)
        # Expose mocks for assertions in tests.
        app._vad_mock = vad_instance
        app._hotkey_mock = hotkey_instance
        app._injector_mock = injector_instance
        app._capture_mock = capture_instance
        app._buffer_mock = buffer_instance
        yield app


@pytest.mark.integration
def test_push_to_talk_flow_injects_text(app: App) -> None:
    """Press -> capture -> release -> transcribe -> inject should emit text."""
    audio = np.ones(16000, dtype=np.float32)
    app._buffer_mock.get.return_value = audio
    app._vad_mock.split_on_silence.return_value = [audio]

    asr_mock = MagicMock()
    asr_mock.transcribe.return_value.text = "push to talk text"
    app._asr = asr_mock

    app.start_recording()
    app.stop_recording()

    app._capture_mock.start.assert_called_once()
    app._capture_mock.stop.assert_called_once()
    asr_mock.transcribe.assert_called_once()
    app._injector_mock.inject_with_delay.assert_called_once_with(
        "Push to talk text.",
        app.settings.injection_delay_ms,
    )


@pytest.mark.integration
def test_toggle_mode_flow_injects_text(app: App) -> None:
    """Toggle mode: first press starts, second press stops and injects text."""
    audio = np.ones(16000, dtype=np.float32)
    app._capture_mock.is_recording.side_effect = [False, True]
    app._buffer_mock.get.return_value = audio
    app._vad_mock.split_on_silence.return_value = [audio]

    asr_mock = MagicMock()
    asr_mock.transcribe.return_value.text = "toggle text"
    app._asr = asr_mock

    app.toggle_recording()
    app.toggle_recording()

    app._capture_mock.start.assert_called_once()
    app._capture_mock.stop.assert_called_once()
    asr_mock.transcribe.assert_called_once()
    app._injector_mock.inject_with_delay.assert_called_once_with(
        "Toggle text.",
        app.settings.injection_delay_ms,
    )


@pytest.mark.integration
def test_dry_run_does_not_inject(app: App) -> None:
    """With dry_run=True the full flow should log but never call the injector."""
    app.settings.dry_run = True
    audio = np.ones(16000, dtype=np.float32)
    app._buffer_mock.get.return_value = audio
    app._vad_mock.split_on_silence.return_value = [audio]

    asr_mock = MagicMock()
    asr_mock.transcribe.return_value.text = "dry run text"
    app._asr = asr_mock

    app.start_recording()
    app.stop_recording()

    asr_mock.transcribe.assert_called_once()
    app._injector_mock.inject_with_delay.assert_not_called()


@pytest.mark.integration
def test_no_speech_after_vad_skips_injection(app: App) -> None:
    """If VAD finds no speech after capture, ASR and injection should be skipped."""
    audio = np.ones(16000, dtype=np.float32)
    app._buffer_mock.get.return_value = audio
    app._vad_mock.split_on_silence.return_value = []

    asr_mock = MagicMock()
    app._asr = asr_mock

    app.start_recording()
    app.stop_recording()

    asr_mock.transcribe.assert_not_called()
    app._injector_mock.inject_with_delay.assert_not_called()
