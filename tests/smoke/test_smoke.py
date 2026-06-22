"""End-to-end smoke tests for the Voice-to-Cursor pipeline.

These tests are excluded from the default test run and require the
``--run-smoke`` pytest option. They exercise the full App orchestration
without real microphone hardware, global hotkeys, or text injection.
"""

from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.app import App
from src.config import Settings


def _synthetic_speech(sample_rate: int, duration: float = 0.5) -> np.ndarray:
    """Generate a sine wave simulating voiced audio."""
    t = np.linspace(0.0, duration, int(sample_rate * duration), endpoint=False)
    return (0.5 * np.sin(2.0 * np.pi * 400.0 * t)).astype(np.float32)


@pytest.mark.smoke
@pytest.mark.timeout(30)
def test_app_smoke_pipeline(tmp_path) -> None:
    """Run the App pipeline in dry-run mode with synthetic audio.

    The test starts ``App`` in a background thread, simulates a push-to-talk
    cycle by calling :meth:`start_recording` and :meth:`stop_recording`, and
    injects synthetic audio directly into the buffer. Platform-specific
    dependencies (hotkey manager, text injector, microphone capture, VAD, and
    ASR model) are replaced with stubs so the test remains safe to run on a
    developer machine.
    """
    settings = Settings(
        data_dir=tmp_path / "data",
        models_dir=tmp_path / "models",
        dry_run=True,
    )
    sample_rate = settings.audio_sample_rate
    synthetic_audio = _synthetic_speech(sample_rate)

    with (
        patch("src.app.create_hotkey_manager") as mock_hotkey_factory,
        patch("src.app.create_text_injector") as mock_injector_factory,
        patch("src.app.AudioCapture") as mock_capture_cls,
        patch("src.app.WebRTCVADProvider") as mock_vad_cls,
        patch("src.asr.whisper_provider.FasterWhisperProvider") as mock_asr_cls,
    ):
        mock_hotkey_factory.return_value = MagicMock()
        mock_injector_factory.return_value = MagicMock()
        mock_capture_cls.return_value = MagicMock()

        mock_vad = MagicMock()
        mock_vad.split_on_silence.return_value = [synthetic_audio]
        mock_vad_cls.return_value = mock_vad

        mock_asr = MagicMock()
        mock_asr.transcribe.return_value.text = "smoke test transcription"
        mock_asr_cls.return_value = mock_asr

        app = App(settings)
        app_thread = threading.Thread(target=app.start, name="smoke-app", daemon=True)
        app_thread.start()

        try:
            app.start_recording()
            app.buffer.append(synthetic_audio)
            app.stop_recording()
        finally:
            app.stop()
            app_thread.join(timeout=5.0)

    assert app._asr is not None
    mock_asr.transcribe.assert_called_once()
