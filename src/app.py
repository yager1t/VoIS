"""Application lifecycle skeleton."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

from src.audio.buffer import AudioBuffer
from src.audio.capture import AudioCapture
from src.config import Settings
from src.hotkey import create_hotkey_manager
from src.injection import WindowsTextInjector, create_text_injector

if TYPE_CHECKING:  # pragma: no cover
    from src.asr.base import ASRProvider
    from src.injection.base import TextInjector


class App:
    """Main application orchestrator."""

    def __init__(self, settings: Settings) -> None:
        """Initialize the application with validated settings.

        Args:
            settings: Parsed application configuration.
        """
        self.settings = settings
        self._running = False
        self.capture = AudioCapture(settings)
        self.buffer = AudioBuffer()
        self.capture.set_callback(self.buffer.append)
        self.injector: TextInjector = create_text_injector()
        if isinstance(self.injector, WindowsTextInjector):
            self.injector.fallback_to_clipboard = settings.injection_fallback_to_clipboard
        self._asr: ASRProvider | None = None
        self.hotkey = create_hotkey_manager(
            settings.hotkey,
            push_to_talk=settings.push_to_talk,
            on_press=self.start_recording,
            on_release=self.stop_recording,
        )
        logger.debug("AudioCapture and HotkeyManager initialized")

    @property
    def asr(self) -> ASRProvider:
        """Lazy-load and return the configured ASR provider."""
        if self._asr is None:
            from src.asr.whisper_provider import FasterWhisperProvider

            self._asr = FasterWhisperProvider(self.settings)
            logger.debug("ASR provider initialized")
        return self._asr

    def start(self) -> None:
        """Start the voice-to-cursor service."""
        self.settings.ensure_dirs()
        self._running = True
        self.hotkey.start()
        logger.info("Voice-to-Cursor started (hotkey: {})", self.settings.hotkey)
        try:
            while self._running:
                # Main thread idle loop; hotkey events run on a daemon thread.
                pass  # pragma: no cover
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt received")
        finally:
            self.stop()

    def stop(self) -> None:
        """Stop the service and release resources."""
        if not self._running:
            return
        self._running = False
        self.hotkey.stop()
        self.capture.stop()
        logger.info("Voice-to-Cursor stopped")

    def start_recording(self) -> None:
        """Start audio recording when the hotkey is pressed."""
        logger.info("Start recording requested")
        self.buffer.clear()
        self.capture.start()

    def stop_recording(self) -> None:
        """Stop audio recording, transcribe, and inject the result."""
        logger.info("Stop recording requested")
        self.capture.stop()

        audio = self.buffer.get()
        self.buffer.clear()
        if audio.size == 0:
            logger.info("No audio captured; nothing to inject")
            return

        text = self.transcribe_audio(audio)
        logger.info("Transcribed text to inject: {}", text)
        self.inject_text(text)

    def inject_text(self, text: str) -> None:
        """Inject the given text at the current cursor position.

        Args:
            text: Text to type or paste.
        """
        if not text:
            return
        self.injector.inject_with_delay(text, self.settings.injection_delay_ms)
        logger.info("Injected {} character(s)", len(text))

    def transcribe_audio(self, audio: np.ndarray) -> str:
        """Transcribe audio and return the recognized text.

        Args:
            audio: One-dimensional ``float32`` audio samples.

        Returns:
            Transcribed text (empty string when no speech is recognized).
        """
        result = self.asr.transcribe(audio, self.settings.audio_sample_rate)
        return result.text
