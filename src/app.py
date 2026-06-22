"""Application lifecycle skeleton."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

from src.audio.buffer import AudioBuffer
from src.audio.capture import AudioCapture
from src.audio.vad import WebRTCVADProvider
from src.config import Settings
from src.hotkey import create_hotkey_manager
from src.injection import WindowsTextInjector, create_text_injector

if TYPE_CHECKING:  # pragma: no cover
    from src.asr.base import ASRProvider
    from src.audio.vad import VADProvider
    from src.injection.base import TextInjector


def trim_silence(
    audio: np.ndarray,
    sample_rate: int,
    vad: VADProvider,
    *,
    trim_seconds: float = 0.3,
    frame_ms: int = 30,
) -> np.ndarray:
    """Trim leading and trailing silence using voice activity detection.

    Args:
        audio: One-dimensional ``float32`` audio samples.
        sample_rate: Sample rate in Hz.
        vad: Voice activity detection provider.
        trim_seconds: Minimum silence duration to trim around speech, in seconds.
        frame_ms: Frame duration used by the VAD in milliseconds.

    Returns:
        Trimmed audio, or an empty array if no speech is detected.
    """
    if audio.size == 0:
        return audio

    frame_seconds = frame_ms / 1000.0
    keep_chunks = max(1, int(round(trim_seconds / frame_seconds)))
    segments = vad.split_on_silence(
        audio,
        sample_rate,
        frame_ms=frame_ms,
        keep_chunks=keep_chunks,
    )
    if not segments:
        return np.array([], dtype=np.float32)
    return np.concatenate(segments, dtype=np.float32)


class App:
    """Main application orchestrator."""

    def __init__(self, settings: Settings) -> None:
        """Initialize the application with validated settings.

        Args:
            settings: Parsed application configuration.
        """
        self.settings = settings
        self._running = False
        self._shutdown_event = threading.Event()
        self.capture = AudioCapture(settings)
        self.buffer = AudioBuffer(
            max_seconds=settings.audio_max_record_seconds,
            sample_rate=settings.audio_sample_rate,
        )
        self.capture.set_callback(self.buffer.append)
        self.injector: TextInjector = create_text_injector()
        if isinstance(self.injector, WindowsTextInjector):
            self.injector.fallback_to_clipboard = settings.injection_fallback_to_clipboard
        self._asr: ASRProvider | None = None
        self.vad = self._create_vad()
        on_press = self.start_recording if settings.push_to_talk else self.toggle_recording
        on_release = self.stop_recording if settings.push_to_talk else None
        self.hotkey = create_hotkey_manager(
            settings.hotkey,
            push_to_talk=settings.push_to_talk,
            on_press=on_press,
            on_release=on_release,
        )
        logger.debug("AudioCapture, VAD, and HotkeyManager initialized")

    def _create_vad(self) -> VADProvider:
        """Create a VAD provider from settings.

        Returns:
            Configured VAD provider.
        """
        return WebRTCVADProvider(aggressiveness=self.settings.vad_aggressiveness)

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
        if self._running:
            logger.debug("Voice-to-Cursor already running")
            return

        self.settings.ensure_dirs()
        self._running = True
        self._shutdown_event.clear()
        self.hotkey.start()
        logger.info("Voice-to-Cursor started (hotkey: {})", self.settings.hotkey)
        try:
            while self._running:
                # Main thread idle loop; hotkey events run on a daemon thread.
                self._shutdown_event.wait(timeout=0.1)
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt received")
        finally:
            self.stop()

    def stop(self) -> None:
        """Stop the service and release resources."""
        if not self._running:
            return
        self._running = False
        self._shutdown_event.set()
        self.hotkey.stop()
        self.capture.stop()
        logger.info("Voice-to-Cursor stopped")

    def start_recording(self) -> None:
        """Start audio recording when the hotkey is pressed."""
        logger.info("Start recording requested")
        self.buffer.clear()
        self.capture.start()

    def toggle_recording(self) -> None:
        """Toggle recording mode and transcribe when a recording is stopped."""
        if self.capture.is_recording():
            self.stop_recording()
        else:
            self.start_recording()

    def stop_recording(self) -> None:
        """Stop audio recording, transcribe, and inject the result."""
        logger.info("Stop recording requested")
        self.capture.stop()

        audio = self.buffer.get()
        self.buffer.clear()
        if audio.size == 0:
            logger.info("No audio captured; nothing to inject")
            return

        prepared = self._prepare_audio(audio)
        if prepared.size == 0:
            logger.info("No speech detected after VAD trimming; skipping")
            return

        text = self.transcribe_audio(prepared)
        logger.info("Transcribed text to inject: {}", text)
        self.inject_text(text)

    def _prepare_audio(self, audio: np.ndarray) -> np.ndarray:
        """Trim silence and validate audio before transcription.

        Args:
            audio: One-dimensional ``float32`` audio samples.

        Returns:
            Trimmed audio ready for ASR, or an empty array if no speech is found.
        """
        return trim_silence(
            audio,
            self.settings.audio_sample_rate,
            self.vad,
            trim_seconds=self.settings.vad_trim_seconds,
        )

    def inject_text(self, text: str) -> None:
        """Inject the given text at the current cursor position.

        In dry-run mode the text is logged but not injected.

        Args:
            text: Text to type or paste.
        """
        if not text:
            return
        if self.settings.dry_run:
            logger.info("[dry-run] would inject: {}", text)
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
