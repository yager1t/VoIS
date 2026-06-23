"""Application lifecycle skeleton."""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

from src.asr.base import TranscriptionResult
from src.asr.streaming import StreamingTranscriber
from src.audio.buffer import AudioBuffer
from src.audio.capture import AudioCapture
from src.audio.vad import WebRTCVADProvider
from src.config import Settings
from src.dictionary import TextCorrector, VocabularyLearner, VocabularyManager
from src.dictionary.context_modes import parse_context_mode
from src.hotkey import create_hotkey_manager
from src.injection import WindowsTextInjector, create_text_injector
from src.postprocess import create_post_processor

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
        self.recording_started: Callable[[], None] | None = None
        self.recording_stopped: Callable[[], None] | None = None
        self.text_injected: Callable[[str], None] | None = None
        self.capture = AudioCapture(settings)
        self.buffer = AudioBuffer(
            max_seconds=settings.audio_max_record_seconds,
            sample_rate=settings.audio_sample_rate,
        )
        self.capture.set_callback(self._on_audio_chunk)
        self.streaming_transcriber: StreamingTranscriber | None = None
        self._asr_warmed_up = False
        self.injector: TextInjector = create_text_injector()
        if isinstance(self.injector, WindowsTextInjector):
            self.injector.fallback_to_clipboard = settings.injection_fallback_to_clipboard
        self.post_processor = create_post_processor(settings)
        self.dictionary = VocabularyManager(settings)
        self.dictionary.load_all()
        self.corrector = TextCorrector(self.dictionary)
        self.learner = VocabularyLearner(self.dictionary, settings)
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

    def _on_audio_chunk(self, chunk: np.ndarray) -> None:
        """Route an incoming audio chunk to the buffer and optional streamer."""
        self.buffer.append(chunk)
        if self.settings.streaming_enabled and self.streaming_transcriber is not None:
            self.streaming_transcriber.add_audio(chunk)

    def _concatenate_streaming_results(self, results: list[TranscriptionResult]) -> str:
        """Join final streaming results into a single transcript."""
        texts = [
            result.text.strip()
            for result in results
            if result.is_final and result.text.strip()
        ]
        return " ".join(texts)

    def _warmup_asr(self) -> None:
        """Load the ASR model in the background so first transcription is fast."""
        self._asr_warmed_up = True
        try:
            self.asr.warmup()
        except Exception:
            logger.exception("ASR warmup failed")

    def start_recording(self) -> None:
        """Start audio recording when the hotkey is pressed."""
        logger.info("Start recording requested")
        self.buffer.clear()

        if self.settings.streaming_enabled:
            if self.streaming_transcriber is not None:
                self.streaming_transcriber.stop()
            self.streaming_transcriber = StreamingTranscriber(
                self.settings,
                self.asr,
                self.vad,
            )
            self.streaming_transcriber.start()

        if self.settings.asr_warmup_at_start and not self._asr_warmed_up:
            threading.Thread(target=self._warmup_asr, daemon=True).start()

        self.capture.start()
        self._invoke_callback("recording_started")

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

        text = ""
        if audio.size > 0:
            prepared = self._prepare_audio(audio)

            if self.settings.streaming_enabled and self.streaming_transcriber is not None:
                self.streaming_transcriber.stop()
                results = self.streaming_transcriber.get_results()
                streaming_text = self._concatenate_streaming_results(results)
                self.streaming_transcriber = None

                if streaming_text:
                    text = streaming_text
                elif prepared.size > 0:
                    text = self.transcribe_audio(prepared)
            elif prepared.size > 0:
                text = self.transcribe_audio(prepared)

            if text:
                if self.settings.dictionary_enabled:
                    context_mode = parse_context_mode(self.settings.context_mode)
                    text = self.corrector.correct(text, context_mode)
                    logger.info("Corrected text: {}", text)
                text = self.post_processor.process(text)
                logger.info("Post-processed text: {}", text)
                self.inject_text(text)
            else:
                logger.info("No speech detected after VAD trimming; skipping")
        else:
            logger.info("No audio captured; nothing to inject")

        self._invoke_callback("recording_stopped")
        if text:
            self._invoke_callback("text_injected", text)
            if self.settings.dictionary_learning_enabled:
                self.learner.learn_from_text(text)

    def _invoke_callback(self, name: str, *args: object) -> None:
        """Invoke a callback attribute if it has been set.

        Args:
            name: Callback attribute name.
            *args: Arguments to pass to the callback.
        """
        callback = getattr(self, name, None)
        if callback is not None:
            callback(*args)

    def is_running(self) -> bool:
        """Return whether the service is currently running."""
        return self._running

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

    def record_correction(self, original: str, corrected: str) -> None:
        """Record a user correction for adaptive learning.

        Args:
            original: The original (incorrect) text.
            corrected: The corrected text.
        """
        self.learner.record_correction(original, corrected)

    def transcribe_audio(self, audio: np.ndarray) -> str:
        """Transcribe audio and return the recognized text.

        Args:
            audio: One-dimensional ``float32`` audio samples.

        Returns:
            Transcribed text (empty string when no speech is recognized).
        """
        result = self.asr.transcribe(audio, self.settings.audio_sample_rate)
        return result.text
