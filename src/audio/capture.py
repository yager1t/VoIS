"""Microphone audio capture using ``sounddevice``."""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

import numpy as np
import sounddevice as sd
from loguru import logger

from src.config import Settings

AudioCallback = Callable[[np.ndarray], None]


class AudioCapture:
    """Push-to-talk / toggle microphone capture wrapper.

    Runs a ``sounddevice.InputStream`` and forwards each captured chunk to a
    registered callback. Recording can be controlled explicitly (push-to-talk)
    or toggled via :meth:`start` depending on ``push_to_talk`` mode.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        sample_rate: int | None = None,
        channels: int | None = None,
        device: int | str | None = None,
        push_to_talk: bool | None = None,
        block_size: int = 1024,
    ) -> None:
        """Create a capture instance from a Settings object or explicit values.

        Args:
            settings: Optional application settings. Explicit keyword arguments
                override values from ``settings``.
            sample_rate: Capture sample rate in Hz. Defaults to 16000.
            channels: Number of channels. Defaults to 1 (mono).
            device: ``sounddevice`` input device identifier.
            push_to_talk: If ``True``, :meth:`start` and :meth:`stop` are explicit.
                If ``False``, :meth:`start` toggles recording on/off.
            block_size: Number of frames per callback block.
        """
        if settings is not None:
            self.sample_rate: int = sample_rate or settings.audio_sample_rate
            self.channels: int = channels or settings.audio_channels
            self.push_to_talk: bool = (
                push_to_talk if push_to_talk is not None else settings.push_to_talk
            )
        else:
            self.sample_rate = sample_rate if sample_rate is not None else 16000
            self.channels = channels if channels is not None else 1
            self.push_to_talk = push_to_talk if push_to_talk is not None else True

        self.device = device
        self.block_size = block_size

        self._callback: AudioCallback | None = None
        self._stream: sd.InputStream | None = None
        self._recording = False
        self._lock = threading.Lock()

    def set_callback(self, callback: AudioCallback) -> None:
        """Register a function called for each captured chunk.

        Args:
            callback: Function receiving a one-dimensional ``float32`` array.
        """
        self._callback = callback

    def _on_audio(
        self,
        indata: np.ndarray,
        _frames: int,
        _time: Any,
        status: sd.CallbackFlags,
    ) -> None:
        """Internal ``sounddevice`` callback.

        Args:
            indata: Raw input frames from PortAudio.
            _frames: Frame count (unused).
            _time: Stream timing info (unused).
            status: PortAudio status flags.
        """
        if status:
            logger.warning("Audio stream status: {}", status)

        if not self._recording:
            return

        # Convert interleaved/multi-channel data to mono float32.
        if indata.ndim == 1:
            chunk = indata.astype(np.float32, copy=False)
        else:
            chunk = np.asarray(np.mean(indata, axis=1, dtype=np.float32), dtype=np.float32)

        if self._callback is not None:
            try:
                self._callback(chunk)
            except Exception:  # pragma: no cover - defensive
                logger.exception("Audio callback failed")

    def start(self) -> None:
        """Start or toggle recording.

        In push-to-talk mode this opens the input stream if not already recording.
        In toggle mode this flips the recording state, opening or closing the stream.
        """
        with self._lock:
            if self.push_to_talk:
                if self._recording:
                    logger.debug("Already recording (push-to-talk)")
                    return
                self._open_stream()
                self._recording = True
                logger.info("Recording started (push-to-talk)")
            else:
                if self._recording:
                    self._close_stream()
                    self._recording = False
                    logger.info("Recording stopped (toggle)")
                else:
                    self._open_stream()
                    self._recording = True
                    logger.info("Recording started (toggle)")

    def stop(self) -> None:
        """Stop recording and release the audio stream."""
        with self._lock:
            if not self._recording:
                return
            self._close_stream()
            self._recording = False
            logger.info("Recording stopped")

    def is_recording(self) -> bool:
        """Return whether the capture is currently active."""
        with self._lock:
            return self._recording

    def _open_stream(self) -> None:
        """Open the ``sounddevice`` input stream."""
        try:
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                device=self.device,
                blocksize=self.block_size,
                dtype=np.float32,
                callback=self._on_audio,
            )
            self._stream.start()
        except Exception as exc:
            logger.exception("Failed to open audio input stream: {}", exc)
            self._stream = None
            raise

    def _close_stream(self) -> None:
        """Close and release the ``sounddevice`` input stream."""
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Error closing audio stream: {}", exc)
            finally:
                self._stream = None
