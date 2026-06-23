"""Voice activity detection abstraction and WebRTC VAD implementation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    import webrtcvad
else:
    try:
        import webrtcvad
    except ImportError:  # pragma: no cover
        webrtcvad = None  # type: ignore[assignment]


class VADProvider(ABC):
    """Abstract interface for voice activity detection backends."""

    @abstractmethod
    def is_speech(self, audio: np.ndarray, sample_rate: int) -> bool:
        """Return whether the audio frame contains speech.

        Args:
            audio: One-dimensional ``float32`` audio samples.
            sample_rate: Sample rate in Hz (must be 8000, 16000, 32000, or 48000).

        Returns:
            ``True`` if the frame is classified as speech.
        """

    @abstractmethod
    def process_stream(
        self,
        chunks: list[np.ndarray],
        sample_rate: int,
    ) -> list[tuple[np.ndarray, bool]]:
        """Classify each chunk and return it paired with the speech decision.

        Args:
            chunks: List of one-dimensional audio arrays.
            sample_rate: Sample rate in Hz.

        Returns:
            List of ``(audio_chunk, is_speech)`` tuples.
        """

    def split_on_silence(
        self,
        audio: np.ndarray,
        sample_rate: int,
        *,
        frame_ms: int = 30,
        keep_chunks: int = 1,
    ) -> list[np.ndarray]:
        """Segment audio, dropping chunks classified as silence.

        Args:
            audio: One-dimensional audio samples.
            sample_rate: Sample rate in Hz.
            frame_ms: Frame duration for VAD in milliseconds (10, 20, or 30).
            keep_chunks: Number of consecutive silent frames to keep around speech.

        Returns:
            List of audio segments that contain speech (with small context kept).
        """
        frames = frame_audio(audio, sample_rate, frame_ms=frame_ms)
        decisions = [self.is_speech(frame, sample_rate) for frame in frames]

        segments: list[np.ndarray] = []
        current: list[np.ndarray] = []
        trailing_silence: list[np.ndarray] = []
        silence_streak = 0

        for frame, is_speech in zip(frames, decisions, strict=True):
            if is_speech:
                if not current and trailing_silence:
                    current.extend(trailing_silence[-keep_chunks:])
                current.append(frame)
                trailing_silence.clear()
                silence_streak = 0
            else:
                silence_streak += 1
                trailing_silence.append(frame)
                if len(trailing_silence) > keep_chunks:
                    trailing_silence = trailing_silence[-keep_chunks:]
                if current:
                    if silence_streak <= keep_chunks:
                        current.append(frame)
                    else:
                        segments.append(np.concatenate(current, dtype=np.float32))
                        current = []
                        trailing_silence = trailing_silence[-keep_chunks:]

        if current:
            segments.append(np.concatenate(current, dtype=np.float32))

        return segments


class WebRTCVADProvider(VADProvider):
    """WebRTC VAD backend wrapper.

    The underlying library requires int16 PCM data with specific frame durations
    (10, 20, or 30 ms) and sample rates of 8000, 16000, 32000, or 48000 Hz.
    """

    def __init__(self, aggressiveness: int = 2) -> None:
        """Initialize the WebRTC VAD.

        Args:
            aggressiveness: VAD aggressiveness from 0 (least) to 3 (most aggressive).

        Raises:
            RuntimeError: If ``webrtcvad`` is not installed.
        """
        if webrtcvad is None:
            raise RuntimeError("webrtcvad is not installed")
        self._vad = webrtcvad.Vad(aggressiveness)

    def is_speech(self, audio: np.ndarray, sample_rate: int) -> bool:
        """Classify a single audio frame as speech or not.

        Args:
            audio: One-dimensional ``float32`` audio samples.
            sample_rate: Sample rate in Hz.

        Returns:
            ``True`` if the frame contains speech.
        """
        pcm = float_to_int16(audio)
        return bool(self._vad.is_speech(pcm.tobytes(), sample_rate))

    def process_stream(
        self,
        chunks: list[np.ndarray],
        sample_rate: int,
    ) -> list[tuple[np.ndarray, bool]]:
        """Return each chunk paired with its speech decision."""
        return [(chunk, self.is_speech(chunk, sample_rate)) for chunk in chunks]


def float_to_int16(audio: np.ndarray) -> np.ndarray:
    """Convert ``float32`` audio in [-1.0, 1.0] to 16-bit PCM.

    Args:
        audio: One-dimensional audio samples.

    Returns:
        ``int16`` PCM samples clipped to the valid range.
    """
    samples = np.asarray(audio, dtype=np.float32)
    samples = np.clip(samples, -1.0, 1.0)
    return (samples * 32767).astype(np.int16)


def frame_audio(audio: np.ndarray, sample_rate: int, *, frame_ms: int = 30) -> list[np.ndarray]:
    """Split audio into fixed-duration frames suitable for WebRTC VAD.

    Args:
        audio: One-dimensional audio samples.
        sample_rate: Sample rate in Hz.
        frame_ms: Frame duration in milliseconds (10, 20, or 30).

    Returns:
        List of frame arrays. The trailing partial frame is dropped.
    """
    if frame_ms not in (10, 20, 30):
        raise ValueError("frame_ms must be 10, 20, or 30")

    frame_size = int(sample_rate * frame_ms / 1000)
    samples = np.asarray(audio, dtype=np.float32)
    total_frames = samples.shape[0] // frame_size
    return [samples[i * frame_size : (i + 1) * frame_size] for i in range(total_frames)]
