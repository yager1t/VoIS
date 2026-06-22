"""Smoke test: record a few seconds of audio, save it, and report VAD ratio."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from src.audio.buffer import AudioBuffer
from src.audio.capture import AudioCapture
from src.audio.vad import WebRTCVADProvider, frame_audio


def record_audio(seconds: float, sample_rate: int = 16000) -> np.ndarray:
    """Record audio for a fixed duration using the default input device.

    Args:
        seconds: Duration to record.
        sample_rate: Capture sample rate.

    Returns:
        Concatenated mono ``float32`` audio samples.
    """
    buffer = AudioBuffer()
    capture = AudioCapture(sample_rate=sample_rate, channels=1, push_to_talk=False)
    capture.set_callback(buffer.append)

    print(f"Recording {seconds:.1f} seconds...")
    capture.start()
    time.sleep(seconds)
    capture.stop()

    return buffer.get()


def speech_ratio(audio: np.ndarray, sample_rate: int) -> float:
    """Return the fraction of VAD frames classified as speech.

    Args:
        audio: Recorded audio samples.
        sample_rate: Capture sample rate.

    Returns:
        Ratio between 0.0 and 1.0. Returns 0.0 when no frames are available.
    """
    provider = WebRTCVADProvider(aggressiveness=2)
    frames = frame_audio(audio, sample_rate, frame_ms=30)
    if not frames:
        return 0.0
    decisions = [provider.is_speech(frame, sample_rate) for frame in frames]
    return sum(decisions) / len(decisions)


def main(argv: list[str] | None = None) -> int:
    """Run the audio smoke test.

    Args:
        argv: Optional argument list.

    Returns:
        Exit code.
    """
    parser = argparse.ArgumentParser(description="Record audio and report VAD speech ratio.")
    parser.add_argument("--seconds", type=float, default=3.0, help="Recording duration.")
    parser.add_argument("--output", type=Path, default=Path("data/debug_record.wav"))
    args = parser.parse_args(argv)

    sample_rate = 16000
    audio = record_audio(args.seconds, sample_rate=sample_rate)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.output, audio, sample_rate)

    ratio = speech_ratio(audio, sample_rate)
    print(f"Saved {len(audio)} samples ({len(audio) / sample_rate:.2f} s) to {args.output}")
    print(f"VAD speech ratio: {ratio:.2%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
