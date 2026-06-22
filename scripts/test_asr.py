"""Optional smoke test: record audio, run VAD, and transcribe with Whisper.

This script is meant to be run manually on a developer workstation. It records
audio from the default microphone, trims silence with WebRTC VAD, and runs the
configured faster-whisper model. The first run downloads the model if it is not
cached locally under ``models/``.

Example:
    python scripts/test_asr.py --seconds 3 --model base
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from src.asr.whisper_provider import FasterWhisperProvider
from src.audio.buffer import AudioBuffer
from src.audio.capture import AudioCapture
from src.audio.vad import WebRTCVADProvider
from src.config import Settings


SAMPLE_RATE = 16000


def record_audio(seconds: float, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
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


def trim_silence(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Drop silent frames using WebRTC VAD.

    Args:
        audio: Recorded audio samples.
        sample_rate: Capture sample rate.

    Returns:
        Audio with leading/trailing silence removed, or the original audio when
        no speech is detected.
    """
    provider = WebRTCVADProvider(aggressiveness=2)
    segments = provider.split_on_silence(audio, sample_rate, frame_ms=30)
    if not segments:
        return audio
    return np.concatenate(segments, dtype=np.float32)


def main(argv: list[str] | None = None) -> int:
    """Run the ASR smoke test.

    Args:
        argv: Optional argument list.

    Returns:
        Exit code.
    """
    parser = argparse.ArgumentParser(
        description="Record audio and transcribe it with faster-whisper.",
    )
    parser.add_argument("--seconds", type=float, default=3.0, help="Recording duration.")
    parser.add_argument("--model", type=str, default="base", help="Whisper model name.")
    parser.add_argument("--output", type=Path, default=Path("data/debug_asr.wav"))
    args = parser.parse_args(argv)

    audio = record_audio(args.seconds, sample_rate=SAMPLE_RATE)
    trimmed = trim_silence(audio, SAMPLE_RATE)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.output, trimmed, SAMPLE_RATE)

    print(f"Saved {len(trimmed)} samples ({len(trimmed) / SAMPLE_RATE:.2f} s) to {args.output}")

    settings = Settings(asr_model=args.model, asr_language="auto")
    provider = FasterWhisperProvider(settings)
    print("Loading model (this may download on first run)...")
    result = provider.transcribe(trimmed, SAMPLE_RATE)

    print(f"Transcription: {result.text!r}")
    print(f"Language: {result.language}")
    if result.confidence is not None:
        print(f"Confidence (avg logprob): {result.confidence:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
