#!/usr/bin/env python3
"""Command-line latency benchmark for streaming vs batch ASR.

Run without a real model (default) to exercise the harness in CI::

    python benchmarks/run_latency.py --mode both --duration 2.0

Run against a real faster-whisper model (slow, requires download)::

    python benchmarks/run_latency.py --mode both --real-model --model base
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Ensure the project root is on sys.path so ``src`` and ``benchmarks`` can be
# imported regardless of where the script is invoked from.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.benchmarks.latency import LatencyBenchmark, MockASRProvider  # noqa: E402
from src.config import Settings  # noqa: E402


def generate_silence(duration: float, sample_rate: int) -> np.ndarray:
    """Generate ``duration`` seconds of silent float32 audio."""
    return np.zeros(int(duration * sample_rate), dtype=np.float32)


def load_audio(wav_path: Path, sample_rate: int) -> np.ndarray:
    """Load a mono WAV file and validate the sample rate.

    Args:
        wav_path: Path to the WAV file to load.
        sample_rate: Expected sample rate in Hz.

    Returns:
        One-dimensional ``float32`` audio samples.

    Raises:
        ValueError: If the file's sample rate does not match ``sample_rate``.
    """
    import soundfile as sf

    audio, file_rate = sf.read(str(wav_path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if file_rate != sample_rate:
        raise ValueError(
            f"Sample rate mismatch: {wav_path} is {file_rate} Hz, expected {sample_rate} Hz",
        )
    return np.asarray(audio, dtype=np.float32)


def load_or_generate_audio(
    wav_path: Path | None,
    duration: float,
    sample_rate: int,
) -> np.ndarray:
    """Return audio from ``wav_path`` or generate silence of ``duration``."""
    if wav_path is not None:
        return load_audio(wav_path, sample_rate)
    return generate_silence(duration, sample_rate)


def create_asr_provider(settings: Settings, real_model: bool) -> MockASRProvider:  # type: ignore[return]
    """Return an ASR provider, either mock or real.

    Args:
        settings: Parsed application settings.
        real_model: When ``True``, load a real faster-whisper model.

    Returns:
        An instance of :class:`MockASRProvider` or
        :class:`src.asr.whisper_provider.FasterWhisperProvider`.
    """
    if real_model:
        from src.asr.whisper_provider import FasterWhisperProvider

        provider = FasterWhisperProvider(settings)
        provider.load_model()
        return provider
    return MockASRProvider(settings)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the benchmark runner."""
    parser = argparse.ArgumentParser(
        description="Benchmark streaming vs batch ASR latency.",
    )
    parser.add_argument(
        "--mode",
        choices=["streaming", "batch", "both"],
        default="both",
        help="Benchmark mode (default: both).",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=2.0,
        help="Duration in seconds of synthetic silence when no --wav is given.",
    )
    parser.add_argument(
        "--chunk-seconds",
        type=float,
        default=1.0,
        help="Streaming chunk duration in seconds.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Audio sample rate in Hz.",
    )
    parser.add_argument(
        "--wav",
        type=Path,
        default=None,
        help="Path to a mono WAV file to transcribe instead of synthetic audio.",
    )
    parser.add_argument(
        "--real-model",
        action="store_true",
        help="Use a real faster-whisper model instead of the mock provider.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="base",
        help="Whisper model size when --real-model is used.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for the real model.",
    )
    parser.add_argument(
        "--compute-type",
        type=str,
        default="int8",
        help="Compute type for the real model.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the benchmark and print a JSON summary.

    Args:
        argv: Command-line arguments. When ``None``, ``sys.argv`` is used.

    Returns:
        Exit code (``0`` on success).
    """
    args = parse_args(argv)

    settings = Settings(
        audio_sample_rate=args.sample_rate,
        asr_model=args.model,
        asr_device=args.device,
        asr_compute_type=args.compute_type,
        streaming_chunk_seconds=args.chunk_seconds,
    )

    audio = load_or_generate_audio(args.wav, args.duration, args.sample_rate)
    chunk_size = int(args.chunk_seconds * args.sample_rate)
    asr_provider = create_asr_provider(settings, args.real_model)

    modes: list[str] = []
    if args.mode in ("streaming", "both"):
        modes.append("streaming")
    if args.mode in ("batch", "both"):
        modes.append("batch")

    results: dict[str, dict[str, object]] = {}
    for mode in modes:
        benchmark = LatencyBenchmark(
            settings,
            asr_provider,
            audio,
            mode,
            chunk_size=chunk_size,
        )
        results[mode] = benchmark.run().summary()

    summary = {
        "settings": {
            "mode": args.mode,
            "real_model": args.real_model,
            "duration": args.duration,
            "chunk_seconds": args.chunk_seconds,
            "sample_rate": args.sample_rate,
        },
        "results": results,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
