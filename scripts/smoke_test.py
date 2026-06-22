"""Smoke test: run the Voice-to-Cursor pipeline for a limited time.

.. deprecated::
    This manual script is kept for ad-hoc interactive smoke testing. The
    automated smoke test harness lives in ``tests/smoke/test_smoke.py`` and is
    run via ``scripts/run_smoke.sh`` or ``scripts/run_smoke.bat``.

This script starts the full application in dry-run mode by default so it
prints transcribed text instead of injecting it into the active window.

Usage:
    python scripts/smoke_test.py
    python scripts/smoke_test.py --duration 60
    python scripts/smoke_test.py --model tiny --language en

Press the configured hotkey to record speech. The script stops automatically
after ``--duration`` seconds or when you press Ctrl+C.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from threading import Thread

# Allow the script to be executed directly from the scripts/ directory.
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.app import App  # noqa: E402
from src.config import Settings  # noqa: E402
from src.logging_config import configure_logging, logger  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse smoke-test CLI arguments."""
    parser = argparse.ArgumentParser(
        prog="smoke-test",
        description="Run the Voice-to-Cursor pipeline for a short time.",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=30,
        help="Number of seconds to run the smoke test (default: 30).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Whisper model size override.",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="ASR language override.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda"],
        help="ASR device override.",
    )
    parser.add_argument(
        "--hotkey",
        type=str,
        default=None,
        help="Global hotkey override.",
    )
    parser.add_argument(
        "--no-dry-run",
        action="store_true",
        help="Actually inject text at the cursor instead of printing it.",
    )
    return parser.parse_args(argv)


def apply_overrides(settings: Settings, args: argparse.Namespace) -> Settings:
    """Apply smoke-test overrides to settings."""
    if args.model is not None:
        settings.asr_model = args.model
    if args.language is not None:
        settings.asr_language = args.language
    if args.device is not None:
        settings.asr_device = args.device
    if args.hotkey is not None:
        settings.hotkey = args.hotkey
    if not args.no_dry_run:
        settings.dry_run = True
    return settings


def print_instructions(args: argparse.Namespace, settings: Settings) -> None:
    """Show the user how to interact with the smoke test."""
    if settings.dry_run:
        mode = "DRY-RUN (text is printed, not injected)"
    else:
        mode = "LIVE (text will be injected)"
    print("\n" + "=" * 60)
    print("Voice-to-Cursor Smoke Test")
    print("=" * 60)
    print(f"Mode:        {mode}")
    print(f"Hotkey:      {settings.hotkey}")
    print(f"Model:       {settings.asr_model}")
    print(f"Language:    {settings.asr_language}")
    print(f"Device:      {settings.asr_device}")
    print(f"Duration:    {args.duration} seconds")
    print("-" * 60)
    print("Instructions:")
    if settings.push_to_talk:
        print(f"  Hold {settings.hotkey}, speak, then release it.")
    else:
        print(f"  Press {settings.hotkey} once to start, press again to stop.")
    print("  Say a few words and watch the transcribed output.")
    print("  Press Ctrl+C to stop early.")
    print("=" * 60 + "\n")


def main(argv: list[str] | None = None) -> int:
    """Run the smoke test."""
    args = parse_args(argv)

    configure_logging()
    settings = Settings()
    settings = apply_overrides(settings, args)

    print_instructions(args, settings)

    app = App(settings)

    def auto_stop() -> None:
        """Stop the app after the configured duration."""
        time.sleep(args.duration)
        if app._running:
            logger.info("Smoke test duration reached; stopping")
            app.stop()

    app_thread = Thread(target=app.start, name="app-main", daemon=True)
    app_thread.start()

    stop_thread = Thread(target=auto_stop, name="auto-stop", daemon=True)
    stop_thread.start()

    try:
        while app_thread.is_alive():
            app_thread.join(timeout=0.5)
    except KeyboardInterrupt:
        print("\nStopping smoke test...")
        app.stop()
        app_thread.join(timeout=5.0)

    print("\nSmoke test finished.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
