"""CLI entry point for the Voice-to-Cursor application."""

import argparse
import os
import sys

from src.app import App
from src.config import Settings
from src.logging_config import configure_logging, logger


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Optional argument list for testing.

    Returns:
        Parsed namespace.
    """
    parser = argparse.ArgumentParser(
        prog="voice-to-cursor",
        description="Press a global hotkey, speak, and insert text at the cursor.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to an optional .env configuration file.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Whisper model size (tiny/base/small/medium/large).",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="ASR language code, or 'auto' for language detection.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda"],
        help="Device to run the ASR model on.",
    )
    parser.add_argument(
        "--hotkey",
        type=str,
        default=None,
        help="Global hotkey override, e.g. 'f9' or '<ctrl>+f9'.",
    )
    parser.add_argument(
        "--toggle",
        action="store_true",
        help="Disable push-to-talk; each hotkey press toggles recording.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Transcribe and print text instead of injecting it.",
    )
    return parser.parse_args(argv)


def apply_cli_overrides(settings: Settings, args: argparse.Namespace) -> Settings:
    """Apply CLI overrides on top of loaded settings.

    Args:
        settings: Settings loaded from environment and optional .env file.
        args: Parsed CLI arguments.

    Returns:
        The same ``Settings`` instance with overrides applied.
    """
    if args.model is not None:
        settings.asr_model = args.model
    if args.language is not None:
        settings.asr_language = args.language
    if args.device is not None:
        settings.asr_device = args.device
    if args.hotkey is not None:
        settings.hotkey = args.hotkey
    if args.toggle:
        settings.push_to_talk = False
    if args.dry_run:
        settings.dry_run = True
    return settings


def main(argv: list[str] | None = None) -> int:
    """Application entry point.

    Args:
        argv: Optional argument list.

    Returns:
        Exit code.
    """
    args = parse_args(argv)
    env_file = args.config or ".env"

    if args.config:
        os.environ.setdefault("VOICE_CONFIG_PATH", args.config)

    settings = Settings(_env_file=env_file)  # type: ignore[call-arg]
    settings = apply_cli_overrides(settings, args)
    os.environ.setdefault("VOICE_DATA_DIR", str(settings.data_dir))

    configure_logging()
    logger.info("Configuration loaded: {}", settings.model_dump(exclude={"llm_url"}))

    app = App(settings)
    try:
        app.start()
    except Exception as exc:  # pragma: no cover - defensive catch-all
        logger.exception("Unhandled error: {}", exc)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
