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
    return parser.parse_args(argv)


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
