"""CLI entry point for the Voice-to-Cursor application."""

from __future__ import annotations

import argparse
import os
import sys

from PyQt6.QtCore import QObject, QThread
from PyQt6.QtWidgets import QApplication

from src.app import App
from src.config import Settings
from src.logging_config import configure_logging, logger
from src.ui.settings_window import SettingsWindow
from src.ui.tray import TrayIcon


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


class _Worker(QObject):
    """QObject wrapper that runs ``App`` in a background ``QThread``."""

    def __init__(self, app: App) -> None:
        """Initialize the worker.

        Args:
            app: Application orchestrator to run in the thread.
        """
        super().__init__()
        self.app = app

    def start(self) -> None:
        """Start the App loop in this thread."""
        self.app.start()


def main(argv: list[str] | None = None) -> int:
    """Application entry point.

    Creates the Qt application, system tray icon, and runs the dictation
    service in a background ``QThread`` so the tray UI stays responsive.

    Args:
        argv: Optional argument list.

    Returns:
        Exit code from the Qt event loop.
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

    qt_app = QApplication(sys.argv)
    qt_app.setQuitOnLastWindowClosed(False)

    app = App(settings)
    settings_window = SettingsWindow(settings, env_file=env_file)
    tray = TrayIcon(app, settings, settings_window=settings_window)
    tray.show()

    def _on_recording_started() -> None:
        """Update tray indicator and notify when recording starts."""
        tray.set_recording(True)
        tray.notify("Recording", "Recording...")

    def _on_recording_stopped() -> None:
        """Restore idle tray indicator when recording stops."""
        tray.set_recording(False)

    def _on_text_injected(text: str) -> None:
        """Notify when text was transcribed/injected after a recording."""
        if settings.dry_run:
            preview = text if len(text) <= 40 else f"{text[:40]}..."
            tray.notify("Dictation", f"Transcribed: {preview}")
        else:
            tray.notify("Dictation", "Text injected")

    app.recording_started = _on_recording_started
    app.recording_stopped = _on_recording_stopped
    app.text_injected = _on_text_injected

    def _on_settings_saved(new_settings: Settings) -> None:
        """Update in-memory settings and advise the user about restart."""
        app.settings = new_settings
        tray.settings = new_settings
        tray.show_message(
            "Settings saved",
            "Restart the application for hotkey and ASR model changes to take effect.",
        )

    settings_window.settings_saved.connect(_on_settings_saved)

    worker = _Worker(app)
    thread = QThread()
    worker.moveToThread(thread)
    thread.started.connect(worker.start)
    thread.finished.connect(worker.deleteLater)
    thread.finished.connect(thread.deleteLater)

    def _on_quit() -> None:
        app.stop()
        thread.quit()
        thread.wait(5000)

    qt_app.aboutToQuit.connect(_on_quit)
    thread.start()

    return qt_app.exec()


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
