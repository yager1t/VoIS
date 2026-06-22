"""Application lifecycle skeleton."""

from loguru import logger

from src.config import Settings


class App:
    """Main application orchestrator."""

    def __init__(self, settings: Settings) -> None:
        """Initialize the application with validated settings.

        Args:
            settings: Parsed application configuration.
        """
        self.settings = settings
        self._running = False

    def start(self) -> None:
        """Start the voice-to-cursor service."""
        self.settings.ensure_dirs()
        self._running = True
        logger.info("Voice-to-Cursor started (hotkey: {})", self.settings.hotkey)
        try:
            while self._running:
                # Placeholder: event loop / hotkey listener will be wired here.
                pass  # pragma: no cover
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt received")
        finally:
            self.stop()

    def stop(self) -> None:
        """Stop the service and release resources."""
        if not self._running:
            return
        self._running = False
        logger.info("Voice-to-Cursor stopped")
