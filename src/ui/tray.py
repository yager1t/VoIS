"""System tray icon for Voice-to-Cursor."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PyQt6.QtGui import QAction, QIcon
from PyQt6.QtWidgets import (
    QApplication,
    QMenu,
    QStyle,
    QSystemTrayIcon,
)

if TYPE_CHECKING:  # pragma: no cover
    from src.app import App
    from src.config import Settings
    from src.ui.settings_window import SettingsWindow


class TrayIcon(QSystemTrayIcon):
    """System tray icon with context menu and recording indicator."""

    def __init__(
        self,
        app: App,
        settings: Settings,
        settings_window: SettingsWindow | None = None,
    ) -> None:
        """Initialize tray icon and build context menu.

        Args:
            app: Application orchestrator.
            settings: Parsed application settings.
            settings_window: Optional settings window shown by the Settings action.
        """
        super().__init__()
        self.app = app
        self.settings = settings
        self.settings_window = settings_window

        self._base_icon = self._load_fallback_icon("base")
        self._recording_icon = self._load_fallback_icon("recording")
        self.setIcon(self._base_icon)

        self._menu = QMenu()
        self._toggle_action = QAction("Start")
        self._toggle_action.triggered.connect(self._on_toggle)
        self._menu.addAction(self._toggle_action)

        self._settings_action = QAction("Settings")
        self._settings_action.triggered.connect(self._on_settings)
        self._menu.addAction(self._settings_action)

        self._exit_action = QAction("Exit")
        self._exit_action.triggered.connect(QApplication.quit)
        self._menu.addAction(self._exit_action)

        self.setContextMenu(self._menu)
        self.activated.connect(self._on_activated)

        self._refresh_toggle()
        self._wire_app_callbacks()

    def _load_fallback_icon(self, kind: str) -> QIcon:
        """Return a fallback QStyle icon.

        Args:
            kind: ``"base"`` for idle icon or ``"recording"`` for active icon.

        Returns:
            A QIcon from the current style.
        """
        style = QApplication.style()
        if style is None:  # pragma: no cover - defensive guard for type checker
            return QIcon()
        if kind == "recording":
            return style.standardIcon(QStyle.StandardPixmap.SP_DialogApplyButton)
        return style.standardIcon(QStyle.StandardPixmap.SP_ComputerIcon)

    def _wire_app_callbacks(self) -> None:
        """Connect App callbacks to tray indicator methods."""
        if hasattr(self.app, "recording_started"):
            self.app.recording_started = self._on_recording_started
        if hasattr(self.app, "recording_stopped"):
            self.app.recording_stopped = self._on_recording_stopped
        if hasattr(self.app, "text_injected"):
            self.app.text_injected = self._on_text_injected

    def _on_toggle(self) -> None:
        """Toggle application start/stop."""
        if self.app.is_running():
            self.app.stop()
        else:
            self.app.start()
        self._refresh_toggle()

    def _refresh_toggle(self) -> None:
        """Update toggle action text based on application state."""
        if self.app.is_running():
            self._toggle_action.setText("Stop")
        else:
            self._toggle_action.setText("Start")

    def _on_settings(self) -> None:
        """Show the settings window when the Settings action is triggered."""
        if self.settings_window is not None:
            self.settings_window.show()
            self.settings_window.raise_()

    def _on_activated(self, reason: QSystemTrayIcon.ActivationReason) -> None:
        """Handle tray activation (e.g. left click).

        Args:
            reason: The activation reason emitted by QSystemTrayIcon.
        """
        if reason == QSystemTrayIcon.ActivationReason.Trigger:
            self._on_toggle()

    def _on_recording_started(self) -> None:
        """Handle recording started callback."""
        self.set_recording_icon(True)
        self.show_message("Recording", "Recording started")

    def _on_recording_stopped(self) -> None:
        """Handle recording stopped callback."""
        self.set_recording_icon(False)
        self.show_message("Recording", "Recording stopped")

    def _on_text_injected(self, text: str) -> None:
        """Handle text injected callback.

        Args:
            text: The text that was injected.
        """
        self.show_message("Dictation", text)

    def show_message(
        self,
        title: str,
        message: str,
        icon: QSystemTrayIcon.MessageIcon | None = None,
        timeout: int = 3000,
    ) -> None:
        """Show a balloon notification.

        Args:
            title: Notification title.
            message: Notification body.
            icon: Optional message icon; defaults to Information.
            timeout: Display duration in milliseconds.
        """
        if icon is None:
            icon = QSystemTrayIcon.MessageIcon.Information
        super().showMessage(title, message, icon, timeout)

    def set_recording_icon(self, recording: bool) -> None:
        """Switch between base and recording icons.

        Args:
            recording: Whether recording is active.
        """
        self.setIcon(self._recording_icon if recording else self._base_icon)
