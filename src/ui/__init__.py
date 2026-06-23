"""User interface components for Voice-to-Cursor."""

from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = ["SettingsWindow", "TrayIcon", "VocabularyEditor"]

if TYPE_CHECKING:  # pragma: no cover
    from src.ui.settings_window import SettingsWindow
    from src.ui.tray import TrayIcon
    from src.ui.vocab_editor import VocabularyEditor


def __getattr__(name: str) -> object:
    """Lazy-load UI classes without importing every Qt module at package import."""
    if name == "SettingsWindow":
        from src.ui.settings_window import SettingsWindow

        return SettingsWindow
    if name == "TrayIcon":
        from src.ui.tray import TrayIcon

        return TrayIcon
    if name == "VocabularyEditor":
        from src.ui.vocab_editor import VocabularyEditor

        return VocabularyEditor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
