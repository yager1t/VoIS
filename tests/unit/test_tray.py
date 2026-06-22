"""Unit tests for the system tray icon."""

# ruff: noqa: N802

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.config import Settings


class _FakeMessageIcon:
    Information = "info"


class _FakeActivationReason:
    Trigger = "trigger"


class _FakeSystemTrayIcon:
    MessageIcon = _FakeMessageIcon
    ActivationReason = _FakeActivationReason

    def __init__(self) -> None:
        self._icon: object | None = None
        self._menu: object | None = None
        self._shown = False
        self._messages: list[tuple[tuple[object, ...], dict[str, object]]] = []
        self.activated = MagicMock()

    def setIcon(self, icon: object) -> None:
        self._icon = icon

    def setContextMenu(self, menu: object) -> None:
        self._menu = menu

    def show(self) -> None:
        self._shown = True

    def showMessage(self, *args: object, **kwargs: object) -> None:
        self._messages.append((args, kwargs))


class _FakeAction:
    def __init__(self, text: str) -> None:
        self.text = text
        self.enabled = True
        self.triggered = MagicMock()

    def setEnabled(self, enabled: bool) -> None:
        self.enabled = enabled

    def setText(self, text: str) -> None:
        self.text = text


class _FakeMenu:
    def __init__(self) -> None:
        self.actions: list[_FakeAction] = []

    def addAction(self, action: _FakeAction) -> None:
        self.actions.append(action)


class _FakeStyle:
    StandardPixmap = MagicMock()

    def standardIcon(self, icon: object) -> str:
        return f"icon-{icon}"


class _FakeApplication:
    _style = _FakeStyle()

    @classmethod
    def style(cls) -> _FakeStyle:
        return cls._style

    @classmethod
    def quit(cls) -> None:
        pass


class _FakeQtWidgetsModule:
    QSystemTrayIcon = _FakeSystemTrayIcon
    QMenu = _FakeMenu
    QApplication = _FakeApplication
    QStyle = _FakeStyle


class _FakeQtGuiModule:
    QAction = _FakeAction
    QIcon = str


class _FakeQtCoreModule:
    pass


class _FakeApp:
    def __init__(self) -> None:
        self.running = False
        self.recording_started = None
        self.recording_stopped = None
        self.text_injected = None

    def is_running(self) -> bool:
        return self.running

    def start(self) -> None:
        self.running = True

    def stop(self) -> None:
        self.running = False


@pytest.fixture
def mock_qt():
    """Provide mocked PyQt6 modules using real fake classes."""
    modules = {
        "PyQt6.QtWidgets": _FakeQtWidgetsModule(),
        "PyQt6.QtGui": _FakeQtGuiModule(),
        "PyQt6.QtCore": _FakeQtCoreModule(),
    }
    with patch.dict("sys.modules", modules):
        yield modules


@pytest.fixture
def tray(mock_qt, tmp_path):
    """Return a TrayIcon instance built against mocked Qt objects."""
    import sys

    # Force a fresh import so the PyQt6 sys.modules patch takes effect.
    sys.modules.pop("src.ui.tray", None)
    sys.modules.pop("src.ui", None)

    settings = Settings(data_dir=tmp_path / "data", models_dir=tmp_path / "models")
    app = _FakeApp()
    from src.ui.tray import TrayIcon

    tray_icon = TrayIcon(app, settings)
    return tray_icon, app


def test_tray_creates_menu_and_actions(tray) -> None:
    """TrayIcon should build a context menu with Start, Settings, and Exit."""
    tray_icon, app = tray

    assert isinstance(tray_icon._menu, _FakeMenu)
    assert len(tray_icon._menu.actions) == 3
    assert tray_icon._toggle_action.text == "Start"
    assert tray_icon._settings_action.enabled is True
    assert tray_icon._exit_action.text == "Exit"
    assert app.is_running() is False


def test_tray_wires_app_callbacks(tray) -> None:
    """TrayIcon should attach its UI handlers to App callback attributes."""
    tray_icon, app = tray

    assert app.recording_started is not None
    assert app.recording_started.__func__ is tray_icon._on_recording_started.__func__
    assert app.recording_stopped.__func__ is tray_icon._on_recording_stopped.__func__
    assert app.text_injected.__func__ is tray_icon._on_text_injected.__func__


def test_tray_show_message_delegates_to_show_message(tray) -> None:
    """show_message should call the underlying QSystemTrayIcon.showMessage."""
    tray_icon, _ = tray

    tray_icon.show_message("Title", "Body")

    assert len(tray_icon._messages) == 1
    args, kwargs = tray_icon._messages[0]
    assert args == ("Title", "Body", _FakeMessageIcon.Information, 3000)
    assert kwargs == {}


def test_tray_set_recording_icon_switches_icon(tray) -> None:
    """set_recording_icon should switch between base and recording icons."""
    tray_icon, _ = tray

    tray_icon.set_recording_icon(True)
    assert tray_icon._icon == tray_icon._recording_icon
    tray_icon.set_recording_icon(False)
    assert tray_icon._icon == tray_icon._base_icon


def test_tray_toggle_starts_app_when_idle(tray) -> None:
    """Clicking the toggle action should start the app when it is not running."""
    tray_icon, app = tray
    assert app.is_running() is False

    tray_icon._on_toggle()

    assert app.is_running() is True


def test_tray_toggle_stops_app_when_running(tray) -> None:
    """Clicking the toggle action should stop the app when it is running."""
    tray_icon, app = tray
    app.running = True

    tray_icon._on_toggle()

    assert app.is_running() is False


def test_tray_activation_trigger_toggles(tray) -> None:
    """A left-click (Trigger) on the tray icon should toggle the app."""
    tray_icon, app = tray
    assert app.is_running() is False

    tray_icon._on_activated(_FakeActivationReason.Trigger)

    assert app.is_running() is True


def test_tray_settings_action_shows_settings_window(tray) -> None:
    """The Settings action should show and raise the settings window."""
    tray_icon, _ = tray
    fake_window = MagicMock()
    tray_icon.settings_window = fake_window

    tray_icon._on_settings()

    fake_window.show.assert_called_once()
    fake_window.raise_.assert_called_once()


def test_tray_settings_action_without_window_is_safe(tray) -> None:
    """The Settings action should be a no-op when no window is attached."""
    tray_icon, _ = tray
    tray_icon.settings_window = None

    tray_icon._on_settings()
