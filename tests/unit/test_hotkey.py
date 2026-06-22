"""Unit tests for the global hotkey manager."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from pynput import keyboard

from src.hotkey import PynputHotkeyManager, create_hotkey_manager, parse_hotkey


def test_parse_hotkey_simple() -> None:
    """A single key should produce no modifiers."""
    modifiers, main_key = parse_hotkey("f9")
    assert modifiers == set()
    assert main_key == "f9"


def test_parse_hotkey_combo_with_brackets() -> None:
    """Angle-bracket modifiers should be normalized."""
    modifiers, main_key = parse_hotkey("<ctrl>+f9")
    assert modifiers == {"ctrl"}
    assert main_key == "f9"


def test_parse_hotkey_combo_without_brackets() -> None:
    """Plain modifier names should also be accepted."""
    modifiers, main_key = parse_hotkey("ctrl+alt+f9")
    assert modifiers == {"ctrl", "alt"}
    assert main_key == "f9"


def test_parse_hotkey_ignores_whitespace() -> None:
    """Spaces around plus signs should be tolerated."""
    modifiers, main_key = parse_hotkey("  <ctrl> + <shift> + f9  ")
    assert modifiers == {"ctrl", "shift"}
    assert main_key == "f9"


def test_parse_hotkey_rejects_empty() -> None:
    """Empty hotkeys should raise a clear error."""
    with pytest.raises(ValueError, match="hotkey must not be empty"):
        parse_hotkey("   ")


def test_parse_hotkey_rejects_unknown_modifier() -> None:
    """Unknown modifiers should raise a clear error."""
    with pytest.raises(ValueError, match="unknown modifier"):
        parse_hotkey("<foo>+f9")


def test_create_hotkey_manager_wires_callbacks() -> None:
    """The factory should attach provided callbacks."""
    press_mock = MagicMock()
    release_mock = MagicMock()
    manager = create_hotkey_manager(
        "f9",
        push_to_talk=True,
        on_press=press_mock,
        on_release=release_mock,
    )
    assert manager.on_press is press_mock
    assert manager.on_release is release_mock
    assert manager.push_to_talk is True


def test_invalid_main_key_raises() -> None:
    """An unresolvable main key should fail fast at construction."""
    with pytest.raises(ValueError, match="invalid hotkey"):
        PynputHotkeyManager("<unknown_key>")


def _capture_callbacks(listener_cls: MagicMock) -> tuple[MagicMock, MagicMock]:
    """Return the on_press/on_release callbacks passed to a mocked Listener."""
    _, kwargs = listener_cls.call_args
    return kwargs["on_press"], kwargs["on_release"]


@patch("src.hotkey.windows.keyboard.Listener")
def test_start_stop_lifecycle(mock_listener_cls: MagicMock) -> None:
    """Start should run a daemon listener; stop should clean it up."""
    listener_instance = MagicMock()
    listener_instance.is_alive.return_value = True
    mock_listener_cls.return_value = listener_instance

    manager = PynputHotkeyManager("f9")
    assert not manager.is_alive()

    manager.start()
    mock_listener_cls.assert_called_once()
    listener_instance.run.assert_called_once()
    assert manager.is_alive()

    manager.stop()
    listener_instance.stop.assert_called_once()
    assert not manager.is_alive()


@patch("src.hotkey.windows.keyboard.Listener")
def test_push_to_talk_emits_press_and_release(mock_listener_cls: MagicMock) -> None:
    """In push-to-talk mode, press fires on combo and release fires on key up."""
    listener_instance = MagicMock()
    mock_listener_cls.return_value = listener_instance

    press_mock = MagicMock()
    release_mock = MagicMock()
    manager = PynputHotkeyManager("<ctrl>+f9", push_to_talk=True)
    manager.on_press = press_mock
    manager.on_release = release_mock
    manager.start()
    on_press, on_release = _capture_callbacks(mock_listener_cls)

    on_press(keyboard.Key.ctrl)
    press_mock.assert_not_called()

    on_press(keyboard.Key.f9)
    press_mock.assert_called_once()
    release_mock.assert_not_called()

    on_release(keyboard.Key.f9)
    release_mock.assert_called_once()

    on_release(keyboard.Key.ctrl)
    press_mock.assert_called_once()
    release_mock.assert_called_once()

    manager.stop()


@patch("src.hotkey.windows.keyboard.Listener")
def test_toggle_emits_only_press(mock_listener_cls: MagicMock) -> None:
    """In toggle mode, only press events are emitted."""
    listener_instance = MagicMock()
    mock_listener_cls.return_value = listener_instance

    press_mock = MagicMock()
    release_mock = MagicMock()
    manager = PynputHotkeyManager("<ctrl>+f9", push_to_talk=False)
    manager.on_press = press_mock
    manager.on_release = release_mock
    manager.start()
    on_press, on_release = _capture_callbacks(mock_listener_cls)

    on_press(keyboard.Key.ctrl)
    on_press(keyboard.Key.f9)
    press_mock.assert_called_once()

    on_release(keyboard.Key.f9)
    on_release(keyboard.Key.ctrl)
    release_mock.assert_not_called()

    manager.stop()


@patch("src.hotkey.windows.keyboard.Listener")
def test_press_held_does_not_repeat(mock_listener_cls: MagicMock) -> None:
    """Holding the combo should only invoke on_press once."""
    listener_instance = MagicMock()
    mock_listener_cls.return_value = listener_instance

    press_mock = MagicMock()
    manager = PynputHotkeyManager("f9", push_to_talk=True)
    manager.on_press = press_mock
    manager.start()
    on_press, _ = _capture_callbacks(mock_listener_cls)

    on_press(keyboard.Key.f9)
    on_press(keyboard.Key.f9)
    on_press(keyboard.Key.f9)
    press_mock.assert_called_once()

    manager.stop()
