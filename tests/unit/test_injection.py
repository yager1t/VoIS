"""Unit tests for the text injection layer."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from src.injection import WindowsTextInjector, create_text_injector
from src.injection.base import TextInjector


@pytest.fixture
def mock_sendinput() -> MagicMock:
    """Patch ``user32.SendInput`` so tests never emit real keystrokes."""
    with patch("src.injection.windows.ctypes.windll.user32.SendInput") as mock:
        mock.return_value = 2
        yield mock


def _total_inputs_sent(mock_sendinput: MagicMock) -> int:
    """Sum the ``nInputs`` argument across all ``SendInput`` calls."""
    return sum(call.args[0] for call in mock_sendinput.call_args_list)


def test_windows_injector_is_available() -> None:
    """The Windows injector reports availability matching ``sys.platform``."""
    assert WindowsTextInjector.is_available() == (sys.platform == "win32")


def test_direct_unicode_sends_two_events_per_character(mock_sendinput: MagicMock) -> None:
    """Direct injection should emit key-down and key-up for every character."""
    injector = WindowsTextInjector(fallback_to_clipboard=False)
    injector.inject("abc")

    assert _total_inputs_sent(mock_sendinput) == 6
    assert mock_sendinput.call_count == 3


def test_inject_with_delay_sends_one_character_at_a_time(mock_sendinput: MagicMock) -> None:
    """Per-character delay should still send the correct total events."""
    injector = WindowsTextInjector(fallback_to_clipboard=False)
    with patch("src.injection.windows.time.sleep") as sleep_mock:
        injector.inject_with_delay("ab", delay_ms=1.0)

    assert _total_inputs_sent(mock_sendinput) == 4
    assert sleep_mock.call_count == 2


def test_fallback_uses_clipboard_and_ctrl_v(mock_sendinput: MagicMock) -> None:
    """Clipboard fallback should copy text and simulate ``Ctrl+V``."""
    injector = WindowsTextInjector(fallback_to_clipboard=True)
    text = "hello"

    with patch("src.injection.windows.pyperclip") as pyperclip_mock:
        injector.inject(text)

    pyperclip_mock.copy.assert_called_once_with(text)
    # Ctrl down, V down, V up, Ctrl up = 4 inputs in a single SendInput call.
    assert mock_sendinput.call_count == 1
    assert _total_inputs_sent(mock_sendinput) == 4


def test_factory_returns_windows_injector_on_windows() -> None:
    """The factory should return a Windows injector on win32."""
    injector = create_text_injector("win32")
    assert isinstance(injector, WindowsTextInjector)
    assert isinstance(injector, TextInjector)


def test_factory_rejects_unsupported_platform() -> None:
    """The factory should raise for unsupported platforms."""
    with pytest.raises(NotImplementedError, match="not implemented for platform"):
        create_text_injector("darwin")
