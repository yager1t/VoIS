"""Windows text injector implemented with ``ctypes`` and ``SendInput``."""

from __future__ import annotations

import ctypes
import sys
import time
from ctypes import wintypes

import pyperclip
from loguru import logger

from src.injection.base import TextInjector

INPUT_KEYBOARD = 1
KEYEVENTF_UNICODE = 0x0004
KEYEVENTF_KEYUP = 0x0002
VK_CONTROL = 0x11
VK_V = 0x56


class KEYBDINPUT(ctypes.Structure):
    """Win32 ``KEYBDINPUT`` structure for ``SendInput``."""

    _fields_ = [
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.c_size_t),
    ]


class InputUnion(ctypes.Union):
    """Union member of the Win32 ``INPUT`` structure."""

    _fields_ = [
        ("ki", KEYBDINPUT),
        ("mi", ctypes.c_int * 8),  # placeholder, sized for alignment
        ("hi", ctypes.c_int * 8),
    ]


class INPUT(ctypes.Structure):
    """Win32 ``INPUT`` structure accepted by ``SendInput``."""

    _anonymous_ = ("u",)
    _fields_ = [
        ("type", wintypes.DWORD),
        ("u", InputUnion),
    ]


class WindowsTextInjector(TextInjector):
    """Inject text on Windows using ``user32.SendInput``.

    Unicode characters are sent as ``KEYEVENTF_UNICODE`` scan-code events, which
    works in most applications. For applications that do not accept Unicode
    injection, a clipboard-based fallback simulates ``Ctrl+V``.
    """

    def __init__(self, fallback_to_clipboard: bool = False) -> None:
        """Create a Windows text injector.

        Args:
            fallback_to_clipboard: If ``True``, :meth:`inject` uses the clipboard
                and ``Ctrl+V`` instead of direct Unicode input.
        """
        self.fallback_to_clipboard = fallback_to_clipboard

    @classmethod
    def is_available(cls) -> bool:
        """Return ``True`` only on Windows."""
        return sys.platform == "win32"

    def inject(self, text: str) -> None:
        """Type the given text at the current cursor position.

        Args:
            text: Unicode text to inject.
        """
        if self.fallback_to_clipboard:
            self._inject_via_clipboard(text)
        else:
            self._inject_unicode(text)

    def inject_with_delay(self, text: str, delay_ms: float = 0.0) -> None:
        """Type the given text with an optional per-character delay.

        Args:
            text: Unicode text to inject.
            delay_ms: Milliseconds to sleep between individual characters.
        """
        if self.fallback_to_clipboard:
            # Delay is not meaningful for a single paste operation.
            self._inject_via_clipboard(text)
            return

        for char in text:
            self._send_char(char)
            if delay_ms > 0:
                time.sleep(delay_ms / 1000.0)

    def _inject_unicode(self, text: str) -> None:
        """Send each character as a pair of Unicode ``SendInput`` events."""
        for char in text:
            self._send_char(char)

    def _send_char(self, char: str) -> None:
        """Send key-down and key-up for a single Unicode character."""
        if len(char) != 1:
            return

        scan = ord(char)
        down = INPUT(
            type=INPUT_KEYBOARD,
            ki=KEYBDINPUT(
                wVk=0,
                wScan=scan,
                dwFlags=KEYEVENTF_UNICODE,
                time=0,
                dwExtraInfo=0,
            ),
        )
        up = INPUT(
            type=INPUT_KEYBOARD,
            ki=KEYBDINPUT(
                wVk=0,
                wScan=scan,
                dwFlags=KEYEVENTF_UNICODE | KEYEVENTF_KEYUP,
                time=0,
                dwExtraInfo=0,
            ),
        )
        inputs = (INPUT * 2)(down, up)
        sent = ctypes.windll.user32.SendInput(2, ctypes.byref(inputs), ctypes.sizeof(INPUT))
        if sent != 2:
            logger.warning("SendInput returned {} instead of 2", sent)

    def _inject_via_clipboard(self, text: str) -> None:
        """Copy text to the clipboard and simulate ``Ctrl+V``."""
        try:
            pyperclip.copy(text)
        except Exception:  # pragma: no cover - defensive
            logger.exception("Failed to copy text to clipboard")
            return

        self._send_key_combination(VK_CONTROL, VK_V)

    def _send_key_combination(self, *vk_codes: int) -> None:
        """Press and release a sequence of virtual-key codes."""
        count = len(vk_codes)
        if count == 0:
            return

        inputs: list[INPUT] = []
        for vk in vk_codes:
            inputs.append(
                INPUT(
                    type=INPUT_KEYBOARD,
                    ki=KEYBDINPUT(
                        wVk=vk,
                        wScan=0,
                        dwFlags=0,
                        time=0,
                        dwExtraInfo=0,
                    ),
                )
            )
        for vk in reversed(vk_codes):
            inputs.append(
                INPUT(
                    type=INPUT_KEYBOARD,
                    ki=KEYBDINPUT(
                        wVk=vk,
                        wScan=0,
                        dwFlags=KEYEVENTF_KEYUP,
                        time=0,
                        dwExtraInfo=0,
                    ),
                )
            )

        arr = (INPUT * len(inputs))(*inputs)
        sent = ctypes.windll.user32.SendInput(len(inputs), ctypes.byref(arr), ctypes.sizeof(INPUT))
        if sent != len(inputs):
            logger.warning("SendInput returned {} instead of {}", sent, len(inputs))


__all__ = ["WindowsTextInjector"]
