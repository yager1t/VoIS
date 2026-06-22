"""Windows global hotkey manager implemented with ``pynput``."""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Callable

from loguru import logger
from pynput import keyboard

from src.hotkey.base import HotkeyManager, parse_hotkey


class PynputHotkeyManager(HotkeyManager):
    """Global hotkey listener using ``pynput``.

    Supports simple keys (``f9``) and combinations (``<ctrl>+f9``) on Windows
    and other platforms where ``pynput`` can install a low-level keyboard hook.
    The listener runs in a daemon thread so it does not block application exit.
    """

    _MODIFIER_MAP: dict[str, str] = {
        "ctrl": "ctrl",
        "alt": "alt",
        "shift": "shift",
        "cmd": "cmd",
        "win": "cmd",
        "super": "cmd",
    }

    def __init__(self, hotkey: str, push_to_talk: bool = True) -> None:
        """Initialize the listener with a hotkey and interaction mode.

        Args:
            hotkey: Human-readable hotkey such as ``"f9"`` or ``"<ctrl>+f9"``.
            push_to_talk: If ``True``, press/release events are emitted separately.
                If ``False``, only a press event is emitted on each combo press.
        """
        super().__init__(hotkey, push_to_talk)
        self._listener: keyboard.Listener | None = None
        self._thread: threading.Thread | None = None
        self._pressed_keys: set[str] = set()
        self._combo_active = False
        self._lock = threading.Lock()

        try:
            self._main_key_obj = self._resolve_key(self._main_key)
            self._modifier_objs = {self._resolve_key(m) for m in self._modifiers}
        except ValueError as exc:
            raise ValueError(f"invalid hotkey '{hotkey}': {exc}") from exc

    def _resolve_key(self, name: str) -> keyboard.Key | keyboard.KeyCode:
        """Resolve a normalized key name to a ``pynput`` key object."""
        modifier_name = self._MODIFIER_MAP.get(name, name)
        key_attr = modifier_name if modifier_name != name else name

        if hasattr(keyboard.Key, key_attr):
            return getattr(keyboard.Key, key_attr)

        if len(name) == 1:
            return keyboard.KeyCode.from_char(name)

        raise ValueError(f"unknown key '{name}'")

    def start(self) -> None:
        """Start the keyboard listener in a daemon thread."""
        if self._listener is not None and self._listener.is_alive():
            logger.debug("Hotkey listener already running")
            return

        self._pressed_keys.clear()
        self._combo_active = False

        self._listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
            suppress=False,
        )
        self._thread = threading.Thread(target=self._listener.run, daemon=True)
        self._thread.start()
        logger.info(
            "Hotkey listener started for '{}' (push_to_talk={})",
            self.hotkey,
            self.push_to_talk,
        )

    def stop(self) -> None:
        """Stop the keyboard listener and release resources."""
        if self._listener is None:
            return

        try:
            self._listener.stop()
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Error stopping hotkey listener: {}", exc)
        finally:
            self._listener = None
            self._thread = None
            self._pressed_keys.clear()
            self._combo_active = False
            logger.info("Hotkey listener stopped")

    def is_alive(self) -> bool:
        """Return whether the listener thread is currently active."""
        return self._listener is not None and self._listener.is_alive()

    def _key_name(self, key: keyboard.Key | keyboard.KeyCode) -> str:
        """Return a normalized canonical name for a ``pynput`` key."""
        if isinstance(key, keyboard.Key):
            return str(key.name)
        if isinstance(key, keyboard.KeyCode):
            if key.char is not None:
                return str(key.char).lower()
            if key.vk is not None:
                return str(key.vk)
        return ""

    def _combo_matches(self) -> bool:
        """Return whether all required modifiers and the main key are pressed."""
        if not self._pressed_keys:
            return False

        main_name = self._key_name(self._main_key_obj)
        if main_name not in self._pressed_keys:
            return False

        for modifier in self._modifier_objs:
            if self._key_name(modifier) not in self._pressed_keys:
                return False

        return True

    def _on_press(self, key: keyboard.Key | keyboard.KeyCode) -> None:
        """Handle a key press event from ``pynput``."""
        name = self._key_name(key)
        if not name:
            return

        with self._lock:
            self._pressed_keys.add(name)
            if not self._combo_active and self._combo_matches():
                self._combo_active = True
                self._emit(self.on_press)

    def _on_release(self, key: keyboard.Key | keyboard.KeyCode) -> None:
        """Handle a key release event from ``pynput``."""
        name = self._key_name(key)
        if not name:
            return

        with self._lock:
            main_name = self._key_name(self._main_key_obj)
            if self.push_to_talk and self._combo_active and name == main_name:
                self._emit(self.on_release)

            self._pressed_keys.discard(name)

            if not self._combo_matches():
                self._combo_active = False

    def _emit(self, callback: Callable[[], None] | None) -> None:
        """Invoke a callback, logging any exception."""
        if callback is None:
            return
        try:
            callback()
        except Exception:  # pragma: no cover - defensive
            logger.exception("Hotkey callback failed")

    def __del__(self) -> None:
        """Ensure the listener is stopped on garbage collection."""
        with contextlib.suppress(Exception):  # pragma: no cover - best effort
            self.stop()


# Re-export helper so the module can be used standalone.
__all__ = ["PynputHotkeyManager", "parse_hotkey"]
