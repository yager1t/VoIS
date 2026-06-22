"""Global hotkey manager factory and exports."""

from collections.abc import Callable

from src.hotkey.base import HotkeyManager, parse_hotkey
from src.hotkey.windows import PynputHotkeyManager


def create_hotkey_manager(
    hotkey: str,
    *,
    push_to_talk: bool = True,
    on_press: Callable[[], None] | None = None,
    on_release: Callable[[], None] | None = None,
) -> PynputHotkeyManager:
    """Create and configure a global hotkey manager.

    Args:
        hotkey: Human-readable hotkey such as ``"f9"`` or ``"<ctrl>+f9"``.
        push_to_talk: If ``True``, separate press/release events are emitted.
        on_press: Optional callback invoked when the hotkey combo is pressed.
        on_release: Optional callback invoked when the main key is released
            (push-to-talk mode only).

    Returns:
        A configured :class:`PynputHotkeyManager` instance.
    """
    manager = PynputHotkeyManager(hotkey, push_to_talk=push_to_talk)
    manager.on_press = on_press
    manager.on_release = on_release
    return manager


__all__ = ["HotkeyManager", "PynputHotkeyManager", "create_hotkey_manager", "parse_hotkey"]
