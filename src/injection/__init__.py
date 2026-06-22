"""Text injection factory and exports."""

from __future__ import annotations

import sys

from src.injection.base import TextInjector
from src.injection.windows import WindowsTextInjector


def create_text_injector(platform: str | None = None) -> TextInjector:
    """Create a text injector appropriate for the current platform.

    Args:
        platform: Optional platform override. Defaults to ``sys.platform``.

    Returns:
        A configured :class:`TextInjector` instance.

    Raises:
        NotImplementedError: When no injector is available for the platform.
    """
    platform = (platform or sys.platform).lower()
    if platform == "win32":
        return WindowsTextInjector()
    raise NotImplementedError(f"text injection not implemented for platform '{platform}'")


__all__ = ["TextInjector", "WindowsTextInjector", "create_text_injector"]
