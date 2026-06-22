"""Smoke script for the Windows text injection layer.

Focus a text field before the delay expires; the script will type a mixed
English/Cyrillic string via ``SendInput``.
"""

from __future__ import annotations

import time

from src.injection import WindowsTextInjector


def main() -> None:
    """Wait briefly, then inject a smoke-test string."""
    print("Switch to a text field; injection starts in 3 seconds...")
    time.sleep(3)

    injector = WindowsTextInjector()
    injector.inject("Hello from Voice-to-Cursor! Привет!")
    print("Injection complete.")


if __name__ == "__main__":
    main()
