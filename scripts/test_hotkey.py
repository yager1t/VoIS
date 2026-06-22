"""Smoke test: listen to F9 for 10 seconds and print press/release events."""

from __future__ import annotations

import time

from src.hotkey import create_hotkey_manager


def main() -> None:
    """Run a 10-second F9 global hotkey listener demo."""
    press_count = 0
    release_count = 0

    def on_press() -> None:
        nonlocal press_count
        press_count += 1
        print(f"[PRESS] F9 pressed (total: {press_count})")

    def on_release() -> None:
        nonlocal release_count
        release_count += 1
        print(f"[RELEASE] F9 released (total: {release_count})")

    manager = create_hotkey_manager(
        "f9",
        push_to_talk=True,
        on_press=on_press,
        on_release=on_release,
    )

    print("Listening for F9 presses/releases for 10 seconds...")
    manager.start()
    try:
        time.sleep(10)
    finally:
        manager.stop()

    print(f"Done. Presses: {press_count}, Releases: {release_count}")


if __name__ == "__main__":
    main()
