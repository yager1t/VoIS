"""PyInstaller entry point wrapper for voice-to-cursor."""

import sys

from src.main import main

if __name__ == "__main__":
    sys.exit(main())
