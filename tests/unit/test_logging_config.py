"""Unit tests for logging configuration."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

from src.logging_config import configure_logging


def test_configure_logging_adds_stderr_when_available(tmp_path: Path) -> None:
    """Console sink is added when sys.stderr is available."""
    with (
        patch("src.logging_config.os.makedirs"),
        patch("src.logging_config.logger") as mock_logger,
        patch("src.logging_config.os.environ", {"VOICE_DATA_DIR": str(tmp_path)}),
    ):
        configure_logging("INFO")

    mock_logger.remove.assert_called_once()
    assert mock_logger.add.call_count == 2


def test_configure_logging_skips_stderr_when_none(tmp_path: Path) -> None:
    """Console sink is skipped in windowed builds where sys.stderr is None."""
    with (
        patch("src.logging_config.os.makedirs"),
        patch("src.logging_config.logger") as mock_logger,
        patch("src.logging_config.os.environ", {"VOICE_DATA_DIR": str(tmp_path)}),
        patch.object(sys, "stderr", None),
    ):
        configure_logging("INFO")

    mock_logger.remove.assert_called_once()
    # Only the file sink should be added when stderr is None.
    assert mock_logger.add.call_count == 1
