"""Unit tests for logging configuration."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from loguru import logger

from src import logging_config
from src.logging_config import configure_logging


@pytest.fixture(autouse=True)
def reset_logger():
    """Remove all loguru sinks between tests."""
    logger.remove()
    yield
    logger.remove()


def test_configure_logging_uses_log_level_env(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """configure_logging should respect the LOG_LEVEL environment variable."""
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("VOICE_DATA_DIR", str(tmp_path))

    with patch.object(logging_config, "logger") as mock_logger:
        configure_logging()

    mock_logger.remove.assert_called_once()
    calls = mock_logger.add.call_args_list
    assert len(calls) == 2
    assert calls[0].kwargs["level"] == "DEBUG"


def test_configure_logging_creates_log_directory(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """configure_logging should create the data/logs directory."""
    monkeypatch.setenv("VOICE_DATA_DIR", str(tmp_path))

    with patch.object(logging_config, "logger"):
        configure_logging()

    assert (tmp_path / "logs").exists()


def test_configure_logging_adds_two_sinks(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """configure_logging should add exactly two sinks (stderr and file)."""
    monkeypatch.setenv("VOICE_DATA_DIR", str(tmp_path))

    with patch.object(logging_config, "logger") as mock_logger:
        configure_logging()

    assert mock_logger.add.call_count == 2
    first_args, _ = mock_logger.add.call_args_list[0]
    second_args, _ = mock_logger.add.call_args_list[1]
    assert first_args[0] is logging_config.sys.stderr
    assert isinstance(second_args[0], str)
    assert second_args[0].endswith("app.log")


def test_configure_logging_does_not_duplicate_sinks(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Repeated calls to configure_logging should remove existing sinks first."""
    monkeypatch.setenv("VOICE_DATA_DIR", str(tmp_path))

    with patch.object(logging_config, "logger") as mock_logger:
        configure_logging()
        configure_logging()

    assert mock_logger.add.call_count == 4
    assert mock_logger.remove.call_count == 2


def test_configure_logging_default_level(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """configure_logging should default to INFO when LOG_LEVEL is unset."""
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    monkeypatch.setenv("VOICE_DATA_DIR", str(tmp_path))

    with patch.object(logging_config, "logger") as mock_logger:
        configure_logging()

    assert mock_logger.add.call_args_list[0].kwargs["level"] == "INFO"
