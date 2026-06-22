"""Unit tests for the CLI entry point."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.main import apply_cli_overrides, main, parse_args


def test_parse_args_defaults() -> None:
    """parse_args should provide sensible defaults."""
    args = parse_args([])

    assert args.config is None
    assert args.model is None
    assert args.language is None
    assert args.device is None
    assert args.hotkey is None
    assert args.toggle is False
    assert args.dry_run is False


def test_parse_args_model() -> None:
    """--model should be parsed."""
    args = parse_args(["--model", "small"])
    assert args.model == "small"


def test_parse_args_language() -> None:
    """--language should be parsed."""
    args = parse_args(["--language", "fr"])
    assert args.language == "fr"


def test_parse_args_device() -> None:
    """--device should accept cpu or cuda."""
    args = parse_args(["--device", "cuda"])
    assert args.device == "cuda"


def test_parse_args_invalid_device_raises_system_exit() -> None:
    """An invalid --device value should raise SystemExit."""
    with pytest.raises(SystemExit):
        parse_args(["--device", "tpu"])


def test_parse_args_hotkey() -> None:
    """--hotkey should be parsed."""
    args = parse_args(["--hotkey", "ctrl+f9"])
    assert args.hotkey == "ctrl+f9"


def test_parse_args_toggle() -> None:
    """--toggle should set toggle to True."""
    args = parse_args(["--toggle"])
    assert args.toggle is True


def test_parse_args_dry_run() -> None:
    """--dry-run should set dry_run to True."""
    args = parse_args(["--dry-run"])
    assert args.dry_run is True


def test_parse_args_config() -> None:
    """--config should be parsed."""
    args = parse_args(["--config", "custom.env"])
    assert args.config == "custom.env"


def test_apply_cli_overrides_applies_all() -> None:
    """apply_cli_overrides should update settings for every provided argument."""
    from src.config import Settings

    settings = Settings()
    args = parse_args(
        [
            "--model",
            "small",
            "--language",
            "de",
            "--device",
            "cuda",
            "--hotkey",
            "f10",
            "--toggle",
            "--dry-run",
        ]
    )

    apply_cli_overrides(settings, args)

    assert settings.asr_model == "small"
    assert settings.asr_language == "de"
    assert settings.asr_device == "cuda"
    assert settings.hotkey == "f10"
    assert settings.push_to_talk is False
    assert settings.dry_run is True


def test_apply_cli_overrides_leaves_none_args_untouched() -> None:
    """apply_cli_overrides should not change settings for None arguments."""
    from src.config import Settings

    settings = Settings()
    original = settings.model_dump()
    args = parse_args([])

    apply_cli_overrides(settings, args)

    assert settings.model_dump() == original


def test_main_loads_env_from_config(tmp_path) -> None:
    """Main should load .env from --config, create App, and call start."""
    env_file = tmp_path / "custom.env"
    env_file.write_text("ASR_MODEL=small\n")

    mock_app = MagicMock()

    with (
        patch("src.main.Settings") as mock_settings_cls,
        patch("src.main.configure_logging"),
        patch("src.main.App", return_value=mock_app) as mock_app_cls,
    ):
        mock_settings = MagicMock()
        mock_settings.data_dir = tmp_path / "data"
        mock_settings.model_dump.return_value = {}
        mock_settings_cls.return_value = mock_settings

        result = main(["--config", str(env_file)])

    assert result == 0
    mock_settings_cls.assert_called_once()
    _, kwargs = mock_settings_cls.call_args
    assert kwargs["_env_file"] == str(env_file)
    mock_app_cls.assert_called_once_with(mock_settings)
    mock_app.start.assert_called_once()


def test_main_returns_one_on_unhandled_exception() -> None:
    """Main should return exit code 1 on an unhandled exception."""
    with (
        patch("src.main.Settings") as mock_settings_cls,
        patch("src.main.configure_logging"),
        patch("src.main.App") as mock_app_cls,
    ):
        mock_settings = MagicMock()
        mock_settings.data_dir = MagicMock()
        mock_settings.model_dump.return_value = {}
        mock_settings_cls.return_value = mock_settings
        mock_app_cls.return_value.start.side_effect = RuntimeError("boom")

        result = main([])

    assert result == 1
    mock_app_cls.assert_called_once()
