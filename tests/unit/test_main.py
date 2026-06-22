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


@pytest.fixture
def main_mocks(tmp_path):
    """Return shared mocks for main() Qt/lifecycle tests."""
    mock_app = MagicMock()
    mock_qapp = MagicMock()
    mock_qapp.exec.return_value = 42
    mock_thread = MagicMock()
    mock_worker = MagicMock()
    mock_tray = MagicMock()
    mock_settings_window = MagicMock()

    env_file = tmp_path / "custom.env"
    env_file.write_text("ASR_MODEL=small\n")

    with (
        patch("src.main.Settings") as mock_settings_cls,
        patch("src.main.configure_logging"),
        patch("src.main.App", return_value=mock_app) as mock_app_cls,
        patch("src.main.QApplication", return_value=mock_qapp) as mock_qapp_cls,
        patch("src.main.TrayIcon", return_value=mock_tray) as mock_tray_cls,
        patch(
            "src.main.SettingsWindow", return_value=mock_settings_window
        ) as mock_settings_window_cls,
        patch("src.main.QThread", return_value=mock_thread) as mock_qthread_cls,
        patch("src.main._Worker", return_value=mock_worker) as mock_worker_cls,
    ):
        mock_settings = MagicMock()
        mock_settings.data_dir = tmp_path / "data"
        mock_settings.model_dump.return_value = {}
        mock_settings_cls.return_value = mock_settings

        yield {
            "settings": mock_settings,
            "settings_cls": mock_settings_cls,
            "app": mock_app,
            "app_cls": mock_app_cls,
            "qapp": mock_qapp,
            "qapp_cls": mock_qapp_cls,
            "tray": mock_tray,
            "tray_cls": mock_tray_cls,
            "settings_window": mock_settings_window,
            "settings_window_cls": mock_settings_window_cls,
            "thread": mock_thread,
            "qthread_cls": mock_qthread_cls,
            "worker": mock_worker,
            "worker_cls": mock_worker_cls,
            "env_file": env_file,
        }


def test_main_loads_env_from_config(main_mocks) -> None:
    """Main should load .env, create Qt/App/Tray objects, start worker, and exec."""
    mocks = main_mocks

    result = main(["--config", str(mocks["env_file"])])

    assert result == 42
    mocks["settings_cls"].assert_called_once()
    _, kwargs = mocks["settings_cls"].call_args
    assert kwargs["_env_file"] == str(mocks["env_file"])

    mocks["qapp_cls"].assert_called_once()
    mocks["app_cls"].assert_called_once_with(mocks["settings"])
    mocks["settings_window_cls"].assert_called_once_with(
        mocks["settings"], env_file=str(mocks["env_file"])
    )
    mocks["tray_cls"].assert_called_once_with(
        mocks["app"], mocks["settings"], settings_window=mocks["settings_window"]
    )
    mocks["tray"].show.assert_called_once()
    mocks["worker_cls"].assert_called_once_with(mocks["app"])
    mocks["qthread_cls"].assert_called_once()
    mocks["thread"].start.assert_called_once()
    mocks["qapp"].exec.assert_called_once()


def test_main_quit_handler_stops_app_and_thread(main_mocks) -> None:
    """The aboutToQuit handler should stop App and the background QThread."""
    mocks = main_mocks

    main(["--config", str(mocks["env_file"])])

    connected = mocks["qapp"].aboutToQuit.connect.call_args[0][0]
    connected()

    mocks["app"].stop.assert_called_once()
    mocks["thread"].quit.assert_called_once()
    mocks["thread"].wait.assert_called_once_with(5000)
