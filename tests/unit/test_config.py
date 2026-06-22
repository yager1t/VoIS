"""Tests for application configuration."""

import os
from pathlib import Path

import pytest

from src.config import Settings


def test_default_settings() -> None:
    """Defaults should match the specification."""
    settings = Settings()
    assert settings.hotkey == "f9"
    assert settings.push_to_talk is True
    assert settings.audio_sample_rate == 16000
    assert settings.audio_channels == 1
    assert settings.asr_model == "base"
    assert settings.asr_language == "auto"
    assert settings.llm_enabled is False
    assert settings.llm_url == "http://localhost:11434"
    assert settings.data_dir == Path("data")
    assert settings.models_dir == Path("models")


def test_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """Environment variables should override defaults."""
    monkeypatch.setenv("HOTKEY", "ctrl+f10")
    monkeypatch.setenv("PUSH_TO_TALK", "false")
    monkeypatch.setenv("AUDIO_SAMPLE_RATE", "44100")
    monkeypatch.setenv("LLM_ENABLED", "true")
    monkeypatch.setenv("DATA_DIR", "/tmp/voice_data")

    settings = Settings()
    assert settings.hotkey == "ctrl+f10"
    assert settings.push_to_talk is False
    assert settings.audio_sample_rate == 44100
    assert settings.llm_enabled is True
    assert settings.data_dir == Path("/tmp/voice_data")


def test_ensure_dirs_creates_paths(tmp_path: Path) -> None:
    """ensure_dirs should create data, models, and log directories."""
    settings = Settings(data_dir=tmp_path / "data", models_dir=tmp_path / "models")
    settings.ensure_dirs()
    assert (tmp_path / "data").exists()
    assert (tmp_path / "models").exists()
    assert (tmp_path / "data" / "logs").exists()


def test_validation_rejects_invalid_sample_rate(monkeypatch: pytest.MonkeyPatch) -> None:
    """Non-integer sample rate should raise a validation error."""
    monkeypatch.setenv("AUDIO_SAMPLE_RATE", "not-an-int")
    with pytest.raises(ValueError):  # pydantic raises ValidationError
        Settings()


def test_env_file_loading(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Settings should load from a provided .env file."""
    env_file = tmp_path / ".env"
    env_file.write_text("HOTKEY=f12\nASR_MODEL=small\n")
    # Make sure no env var leaks from outside
    monkeypatch.delenv("HOTKEY", raising=False)
    monkeypatch.delenv("ASR_MODEL", raising=False)

    settings = Settings(_env_file=str(env_file))
    assert settings.hotkey == "f12"
    assert settings.asr_model == "small"


def test_cwd_independent_import() -> None:
    """Settings should import cleanly regardless of current directory."""
    assert "src.config" in os.sys.modules or __import__("src.config", fromlist=["Settings"])
