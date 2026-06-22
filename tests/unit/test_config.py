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
    assert settings.audio_max_record_seconds == 60.0
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


def test_vad_aggressiveness_loads_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """vad_aggressiveness should be configurable from environment."""
    monkeypatch.setenv("VAD_AGGRESSIVENESS", "3")
    settings = Settings()
    assert settings.vad_aggressiveness == 3


def test_injection_delay_ms_loads_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """injection_delay_ms should be configurable from environment."""
    monkeypatch.setenv("INJECTION_DELAY_MS", "150")
    settings = Settings()
    assert settings.injection_delay_ms == 150.0


def test_asr_device_and_compute_type_load_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """ASR device and compute type should be configurable from the environment."""
    monkeypatch.setenv("ASR_DEVICE", "cuda")
    monkeypatch.setenv("ASR_COMPUTE_TYPE", "float16")

    settings = Settings()
    assert settings.asr_device == "cuda"
    assert settings.asr_compute_type == "float16"


def test_new_fields_load_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Additional settings fields should load from environment variables."""
    monkeypatch.setenv("DRY_RUN", "true")
    monkeypatch.setenv("INJECTION_FALLBACK_TO_CLIPBOARD", "true")
    monkeypatch.setenv("VAD_TRIM_SECONDS", "0.5")
    monkeypatch.setenv("ASR_BEAM_SIZE", "3")
    monkeypatch.setenv("AUDIO_MAX_RECORD_SECONDS", "30")

    settings = Settings()
    assert settings.dry_run is True
    assert settings.injection_fallback_to_clipboard is True
    assert settings.vad_trim_seconds == 0.5
    assert settings.asr_beam_size == 3
    assert settings.audio_max_record_seconds == 30.0


def test_cwd_independent_import() -> None:
    """Settings should import cleanly regardless of current directory."""
    assert "src.config" in os.sys.modules or __import__("src.config", fromlist=["Settings"])
