"""Application settings via pydantic-settings."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment and .env files."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Hotkey & interaction
    hotkey: str = "f9"
    push_to_talk: bool = True

    # Audio capture
    audio_sample_rate: int = 16000
    audio_channels: int = 1

    # ASR
    asr_model: str = "base"
    asr_language: str = "auto"
    asr_device: str = "cpu"
    asr_compute_type: str = "int8"
    asr_beam_size: int = 5

    # LLM post-processing
    llm_enabled: bool = False
    llm_url: str = "http://localhost:11434"

    # Text injection
    injection_delay_ms: float = 0.0
    injection_fallback_to_clipboard: bool = False

    # Paths
    data_dir: Path = Path("data")
    models_dir: Path = Path("models")

    def ensure_dirs(self) -> None:
        """Create required application directories if they do not exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "logs").mkdir(parents=True, exist_ok=True)
