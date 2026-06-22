"""Unit tests for the ASR model manager."""

from pathlib import Path

from src.asr.model_manager import ModelManager
from src.config import Settings


def test_model_path_resolves_under_models_dir(tmp_path: Path) -> None:
    """Model paths should be nested under the configured models directory."""
    settings = Settings(models_dir=tmp_path / "models")
    manager = ModelManager(settings)

    assert manager.model_path("base") == tmp_path / "models" / "base"


def test_list_available_returns_cached_models(tmp_path: Path) -> None:
    """Only non-empty directories inside models_dir should be listed."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "base").mkdir()
    (models_dir / "base" / "model.bin").write_text("fake")
    (models_dir / "empty").mkdir()
    (models_dir / "not_a_dir.txt").write_text("ignore me")

    manager = ModelManager(Settings(models_dir=models_dir))
    available = manager.list_available()

    assert available == ["base"]


def test_list_available_empty_when_dir_missing() -> None:
    """An absent models_dir should yield an empty list."""
    manager = ModelManager(Settings(models_dir=Path("/does/not/exist")))

    assert manager.list_available() == []
