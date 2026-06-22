"""Unit tests for the ASR model manager."""

from pathlib import Path
from unittest.mock import Mock, patch

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


def test_ensure_model_downloads_when_missing(tmp_path: Path) -> None:
    """ensure_model should download the model when the target directory is empty."""
    models_dir = tmp_path / "models"
    manager = ModelManager(Settings(models_dir=models_dir))

    with patch("src.asr.model_manager._download_model") as mock_download:
        result = manager.ensure_model("tiny")

    mock_download.assert_called_once_with("tiny", output_dir=str(manager.model_path("tiny")))
    assert result == manager.model_path("tiny")


def test_ensure_model_skips_when_present(tmp_path: Path) -> None:
    """ensure_model should skip download when the model directory already has files."""
    models_dir = tmp_path / "models"
    target = models_dir / "base"
    target.mkdir(parents=True)
    (target / "model.bin").write_text("fake")
    manager = ModelManager(Settings(models_dir=models_dir))

    with patch("src.asr.model_manager._download_model") as mock_download:
        result = manager.ensure_model("base")

    mock_download.assert_not_called()
    assert result == target


def test_load_whisper_model_returns_model(tmp_path: Path) -> None:
    """load_whisper_model should return a WhisperModel instance."""
    settings = Settings(models_dir=tmp_path / "models")
    manager = ModelManager(settings)

    fake_model = Mock()
    with (
        patch.object(manager, "ensure_model") as mock_ensure,
        patch(
            "src.asr.model_manager._create_whisper_model",
            return_value=fake_model,
        ) as mock_create,
    ):
        result = manager.load_whisper_model("tiny", device="cpu", compute_type="int8")

    mock_ensure.assert_called_once_with("tiny")
    mock_create.assert_called_once_with(
        "tiny",
        device="cpu",
        compute_type="int8",
        download_root=str(settings.models_dir),
    )
    assert result is fake_model
