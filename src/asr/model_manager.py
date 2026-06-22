"""Local model discovery and downloading for ASR backends."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:  # pragma: no cover
    from faster_whisper import WhisperModel

from src.config import Settings


def _download_model(model_name: str, output_dir: str) -> None:
    """Download a faster-whisper model.

    Kept as a small wrapper so unit tests can patch it without importing the
    heavy ASR package or touching the network.
    """
    from faster_whisper import download_model

    download_model(model_name, output_dir=output_dir)


def _create_whisper_model(
    model_name: str,
    *,
    device: str,
    compute_type: str,
    download_root: str,
) -> WhisperModel:
    """Create a faster-whisper model instance behind a patchable wrapper."""
    from faster_whisper import WhisperModel

    return WhisperModel(
        model_name,
        device=device,
        compute_type=compute_type,
        local_files_only=False,
        download_root=download_root,
    )


class ModelManager:
    """Resolve and download faster-whisper model checkpoints.

    Models are stored under ``models/<model_name>`` relative to the project root
    or the configured ``models_dir``.
    """

    DEFAULT_MODELS = {"tiny", "tiny.en", "base", "base.en", "small", "small.en"}

    def __init__(self, settings: Settings | None = None) -> None:
        """Create a model manager from settings or project defaults.

        Args:
            settings: Optional application settings. When omitted, a default
                ``Settings`` instance is used.
        """
        self.settings = settings or Settings()
        self._models_dir: Path = self.settings.models_dir

    def model_path(self, model_name: str) -> Path:
        """Return the local directory path for ``model_name``.

        Args:
            model_name: Name of the faster-whisper model.

        Returns:
            Absolute directory path where the model should be cached.
        """
        return self._models_dir / model_name

    def list_available(self) -> list[str]:
        """Return locally cached model names found in ``models_dir``.

        Returns:
            Sorted list of directory names that contain model files.
        """
        if not self._models_dir.exists():
            return []

        available: list[str] = []
        for path in self._models_dir.iterdir():
            if path.is_dir() and any(path.iterdir()):
                available.append(path.name)
        return sorted(available)

    def ensure_model(self, model_name: str) -> Path:
        """Return the local model path, downloading it if necessary.

        Args:
            model_name: Name of the faster-whisper model to resolve.

        Returns:
            Path to the local model directory.

        Raises:
            RuntimeError: If the model cannot be downloaded or loaded.
        """
        target = self.model_path(model_name)
        if target.exists() and any(target.iterdir()):
            logger.debug("Model '{}' found at {}", model_name, target)
            return target

        logger.info("Downloading model '{}' to {}", model_name, target)
        try:
            _download_model(model_name, output_dir=str(target))
        except Exception as exc:  # pragma: no cover - network/model errors
            logger.exception("Failed to download model '{}': {}", model_name, exc)
            raise RuntimeError(f"Could not download model '{model_name}'") from exc

        return target

    def load_whisper_model(
        self,
        model_name: str,
        device: str | None = None,
        compute_type: str | None = None,
    ) -> WhisperModel:
        """Download if needed and load a ``faster_whisper.WhisperModel``.

        Args:
            model_name: Name of the model to load.
            device: Device string passed to ``WhisperModel`` (e.g. ``cpu``).
            compute_type: Compute type passed to ``WhisperModel`` (e.g. ``int8``).

        Returns:
            Loaded ``WhisperModel`` instance.
        """
        self.ensure_model(model_name)
        return _create_whisper_model(
            model_name,
            device=device or self.settings.asr_device,
            compute_type=compute_type or self.settings.asr_compute_type,
            download_root=str(self._models_dir),
        )
