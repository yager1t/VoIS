"""Settings window for Voice-to-Cursor."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QTextEdit,
    QWidget,
)

from src.config import Settings

ASR_MODEL_SIZES = ["tiny", "base", "small", "medium", "large"]
ASR_DEVICES = ["cpu", "cuda"]
CONTEXT_MODES = ["general", "chat", "email", "code"]


def _serialize_env_value(key: str, value: object) -> str:
    """Serialize a settings value for a ``KEY=value`` .env line.

    Args:
        key: Environment variable name.
        value: Value to serialize.

    Returns:
        Serialized string safe for python-dotenv.
    """
    if isinstance(value, Path):
        text = str(value)
    elif isinstance(value, bool):
        text = "true" if value else "false"
    else:
        text = str(value)

    # Quote values containing whitespace or special characters so python-dotenv
    # parses them as a single value.
    if any(ch in text for ch in (" ", "\t", "\n", '"', "\\", "#", "=")):
        escaped = text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
        return f'{key}="{escaped}"'
    return f"{key}={text}"


def _write_env_file(env_file: Path, settings: Settings) -> None:
    """Write settings to a ``.env`` file.

    Args:
        env_file: Destination path.
        settings: Settings to serialize.
    """
    lines = [
        _serialize_env_value(key.upper(), value)
        for key, value in settings.model_dump().items()
    ]
    env_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


class SettingsWindow(QWidget):
    """Editable settings window for the dictation application."""

    settings_saved = pyqtSignal(Settings)

    def __init__(self, settings: Settings, env_file: str | Path = ".env") -> None:
        """Initialize the settings window.

        Args:
            settings: Current application settings.
            env_file: Path to the ``.env`` file to write on save.
        """
        super().__init__()
        self.settings = settings
        self.env_file = Path(env_file)
        self.setWindowTitle("Voice-to-Cursor Settings")

        self._layout = QFormLayout(self)

        self._hotkey_edit = QLineEdit()
        self._layout.addRow("Hotkey", self._hotkey_edit)

        self._push_to_talk_check = QCheckBox()
        self._layout.addRow("Push-to-talk", self._push_to_talk_check)

        self._asr_model_combo = QComboBox()
        self._asr_model_combo.addItems(ASR_MODEL_SIZES)
        self._layout.addRow("ASR model", self._asr_model_combo)

        self._language_edit = QLineEdit()
        self._layout.addRow("Language", self._language_edit)

        self._device_combo = QComboBox()
        self._device_combo.addItems(ASR_DEVICES)
        self._layout.addRow("Device", self._device_combo)

        self._llm_enabled_check = QCheckBox()
        self._layout.addRow("LLM enabled", self._llm_enabled_check)

        self._context_mode_combo = QComboBox()
        self._context_mode_combo.addItems(CONTEXT_MODES)
        self._layout.addRow("Context mode", self._context_mode_combo)

        self._dictionary_enabled_check = QCheckBox()
        self._layout.addRow("Enable dictionary", self._dictionary_enabled_check)

        self._dictionary_learning_check = QCheckBox()
        self._layout.addRow("Enable vocabulary learning", self._dictionary_learning_check)

        self._open_vocab_editor_button = QPushButton("Open vocabulary editor")
        self._open_vocab_editor_button.clicked.connect(self._on_open_vocab_editor)
        self._layout.addRow(self._open_vocab_editor_button)

        self._llm_url_edit = QLineEdit()
        self._layout.addRow("LLM URL", self._llm_url_edit)

        self._llm_model_edit = QLineEdit()
        self._layout.addRow("LLM model", self._llm_model_edit)

        self._llm_timeout_spin = QDoubleSpinBox()
        self._llm_timeout_spin.setRange(0.5, 300.0)
        self._llm_timeout_spin.setDecimals(1)
        self._llm_timeout_spin.setSingleStep(1.0)
        self._layout.addRow("LLM timeout (s)", self._llm_timeout_spin)

        self._llm_prompt_edit = QTextEdit()
        self._layout.addRow("LLM system prompt", self._llm_prompt_edit)

        self._dry_run_check = QCheckBox()
        self._layout.addRow("Dry-run", self._dry_run_check)

        self._vocab_editor: Any | None = None

        self._button_layout = QHBoxLayout()
        self._save_button = QPushButton("Save")
        self._cancel_button = QPushButton("Cancel")
        self._test_llm_button = QPushButton("Test LLM")
        self._test_llm_button.setEnabled(False)
        self._button_layout.addWidget(self._save_button)
        self._button_layout.addWidget(self._cancel_button)
        self._button_layout.addWidget(self._test_llm_button)
        self._layout.addRow(self._button_layout)

        self._save_button.clicked.connect(self._on_save)
        self._cancel_button.clicked.connect(self._on_cancel)

        self._load_values(settings)

    def _load_values(self, settings: Settings) -> None:
        """Populate widgets from a Settings instance.

        Args:
            settings: Settings to display.
        """
        self._hotkey_edit.setText(settings.hotkey)
        self._push_to_talk_check.setChecked(settings.push_to_talk)
        self._asr_model_combo.setCurrentText(settings.asr_model)
        self._language_edit.setText(settings.asr_language)
        self._device_combo.setCurrentText(settings.asr_device)
        self._llm_enabled_check.setChecked(settings.llm_enabled)
        self._context_mode_combo.setCurrentText(settings.context_mode)
        self._dictionary_enabled_check.setChecked(settings.dictionary_enabled)
        self._dictionary_learning_check.setChecked(settings.dictionary_learning_enabled)
        self._llm_url_edit.setText(settings.llm_url)
        self._llm_model_edit.setText(settings.llm_model)
        self._llm_timeout_spin.setValue(settings.llm_timeout)
        self._llm_prompt_edit.setPlainText(settings.llm_prompt)
        self._dry_run_check.setChecked(settings.dry_run)

    def _collect_values(self) -> dict[str, object]:
        """Return a dict of updated settings values from widgets."""
        return {
            "hotkey": self._hotkey_edit.text(),
            "push_to_talk": self._push_to_talk_check.isChecked(),
            "asr_model": self._asr_model_combo.currentText(),
            "asr_language": self._language_edit.text(),
            "asr_device": self._device_combo.currentText(),
            "llm_enabled": self._llm_enabled_check.isChecked(),
            "context_mode": self._context_mode_combo.currentText(),
            "dictionary_enabled": self._dictionary_enabled_check.isChecked(),
            "dictionary_learning_enabled": self._dictionary_learning_check.isChecked(),
            "llm_url": self._llm_url_edit.text(),
            "llm_model": self._llm_model_edit.text(),
            "llm_timeout": self._llm_timeout_spin.value(),
            "llm_prompt": self._llm_prompt_edit.toPlainText(),
            "dry_run": self._dry_run_check.isChecked(),
        }

    def _on_save(self) -> None:
        """Persist settings to ``.env`` and emit the saved signal."""
        data = self.settings.model_dump()
        data.update(self._collect_values())
        self.settings = Settings(**data)
        _write_env_file(self.env_file, self.settings)
        self.settings_saved.emit(self.settings)
        self.hide()

    def _on_cancel(self) -> None:
        """Hide the window without saving changes."""
        self.hide()

    def set_vocab_editor(self, editor: object | None) -> None:
        """Attach a vocabulary editor instance to be shown by this window.

        Args:
            editor: Vocabulary editor dialog (usually ``VocabularyEditor``).
        """
        self._vocab_editor = editor

    def _on_open_vocab_editor(self) -> None:
        """Show the attached vocabulary editor if one has been provided."""
        if self._vocab_editor is not None:
            self._vocab_editor.show()
            self._vocab_editor.raise_()
