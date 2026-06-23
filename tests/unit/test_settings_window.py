"""Unit tests for the settings window."""

# ruff: noqa: N802,D101,D102,D107,N815

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.config import Settings


class FakeSignal:
    """Standalone fake for PyQt6 signals."""

    def __init__(self, *types: object) -> None:
        self._slots: list[object] = []

    def connect(self, slot: object) -> None:
        self._slots.append(slot)

    def emit(self, *args: object) -> None:
        for slot in self._slots:
            if callable(slot):
                slot(*args)


class FakeQWidget:
    def __init__(self, parent: object | None = None) -> None:
        self._window_title = ""
        self._shown = False
        self._hidden = False
        self._tool_tip = ""

    def setWindowTitle(self, title: str) -> None:
        self._window_title = title

    def setToolTip(self, tip: str) -> None:
        self._tool_tip = tip

    def show(self) -> None:
        self._shown = True

    def hide(self) -> None:
        self._hidden = True


class FakeQFormLayout:
    def __init__(self, parent: object | None = None) -> None:
        self.rows: list[tuple[object, object | None]] = []

    def addRow(self, label_or_layout: object, widget: object | None = None) -> None:
        self.rows.append((label_or_layout, widget))


class FakeQHBoxLayout:
    def __init__(self) -> None:
        self.widgets: list[object] = []

    def addWidget(self, widget: object) -> None:
        self.widgets.append(widget)


class FakeQLineEdit:
    def __init__(self) -> None:
        self._text = ""

    def setText(self, text: str) -> None:
        self._text = str(text)

    def text(self) -> str:
        return self._text


class FakeQCheckBox:
    def __init__(self) -> None:
        self._checked = False
        self._enabled = True

    def setChecked(self, checked: bool) -> None:
        self._checked = bool(checked)

    def isChecked(self) -> bool:
        return self._checked

    def setEnabled(self, enabled: bool) -> None:
        self._enabled = enabled

    def setToolTip(self, tip: str) -> None:
        pass


class FakeQComboBox:
    def __init__(self) -> None:
        self._items: list[str] = []
        self._current = ""

    def addItems(self, items: list[str]) -> None:
        self._items = list(items)

    def setCurrentText(self, text: str) -> None:
        self._current = text

    def currentText(self) -> str:
        return self._current


class FakeQDoubleSpinBox:
    def __init__(self) -> None:
        self._value = 0.0
        self._min = 0.0
        self._max = 99.0
        self._decimals = 0

    def setRange(self, min_: float, max_: float) -> None:
        self._min = min_
        self._max = max_

    def setDecimals(self, decimals: int) -> None:
        self._decimals = decimals

    def setSingleStep(self, step: float) -> None:
        pass

    def setValue(self, value: float) -> None:
        self._value = value

    def value(self) -> float:
        return self._value


class FakeQSpinBox:
    def __init__(self) -> None:
        self._value = 0
        self._min = 0
        self._max = 99

    def setRange(self, min_: int, max_: int) -> None:
        self._min = min_
        self._max = max_

    def setValue(self, value: int) -> None:
        self._value = value

    def value(self) -> int:
        return self._value


class FakeQGroupBox(FakeQWidget):
    def __init__(self, title: str = "") -> None:
        super().__init__()
        self._title = title
        self._layout: object | None = None

    def setLayout(self, layout: object) -> None:
        self._layout = layout


class FakeQTextEdit:
    def __init__(self) -> None:
        self._text = ""

    def setPlainText(self, text: str) -> None:
        self._text = str(text)

    def toPlainText(self) -> str:
        return self._text


class FakeQPushButton:
    def __init__(self, text: str = "") -> None:
        self.text = text
        self._enabled = True
        self.clicked = FakeSignal()

    def setEnabled(self, enabled: bool) -> None:
        self._enabled = enabled


class FakeQtWidgetsModule:
    QWidget = FakeQWidget
    QFormLayout = FakeQFormLayout
    QHBoxLayout = FakeQHBoxLayout
    QLineEdit = FakeQLineEdit
    QCheckBox = FakeQCheckBox
    QComboBox = FakeQComboBox
    QDoubleSpinBox = FakeQDoubleSpinBox
    QSpinBox = FakeQSpinBox
    QGroupBox = FakeQGroupBox
    QTextEdit = FakeQTextEdit
    QPushButton = FakeQPushButton


class FakeQtCoreModule:
    pyqtSignal = FakeSignal


@pytest.fixture
def mock_qt():
    """Patch PyQt6 modules with lightweight fakes."""
    modules = {
        "PyQt6.QtWidgets": FakeQtWidgetsModule(),
        "PyQt6.QtCore": FakeQtCoreModule(),
    }
    with patch.dict("sys.modules", modules):
        yield modules


@pytest.fixture
def settings_window(mock_qt, tmp_path):
    """Return a SettingsWindow instance built against mocked Qt objects."""
    import sys

    sys.modules.pop("src.ui.settings_window", None)
    sys.modules.pop("src.ui", None)

    settings = Settings(
        hotkey="f10",
        push_to_talk=False,
        asr_model="small",
        asr_language="fr",
        asr_device="cuda",
        llm_enabled=True,
        llm_url="http://localhost:1234",
        llm_model="mistral",
        llm_timeout=10.0,
        llm_prompt="Fix this.\nKeep it short.",
        dry_run=True,
        streaming_enabled=True,
        streaming_chunk_seconds=2.5,
        asr_streaming_beam_size=3,
        asr_warmup_at_start=False,
        final_transcription_enabled=False,
        replace_streaming_with_final=True,
        data_dir=tmp_path / "data",
        models_dir=tmp_path / "models",
    )
    env_file = tmp_path / "test.env"

    from src.ui.settings_window import SettingsWindow

    window = SettingsWindow(settings, env_file=env_file)
    return window


def test_window_loads_settings_into_widgets(settings_window) -> None:
    """Widgets should reflect the Settings values passed at construction."""
    assert settings_window._hotkey_edit.text() == "f10"
    assert settings_window._push_to_talk_check.isChecked() is False
    assert settings_window._asr_model_combo.currentText() == "small"
    assert settings_window._language_edit.text() == "fr"
    assert settings_window._device_combo.currentText() == "cuda"
    assert settings_window._llm_enabled_check.isChecked() is True
    assert settings_window._context_mode_combo.currentText() == "general"
    assert settings_window._dictionary_enabled_check.isChecked() is True
    assert settings_window._dictionary_learning_check.isChecked() is False
    assert settings_window._llm_url_edit.text() == "http://localhost:1234"
    assert settings_window._llm_model_edit.text() == "mistral"
    assert settings_window._llm_timeout_spin.value() == 10.0
    assert settings_window._llm_prompt_edit.toPlainText() == "Fix this.\nKeep it short."
    assert settings_window._dry_run_check.isChecked() is True
    assert settings_window._streaming_enabled_check.isChecked() is True
    assert settings_window._streaming_chunk_spin.value() == 2.5
    assert settings_window._asr_streaming_beam_size_spin.value() == 3
    assert settings_window._asr_warmup_check.isChecked() is False
    assert settings_window._final_transcription_enabled_check.isChecked() is False
    assert settings_window._replace_streaming_with_final_check.isChecked() is True


def test_save_writes_env_file_with_expected_values(settings_window, tmp_path) -> None:
    """Clicking Save should persist collected widget values to the .env file."""
    settings_window._hotkey_edit.setText("ctrl+f9")
    settings_window._push_to_talk_check.setChecked(True)
    settings_window._asr_model_combo.setCurrentText("tiny")
    settings_window._language_edit.setText("de")
    settings_window._device_combo.setCurrentText("cpu")
    settings_window._llm_enabled_check.setChecked(False)
    settings_window._context_mode_combo.setCurrentText("code")
    settings_window._dictionary_enabled_check.setChecked(False)
    settings_window._dictionary_learning_check.setChecked(True)
    settings_window._llm_url_edit.setText("http://ollama:11434")
    settings_window._llm_model_edit.setText("llama3.1")
    settings_window._llm_timeout_spin.setValue(3.5)
    settings_window._llm_prompt_edit.setPlainText("New prompt.\nTwo lines.")
    settings_window._dry_run_check.setChecked(False)
    settings_window._streaming_enabled_check.setChecked(False)
    settings_window._streaming_chunk_spin.setValue(1.0)
    settings_window._asr_streaming_beam_size_spin.setValue(5)
    settings_window._asr_warmup_check.setChecked(True)
    settings_window._final_transcription_enabled_check.setChecked(True)
    settings_window._replace_streaming_with_final_check.setChecked(False)

    signal_handler = MagicMock()
    settings_window.settings_saved.connect(signal_handler)

    settings_window._save_button.clicked.emit()

    assert settings_window._hidden is True
    env_text = settings_window.env_file.read_text(encoding="utf-8")

    assert "HOTKEY=ctrl+f9" in env_text
    assert "PUSH_TO_TALK=true" in env_text
    assert "ASR_MODEL=tiny" in env_text
    assert "ASR_LANGUAGE=de" in env_text
    assert "ASR_DEVICE=cpu" in env_text
    assert "LLM_ENABLED=false" in env_text
    assert "CONTEXT_MODE=code" in env_text
    assert "DICTIONARY_ENABLED=false" in env_text
    assert "DICTIONARY_LEARNING_ENABLED=true" in env_text
    assert "LLM_URL=http://ollama:11434" in env_text
    assert "LLM_MODEL=llama3.1" in env_text
    assert "LLM_TIMEOUT=3.5" in env_text
    assert 'LLM_PROMPT="New prompt.\\nTwo lines."' in env_text
    assert "DRY_RUN=false" in env_text
    assert "STREAMING_ENABLED=false" in env_text
    assert "STREAMING_CHUNK_SECONDS=1.0" in env_text
    assert "ASR_STREAMING_BEAM_SIZE=5" in env_text
    assert "ASR_WARMUP_AT_START=true" in env_text
    assert "FINAL_TRANSCRIPTION_ENABLED=true" in env_text
    assert "REPLACE_STREAMING_WITH_FINAL=false" in env_text

    signal_handler.assert_called_once()
    saved_settings = signal_handler.call_args[0][0]
    assert isinstance(saved_settings, Settings)
    assert saved_settings.hotkey == "ctrl+f9"
    assert saved_settings.asr_model == "tiny"
    assert saved_settings.llm_prompt == "New prompt.\nTwo lines."
    assert saved_settings.streaming_enabled is False
    assert saved_settings.streaming_chunk_seconds == 1.0
    assert saved_settings.asr_streaming_beam_size == 5
    assert saved_settings.asr_warmup_at_start is True
    assert saved_settings.final_transcription_enabled is True
    assert saved_settings.replace_streaming_with_final is False


def test_save_preserves_fields_not_in_ui(settings_window, tmp_path) -> None:
    """Fields without widgets are carried over from the original settings."""
    settings_window._save_button.clicked.emit()

    env_text = settings_window.env_file.read_text(encoding="utf-8")
    assert "AUDIO_SAMPLE_RATE=16000" in env_text
    assert "AUDIO_CHANNELS=1" in env_text
    assert "VAD_AGGRESSIVENESS=1" in env_text


def test_cancel_does_not_write_file(settings_window) -> None:
    """Clicking Cancel should hide the window without writing the .env file."""
    settings_window._hotkey_edit.setText("f8")

    signal_handler = MagicMock()
    settings_window.settings_saved.connect(signal_handler)

    settings_window._cancel_button.clicked.emit()

    assert settings_window._hidden is True
    assert settings_window.env_file.exists() is False
    signal_handler.assert_not_called()


def test_test_llm_button_is_disabled(settings_window) -> None:
    """The Test LLM button should be present but disabled."""
    assert settings_window._test_llm_button.text == "Test LLM"
    assert settings_window._test_llm_button._enabled is False


def test_vocab_editor_button_is_present(settings_window) -> None:
    """The settings window should have an Open vocabulary editor button."""
    assert settings_window._open_vocab_editor_button.text == "Open vocabulary editor"


def test_set_vocab_editor_and_open_shows_editor(settings_window) -> None:
    """set_vocab_editor should attach an editor shown by the open button."""
    fake_editor = MagicMock()
    settings_window.set_vocab_editor(fake_editor)

    settings_window._open_vocab_editor_button.clicked.emit()

    fake_editor.show.assert_called_once()
    fake_editor.raise_.assert_called_once()


def test_open_vocab_editor_without_editor_is_safe(settings_window) -> None:
    """Clicking the vocab editor button without an attached editor should not raise."""
    settings_window.set_vocab_editor(None)

    settings_window._open_vocab_editor_button.clicked.emit()


def test_serialize_env_value_quotes_special_characters() -> None:
    """Values with spaces, newlines, or equals signs are quoted for python-dotenv."""
    from src.ui.settings_window import _serialize_env_value

    assert _serialize_env_value("KEY", "hello world") == 'KEY="hello world"'
    assert _serialize_env_value("KEY", "a=b") == 'KEY="a=b"'
    assert _serialize_env_value("KEY", "line1\nline2") == 'KEY="line1\\nline2"'
    assert _serialize_env_value("KEY", True) == "KEY=true"
    assert _serialize_env_value("KEY", Path("data")) == "KEY=data"
