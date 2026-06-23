"""Unit tests for the vocabulary editor dialog."""

# ruff: noqa: N802, N815, D101, D102, D107

from __future__ import annotations

from collections import deque
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.dictionary.base import DictionaryEntry, VocabularySource


class _FakeSelectionBehavior:
    SelectRows = "rows"


class _FakeSelectionMode:
    SingleSelection = "single"


class _FakeAbstractItemView:
    SelectionBehavior = _FakeSelectionBehavior
    SelectionMode = _FakeSelectionMode


class _FakeItemFlag:
    ItemIsEditable = 2


class _FakeQt:
    ItemFlag = _FakeItemFlag


class _FakeCloseEvent:
    def __init__(self) -> None:
        self._ignored = False

    def ignore(self) -> None:
        self._ignored = True


class _FakeSignal:
    def __init__(self) -> None:
        self._slots: list[object] = []

    def connect(self, slot: object) -> None:
        self._slots.append(slot)

    def emit(self, *args: object) -> None:
        for slot in self._slots:
            if callable(slot):
                slot(*args)


class _FakeTableWidgetItem:
    def __init__(self, text: str) -> None:
        self._text = text
        self._flags = 0

    def setFlags(self, flags: int) -> None:
        self._flags = flags

    def flags(self) -> int:
        return self._flags

    def text(self) -> str:
        return self._text


class _FakeTableWidget:
    def __init__(self) -> None:
        self._row_count = 0
        self._column_count = 0
        self._header_labels: list[str] = []
        self._items: dict[tuple[int, int], _FakeTableWidgetItem] = {}
        self._current_row = -1

    def setColumnCount(self, count: int) -> None:
        self._column_count = count

    def setHorizontalHeaderLabels(self, labels: list[str]) -> None:
        self._header_labels = list(labels)

    def setSelectionBehavior(self, behavior: object) -> None:
        pass

    def setSelectionMode(self, mode: object) -> None:
        pass

    def setRowCount(self, rows: int) -> None:
        self._row_count = rows

    def setItem(self, row: int, column: int, item: _FakeTableWidgetItem) -> None:
        self._items[(row, column)] = item

    def item(self, row: int, column: int) -> _FakeTableWidgetItem | None:
        return self._items.get((row, column))

    def currentRow(self) -> int:
        return self._current_row


class _FakeInputDialog:
    _responses: deque[tuple[str, bool]] = deque()

    @staticmethod
    def getText(
        parent: object,
        title: str,
        label: str,
        *,
        text: str = "",
    ) -> tuple[str, bool]:
        return _FakeInputDialog._responses.popleft()


class _FakeMessageBox:
    _calls: list[tuple[object, ...]] = []

    @staticmethod
    def information(parent: object, title: str, message: str) -> None:
        _FakeMessageBox._calls.append((parent, title, message))


class _FakePushButton:
    def __init__(self, text: str = "") -> None:
        self.text = text
        self.clicked = _FakeSignal()


class _FakeLayout:
    def __init__(self, parent: object | None = None) -> None:
        self._widgets: list[object] = []
        self._layouts: list[object] = []

    def addWidget(self, widget: object) -> None:
        self._widgets.append(widget)

    def addLayout(self, layout: object) -> None:
        self._layouts.append(layout)

    def addStretch(self) -> None:
        pass


class _FakeVBoxLayout(_FakeLayout):
    pass


class _FakeHBoxLayout(_FakeLayout):
    pass


class _FakeDialog:
    def __init__(self, parent: object | None = None) -> None:
        self._parent = parent
        self._title = ""
        self._shown = False
        self._hidden = False

    def setWindowTitle(self, title: str) -> None:
        self._title = title

    def resize(self, width: int, height: int) -> None:
        pass

    def show(self) -> None:
        self._shown = True

    def hide(self) -> None:
        self._hidden = True

    def raise_(self) -> None:
        pass


class _FakeQtWidgetsModule:
    QDialog = _FakeDialog
    QWidget = _FakeDialog
    QVBoxLayout = _FakeVBoxLayout
    QHBoxLayout = _FakeHBoxLayout
    QTableWidget = _FakeTableWidget
    QTableWidgetItem = _FakeTableWidgetItem
    QPushButton = _FakePushButton
    QInputDialog = _FakeInputDialog
    QMessageBox = _FakeMessageBox
    QAbstractItemView = _FakeAbstractItemView


class _FakeQtCoreModule:
    Qt = _FakeQt


class _FakeQtGuiModule:
    QCloseEvent = _FakeCloseEvent


@pytest.fixture
def mock_qt():
    """Patch PyQt6 modules with lightweight fakes for the vocab editor tests."""
    modules = {
        "PyQt6.QtWidgets": _FakeQtWidgetsModule(),
        "PyQt6.QtCore": _FakeQtCoreModule(),
        "PyQt6.QtGui": _FakeQtGuiModule(),
    }
    with patch.dict("sys.modules", modules):
        yield modules


@pytest.fixture
def fake_manager():
    """Return a fake vocabulary manager preloaded with sample entries."""
    manager = MagicMock()
    manager.get_all_entries.return_value = [
        DictionaryEntry(term="alpha", replacement="A", source=VocabularySource.static, count=1),
        DictionaryEntry(term="beta", replacement="B", source=VocabularySource.user, count=3),
    ]
    return manager


@pytest.fixture
def editor(mock_qt, fake_manager):
    """Return a VocabularyEditor instance built against mocked Qt objects."""
    import sys
    import types

    sys.modules.pop("src.ui.vocab_editor", None)

    # Stub the parent package so importing the submodule does not execute
    # src/ui/__init__.py (which would pull in unrelated Qt-dependent modules).
    stub_ui = types.ModuleType("src.ui")
    stub_ui.__path__ = [str(Path("src/ui").resolve())]

    with patch.dict("sys.modules", {"src.ui": stub_ui}):
        from src.ui.vocab_editor import VocabularyEditor

        _FakeInputDialog._responses.clear()
        _FakeMessageBox._calls.clear()

        yield VocabularyEditor(fake_manager)


def test_editor_loads_entries_into_table(editor, fake_manager) -> None:
    """The table should display entries returned by the vocabulary manager."""
    fake_manager.get_all_entries.assert_called_once()

    assert editor._table._column_count == 4
    assert editor._table._header_labels == ["Term", "Replacement", "Source", "Count"]
    assert editor._table._row_count == 2
    assert editor._table.item(0, 0).text() == "alpha"
    assert editor._table.item(0, 2).text() == "static"
    assert editor._table.item(1, 3).text() == "3"


def test_add_button_prompts_and_adds_user_term(editor, fake_manager) -> None:
    """Clicking Add should prompt for term/replacement and add a user term."""
    _FakeInputDialog._responses.extend([("acme", True), ("ACME", True)])

    editor._add_button.clicked.emit()

    fake_manager.add_user_term.assert_called_once_with("acme", "ACME")
    assert fake_manager.get_all_entries.call_count == 2


def test_add_button_aborts_when_term_dialog_cancelled(editor, fake_manager) -> None:
    """If the term prompt is cancelled, no user term should be added."""
    _FakeInputDialog._responses.extend([("", False)])

    editor._add_button.clicked.emit()

    fake_manager.add_user_term.assert_not_called()


def test_remove_button_deletes_selected_user_term(editor, fake_manager) -> None:
    """Clicking Remove should delete the selected user-defined entry."""
    editor._table._current_row = 1

    editor._remove_button.clicked.emit()

    fake_manager.remove_user_term.assert_called_once_with("beta")
    assert fake_manager.get_all_entries.call_count == 2


def test_remove_button_ignores_non_user_entries(editor, fake_manager) -> None:
    """Removing a static or context entry should show an info message."""
    editor._table._current_row = 0

    editor._remove_button.clicked.emit()

    fake_manager.remove_user_term.assert_not_called()
    assert any(call[1] == "Cannot remove" for call in _FakeMessageBox._calls)


def test_edit_button_updates_user_term(editor, fake_manager) -> None:
    """Clicking Edit should update the selected user term."""
    editor._table._current_row = 1
    _FakeInputDialog._responses.extend([("beta2", True), ("B2", True)])

    editor._edit_button.clicked.emit()

    fake_manager.remove_user_term.assert_called_once_with("beta")
    fake_manager.add_user_term.assert_called_once_with("beta2", "B2")


def test_edit_button_aborts_on_cancel(editor, fake_manager) -> None:
    """If the edit prompt is cancelled, the user term should not change."""
    editor._table._current_row = 1
    _FakeInputDialog._responses.extend([("beta", False)])

    editor._edit_button.clicked.emit()

    fake_manager.remove_user_term.assert_not_called()
    fake_manager.add_user_term.assert_not_called()


def test_save_button_persists_and_reloads(editor, fake_manager) -> None:
    """Clicking Save should persist user vocabulary and reload all sources."""
    editor._save_button.clicked.emit()

    fake_manager._save_user.assert_called_once()
    fake_manager.load_all.assert_called_once()
    assert fake_manager.get_all_entries.call_count == 2


def test_close_button_hides_dialog(editor) -> None:
    """Clicking Close should hide the dialog instead of destroying it."""
    editor._close_button.clicked.emit()

    assert editor._hidden is True


def test_close_event_ignores_and_hides(editor) -> None:
    """The window close button should hide the reusable dialog."""
    event = _FakeCloseEvent()
    editor.closeEvent(event)

    assert event._ignored is True
    assert editor._hidden is True
