"""Vocabulary editor dialog for managing user dictionary entries."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QCloseEvent
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QHBoxLayout,
    QInputDialog,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from src.dictionary.base import VocabularySource

if TYPE_CHECKING:  # pragma: no cover
    from src.dictionary.base import DictionaryEntry
    from src.dictionary.vocab_manager import VocabularyManager


class VocabularyEditor(QDialog):
    """Dialog for viewing and editing vocabulary entries."""

    _COLUMNS = ["Term", "Replacement", "Source", "Count"]

    def __init__(self, vocab_manager: VocabularyManager, parent: QWidget | None = None) -> None:
        """Initialize the editor with a vocabulary manager.

        Args:
            vocab_manager: Manager that owns the loaded vocabulary.
            parent: Optional parent widget.
        """
        super().__init__(parent)
        self._vocab_manager = vocab_manager

        self.setWindowTitle("Vocabulary Editor")
        self.resize(600, 400)

        layout = QVBoxLayout(self)

        self._table = QTableWidget()
        self._table.setColumnCount(len(self._COLUMNS))
        self._table.setHorizontalHeaderLabels(self._COLUMNS)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        layout.addWidget(self._table)

        button_layout = QHBoxLayout()
        self._add_button = QPushButton("Add")
        self._edit_button = QPushButton("Edit")
        self._remove_button = QPushButton("Remove")
        self._save_button = QPushButton("Save")
        self._close_button = QPushButton("Close")
        button_layout.addWidget(self._add_button)
        button_layout.addWidget(self._edit_button)
        button_layout.addWidget(self._remove_button)
        button_layout.addStretch()
        button_layout.addWidget(self._save_button)
        button_layout.addWidget(self._close_button)
        layout.addLayout(button_layout)

        self._add_button.clicked.connect(self._on_add)
        self._edit_button.clicked.connect(self._on_edit)
        self._remove_button.clicked.connect(self._on_remove)
        self._save_button.clicked.connect(self._on_save)
        self._close_button.clicked.connect(self.hide)

        self._load_entries()

    def _load_entries(self) -> None:
        """Populate the table from the vocabulary manager."""
        entries = self._vocab_manager.get_all_entries()
        self._table.setRowCount(len(entries))
        for row, entry in enumerate(entries):
            self._table.setItem(row, 0, self._create_item(entry.term))
            self._table.setItem(row, 1, self._create_item(entry.replacement))
            self._table.setItem(row, 2, self._create_item(entry.source.value))
            self._table.setItem(row, 3, self._create_item(str(entry.count)))

    @staticmethod
    def _create_item(text: str) -> QTableWidgetItem:
        """Create a read-only table item."""
        item = QTableWidgetItem(text)
        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        return item

    def _selected_entry(self) -> tuple[int, DictionaryEntry] | None:
        """Return the selected row index and entry, if any."""
        row = self._table.currentRow()
        if row < 0:
            return None
        term_item = self._table.item(row, 0)
        replacement_item = self._table.item(row, 1)
        source_item = self._table.item(row, 2)
        count_item = self._table.item(row, 3)
        if (
            term_item is None
            or replacement_item is None
            or source_item is None
            or count_item is None
        ):
            return None
        term = term_item.text()
        replacement = replacement_item.text()
        source_text = source_item.text()
        count_text = count_item.text()
        from src.dictionary.base import DictionaryEntry

        return row, DictionaryEntry(
            term=term,
            replacement=replacement,
            source=VocabularySource(source_text),
            count=int(count_text),
        )

    def _on_add(self) -> None:
        """Add a new user term after prompting for term and replacement."""
        term, ok = QInputDialog.getText(self, "Add term", "Term:")
        if not ok or not term.strip():
            return
        replacement, ok = QInputDialog.getText(self, "Add term", "Replacement:")
        if not ok:
            return
        self._vocab_manager.add_user_term(term.strip(), replacement.strip() or term.strip())
        self._load_entries()

    def _on_edit(self) -> None:
        """Edit the selected user term."""
        selection = self._selected_entry()
        if selection is None:
            return
        _row, entry = selection
        if entry.source != VocabularySource.user:
            QMessageBox.information(
                self,
                "Cannot edit",
                "Only user-defined entries can be edited.",
            )
            return

        new_term, ok = QInputDialog.getText(
            self,
            "Edit term",
            "Term:",
            text=entry.term,
        )
        if not ok or not new_term.strip():
            return
        new_replacement, ok = QInputDialog.getText(
            self,
            "Edit term",
            "Replacement:",
            text=entry.replacement,
        )
        if not ok:
            return

        if new_term.strip() != entry.term:
            self._vocab_manager.remove_user_term(entry.term)
        self._vocab_manager.add_user_term(
            new_term.strip(),
            new_replacement.strip() or new_term.strip(),
        )
        self._load_entries()

    def _on_remove(self) -> None:
        """Remove the selected user term."""
        selection = self._selected_entry()
        if selection is None:
            return
        _row, entry = selection
        if entry.source != VocabularySource.user:
            QMessageBox.information(
                self,
                "Cannot remove",
                "Only user-defined entries can be removed.",
            )
            return
        self._vocab_manager.remove_user_term(entry.term)
        self._load_entries()

    def _on_save(self) -> None:
        """Persist user vocabulary and reload all sources."""
        self._vocab_manager._save_user()
        self._vocab_manager.load_all()
        self._load_entries()

    def closeEvent(self, event: QCloseEvent | None) -> None:  # noqa: N802
        """Override close to hide the dialog so it can be reused."""
        if event is None:
            return
        event.ignore()
        self.hide()
