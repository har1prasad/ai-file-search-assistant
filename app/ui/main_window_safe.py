"""
main_window.py — Main GUI window for AI File Search Assistant.

Provides a clean, modern PySide6 desktop interface that connects directly
to the backend pipeline: FileIndexer, SearchEngine.

Layout:
    ┌─────────────────────────────────────────────┐
    │  Header: AI File Search Assistant            │
    ├─────────────────────────────────────────────┤
    │  Toolbar: [Index Folder] [Open File]         │
    ├─────────────────────────────────────────────┤
    │  Search: [___query_______________] [Search]  │
    ├───────────────────┬─────────────────────────┤
    │  Results List     │  File Preview            │
    │                   │                          │
    │                   │                          │
    ├───────────────────┴─────────────────────────┤
    │  Status Bar                                  │
    └─────────────────────────────────────────────┘
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QAction, QFont, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QStatusBar,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

# ---------------------------------------------------------------------------
# Backend imports
# ---------------------------------------------------------------------------

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.indexing.file_indexer import FileIndexer
from app.search.search_engine import SearchEngine

# ---------------------------------------------------------------------------
# Worker threads (keep UI responsive during heavy operations)
# ---------------------------------------------------------------------------


class IndexWorker(QThread):
    """Background thread for folder indexing.

    Signals:
        finished (dict): Emitted with the summary dict when indexing completes.
        error    (str):  Emitted with an error message on failure.
    """

    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, indexer: FileIndexer, folder_path: str) -> None:
        super().__init__()
        self.indexer = indexer
        self.folder_path = folder_path

    def run(self) -> None:
        try:
            summary = self.indexer.index_folder(self.folder_path)
            self.finished.emit(summary)
        except Exception as exc:
            self.error.emit(str(exc))


class SearchWorker(QThread):
    """Background thread for semantic search.

    Signals:
        finished (list): Emitted with the results list when search completes.
        error    (str):  Emitted with an error message on failure.
    """

    finished = Signal(list)
    error = Signal(str)

    def __init__(self, engine: SearchEngine, query: str, top_k: int = 10) -> None:
        super().__init__()
        self.engine = engine
        self.query = query
        self.top_k = top_k

    def run(self) -> None:
        try:
            results = self.engine.search(self.query, top_k=self.top_k)
            self.finished.emit(results)
        except Exception as exc:
            self.error.emit(str(exc))


# ---------------------------------------------------------------------------
# MainWindow
# ---------------------------------------------------------------------------


class MainWindow(QMainWindow):
    """Main application window for AI File Search Assistant.

    Connects the PySide6 UI to the backend indexing and search pipeline.
    All heavy operations run in background threads to keep the UI responsive.
    """

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("AI File Search Assistant")
        self.setMinimumSize(1200, 700)

        # Backend components
        self.indexer = FileIndexer()
        self.search_engine = SearchEngine()

        # State
        self._results: list[dict[str, Any]] = []
        self._index_worker: IndexWorker | None = None
        self._search_worker: SearchWorker | None = None

        # Build UI
        self.setup_ui()
        self.connect_signals()
        self.apply_styles()

    # ------------------------------------------------------------------
    # UI Construction
    # ------------------------------------------------------------------

    def setup_ui(self) -> None:
        """Build and arrange all UI components."""
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        root_layout.addWidget(self._create_header())
        root_layout.addWidget(self._create_toolbar())
        root_layout.addWidget(self._create_search_bar())
        root_layout.addWidget(self._create_main_content(), stretch=1)

        self._create_status_bar()

    def _create_header(self) -> QWidget:
        """Create the top header with app title."""
        header = QWidget()
        header.setObjectName("header")
        layout = QHBoxLayout(header)
        layout.setContentsMargins(24, 16, 24, 16)

        title = QLabel("AI File Search Assistant")
        title.setObjectName("appTitle")
        layout.addWidget(title)
        layout.addStretch()

        subtitle = QLabel("Semantic · Local · Private")
        subtitle.setObjectName("appSubtitle")
        layout.addWidget(subtitle)

        return header

    def _create_toolbar(self) -> QWidget:
        """Create the toolbar with Index and Open buttons."""
        toolbar = QWidget()
        toolbar.setObjectName("toolbar")
        layout = QHBoxLayout(toolbar)
        layout.setContentsMargins(24, 10, 24, 10)
        layout.setSpacing(12)

        self.btn_index = QPushButton("⊕  Index Folder")
        self.btn_index.setObjectName("btnPrimary")
        self.btn_index.setToolTip("Select a folder to index for semantic search")

        self.btn_open = QPushButton("↗  Open File")
        self.btn_open.setObjectName("btnSecondary")
        self.btn_open.setToolTip("Open the selected file with your default application")

        layout.addWidget(self.btn_index)
        layout.addWidget(self.btn_open)
        layout.addStretch()

        return toolbar

    def _create_search_bar(self) -> QWidget:
        """Create the search input row."""
        search_bar = QWidget()
        search_bar.setObjectName("searchBar")
        layout = QHBoxLayout(search_bar)
        layout.setContentsMargins(24, 12, 24, 12)
        layout.setSpacing(10)

        self.search_input = QLineEdit()
        self.search_input.setObjectName("searchInput")
        self.search_input.setPlaceholderText(
            "Search your files using natural language..."
        )
        self.search_input.setMinimumHeight(40)

        self.btn_search = QPushButton("Search")
        self.btn_search.setObjectName("btnSearch")
        self.btn_search.setMinimumHeight(40)
        self.btn_search.setMinimumWidth(100)

        layout.addWidget(self.search_input, stretch=1)
        layout.addWidget(self.btn_search)

        return search_bar

    def _create_main_content(self) -> QSplitter:
        """Create the main split view with results list and preview panel."""
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setObjectName("mainSplitter")
        splitter.setHandleWidth(2)

        # Left — Results list
        left = QWidget()
        left.setObjectName("leftPanel")
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(16, 16, 8, 16)
        left_layout.setSpacing(8)

        results_label = QLabel("Results")
        results_label.setObjectName("panelLabel")
        left_layout.addWidget(results_label)

        self.results_list = QListWidget()
        self.results_list.setObjectName("resultsList")
        self.results_list.setAlternatingRowColors(True)
        left_layout.addWidget(self.results_list)

        # Right — Preview panel
        right = QWidget()
        right.setObjectName("rightPanel")
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(8, 16, 16, 16)
        right_layout.setSpacing(8)

        preview_label = QLabel("Preview")
        preview_label.setObjectName("panelLabel")
        right_layout.addWidget(preview_label)

        self.preview_area = QTextEdit()
        self.preview_area.setObjectName("previewArea")
        self.preview_area.setReadOnly(True)
        self.preview_area.setPlaceholderText(
            "Select a result to preview its content..."
        )
        self.preview_area.setFont(QFont("JetBrains Mono, Courier New, monospace", 10))
        right_layout.addWidget(self.preview_area)

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setSizes([380, 820])

        return splitter

    def _create_status_bar(self) -> None:
        """Create the bottom status bar."""
        self.status_bar = QStatusBar()
        self.status_bar.setObjectName("statusBar")
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

    # ------------------------------------------------------------------
    # Signals
    # ------------------------------------------------------------------

    def connect_signals(self) -> None:
        """Wire all UI signals to their handlers."""
        self.btn_index.clicked.connect(self.index_folder)
        self.btn_open.clicked.connect(self.open_selected_file)
        self.btn_search.clicked.connect(self.perform_search)
        self.search_input.returnPressed.connect(self.perform_search)
        self.results_list.currentItemChanged.connect(self.show_preview)
        self.results_list.itemDoubleClicked.connect(self.open_selected_file)

        # Ctrl+F focuses the search bar
        shortcut = QShortcut(QKeySequence("Ctrl+F"), self)
        shortcut.activated.connect(self.search_input.setFocus)

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------

    def index_folder(self) -> None:
        folder_path = QFileDialog.getExistingDirectory(
            self,
            "Select Folder to Index",
            str(Path(__file__).resolve().parent.parent.parent / "Sample_files")
        )
        if not folder_path:
            return

    # def index_folder(self) -> None:
    #     """Open a folder picker and index the selected directory."""
    #     folder_path = QFileDialog.getExistingDirectory(
    #         self, "Select Folder to Index", str(Path.home())
    #     )
    #     if not folder_path:
    #         return

    #     self.status_bar.showMessage(f"Indexing: {folder_path} ...")
    #     self.btn_index.setEnabled(False)

    #     self._index_worker = IndexWorker(self.indexer, folder_path)
    #     self._index_worker.finished.connect(self._on_index_finished)
    #     self._index_worker.error.connect(self._on_index_error)
    #     self._index_worker.start()

    def _on_index_finished(self, summary: dict) -> None:
        """Handle successful indexing completion."""
        self.btn_index.setEnabled(True)
        msg = (
            f"Indexed {summary['indexed']} files  ·  "
            f"Skipped {summary['skipped']}  ·  "
            f"Failed {summary['failed']}"
        )
        self.status_bar.showMessage(msg)

    def _on_index_error(self, error: str) -> None:
        """Handle indexing errors."""
        self.btn_index.setEnabled(True)
        self.status_bar.showMessage("Indexing failed.")
        QMessageBox.critical(self, "Indexing Error", error)

    def perform_search(self) -> None:
        """Read the query and run semantic search in a background thread."""
        query = self.search_input.text().strip()
        if not query:
            return

        self.results_list.clear()
        self.preview_area.clear()
        self._results = []
        self.status_bar.showMessage(f'Searching for "{query}" ...')
        self.btn_search.setEnabled(False)

        self._search_worker = SearchWorker(self.search_engine, query, top_k=10)
        self._search_worker.finished.connect(self._on_search_finished)
        self._search_worker.error.connect(self._on_search_error)
        self._search_worker.start()

    def _on_search_finished(self, results: list[dict]) -> None:
        """Populate the results list with returned search results."""
        self.btn_search.setEnabled(True)
        self._results = results

        if not results:
            self.status_bar.showMessage("No results found.")
            self.results_list.addItem("No results found.")
            return

        for result in results:
            filename = result.get("filename", result.get("file_name", "Unknown"))
            score = result.get("similarity_score", 0.0)
            path = result.get("path", result.get("file_path", ""))

            item = QListWidgetItem(f"{filename}\n  score: {score:.2f}")
            item.setToolTip(path)
            self.results_list.addItem(item)

        self.status_bar.showMessage(f"{len(results)} result(s) found.")

    def _on_search_error(self, error: str) -> None:
        """Handle search errors."""
        self.btn_search.setEnabled(True)
        self.status_bar.showMessage("Search failed.")
        QMessageBox.critical(self, "Search Error", error)

    def show_preview(self) -> None:
        """Display extracted content for the selected result."""
        row = self.results_list.currentRow()
        if row < 0 or row >= len(self._results):
            return

        result = self._results[row]
        content = result.get("content", "").strip()

        if content:
            self.preview_area.setPlainText(content)
        else:
            path = result.get("path", result.get("file_path", ""))
            self.preview_area.setPlainText(
                f"No text content available for preview.\n\nFile: {path}"
            )

    def open_selected_file(self) -> None:
        """Open the selected file using the system's default application."""
        row = self.results_list.currentRow()
        if row < 0 or row >= len(self._results):
            QMessageBox.information(self, "No Selection", "Please select a file first.")
            return

        result = self._results[row]
        path = result.get("path", result.get("file_path", ""))

        if not path or not Path(path).exists():
            QMessageBox.warning(self, "File Not Found", f"Cannot find file:\n{path}")
            return

        try:
            if sys.platform == "win32":
                os.startfile(path)
            elif sys.platform == "darwin":
                subprocess.run(["open", path], check=True)
            else:
                subprocess.run(["xdg-open", path], check=True)
        except Exception as exc:
            QMessageBox.critical(self, "Error Opening File", str(exc))

    # ------------------------------------------------------------------
    # Styles
    # ------------------------------------------------------------------

    def apply_styles(self) -> None:
        """Apply the application stylesheet."""
        self.setStyleSheet("""
            /* ── Global ── */
            QMainWindow, QWidget {
                background-color: #0f1117;
                color: #e2e8f0;
                font-family: "Segoe UI", "SF Pro Display", sans-serif;
                font-size: 13px;
            }

            /* ── Header ── */
            #header {
                background-color: #0a0d14;
                border-bottom: 1px solid #1e2433;
            }
            #appTitle {
                font-size: 18px;
                font-weight: 700;
                color: #f8fafc;
                letter-spacing: 0.3px;
            }
            #appSubtitle {
                font-size: 11px;
                color: #475569;
                letter-spacing: 2px;
                text-transform: uppercase;
            }

            /* ── Toolbar ── */
            #toolbar {
                background-color: #0d1018;
                border-bottom: 1px solid #1e2433;
            }

            /* ── Buttons ── */
            #btnPrimary {
                background-color: #3b82f6;
                color: #ffffff;
                border: none;
                border-radius: 6px;
                padding: 8px 18px;
                font-weight: 600;
                font-size: 13px;
            }
            #btnPrimary:hover {
                background-color: #2563eb;
            }
            #btnPrimary:pressed {
                background-color: #1d4ed8;
            }
            #btnPrimary:disabled {
                background-color: #1e3a5f;
                color: #64748b;
            }

            #btnSecondary {
                background-color: transparent;
                color: #94a3b8;
                border: 1px solid #2d3748;
                border-radius: 6px;
                padding: 8px 18px;
                font-weight: 500;
                font-size: 13px;
            }
            #btnSecondary:hover {
                background-color: #1e2433;
                color: #e2e8f0;
                border-color: #3d4f66;
            }

            #btnSearch {
                background-color: #10b981;
                color: #ffffff;
                border: none;
                border-radius: 6px;
                padding: 8px 24px;
                font-weight: 600;
                font-size: 13px;
            }
            #btnSearch:hover {
                background-color: #059669;
            }
            #btnSearch:pressed {
                background-color: #047857;
            }
            #btnSearch:disabled {
                background-color: #064e3b;
                color: #34d399;
            }

            /* ── Search bar ── */
            #searchBar {
                background-color: #0d1018;
                border-bottom: 1px solid #1e2433;
            }
            #searchInput {
                background-color: #161b27;
                color: #f1f5f9;
                border: 1px solid #2d3748;
                border-radius: 6px;
                padding: 0 14px;
                font-size: 13px;
                selection-background-color: #3b82f6;
            }
            #searchInput:focus {
                border-color: #3b82f6;
                background-color: #1a2035;
            }
            #searchInput::placeholder {
                color: #475569;
            }

            /* ── Panels ── */
            #leftPanel, #rightPanel {
                background-color: #0f1117;
            }
            #panelLabel {
                font-size: 11px;
                font-weight: 600;
                color: #475569;
                letter-spacing: 1.5px;
                text-transform: uppercase;
                padding: 0 4px 4px 4px;
            }

            /* ── Results list ── */
            #resultsList {
                background-color: #0d1018;
                border: 1px solid #1e2433;
                border-radius: 6px;
                alternate-background-color: #111520;
                outline: none;
            }
            #resultsList::item {
                padding: 10px 14px;
                border-bottom: 1px solid #1a2030;
                color: #cbd5e1;
                font-size: 13px;
                line-height: 1.5;
            }
            #resultsList::item:selected {
                background-color: #1e3a5f;
                color: #f1f5f9;
                border-left: 3px solid #3b82f6;
            }
            #resultsList::item:hover:!selected {
                background-color: #161b27;
            }

            /* ── Preview area ── */
            #previewArea {
                background-color: #0d1018;
                color: #94a3b8;
                border: 1px solid #1e2433;
                border-radius: 6px;
                padding: 12px;
                font-family: "JetBrains Mono", "Cascadia Code", "Courier New", monospace;
                font-size: 12px;
                line-height: 1.6;
                selection-background-color: #1e3a5f;
            }

            /* ── Splitter ── */
            #mainSplitter::handle {
                background-color: #1e2433;
                width: 2px;
            }

            /* ── Status bar ── */
            #statusBar {
                background-color: #0a0d14;
                color: #475569;
                border-top: 1px solid #1e2433;
                font-size: 12px;
                padding: 0 16px;
            }

            /* ── Scrollbars ── */
            QScrollBar:vertical {
                background: #0d1018;
                width: 8px;
                border-radius: 4px;
            }
            QScrollBar::handle:vertical {
                background: #2d3748;
                border-radius: 4px;
                min-height: 30px;
            }
            QScrollBar::handle:vertical:hover {
                background: #3d4f66;
            }
            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical {
                height: 0px;
            }
            QScrollBar:horizontal {
                background: #0d1018;
                height: 8px;
                border-radius: 4px;
            }
            QScrollBar::handle:horizontal {
                background: #2d3748;
                border-radius: 4px;
                min-width: 30px;
            }
            QScrollBar::handle:horizontal:hover {
                background: #3d4f66;
            }
            QScrollBar::add-line:horizontal,
            QScrollBar::sub-line:horizontal {
                width: 0px;
            }
        """)