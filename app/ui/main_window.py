"""
main_window.py — Main GUI window for AI File Search Assistant.

Provides a sleek, modern, frameless dark-mode PySide6 desktop interface
with advanced UI components, glassmorphism-inspired styles, custom title bar,
interactive sidebar stats, custom result cards, and robust thread integration.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from PySide6.QtCore import QPoint, Qt, QThread, Signal
from PySide6.QtGui import QAction, QFont, QKeySequence, QShortcut, QIcon
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.indexing.file_indexer import FileIndexer
from app.search.search_engine import SearchEngine

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Path Resolver (cross-platform helper)
# ---------------------------------------------------------------------------

def resolve_file_path(path_str: str) -> str:
    """Resolve absolute or relative paths from database to the current machine.

    Handles Unix paths stored in database when running on Windows.
    """
    path = Path(path_str)
    if path.exists():
        return str(path.resolve())

    # Check if we can locate Sample_files in the project root
    parts = path.parts
    for i, part in enumerate(parts):
        if part == "Sample_files":
            rel_path = Path(*parts[i:])
            project_root = Path(__file__).resolve().parent.parent.parent
            resolved = project_root / rel_path
            if resolved.exists():
                return str(resolved.resolve())

    # Fallback to recursively searching the filename in the project directory
    project_root = Path(__file__).resolve().parent.parent.parent
    for p in project_root.rglob(path.name):
        if p.is_file():
            return str(p.resolve())

    return path_str


# ---------------------------------------------------------------------------
# Worker threads (keep UI responsive)
# ---------------------------------------------------------------------------

class IndexWorker(QThread):
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
    finished = Signal(list)
    error = Signal(str)

    def __init__(self, engine: SearchEngine, query: str, top_k: int = 20) -> None:
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
# Custom UI Widgets
# ---------------------------------------------------------------------------

class TitleBar(QWidget):
    """Custom TitleBar widget to implement a frameless window layout."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("titleBar")
        self.setFixedHeight(48)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 0, 16, 0)
        layout.setSpacing(10)

        # App Logo
        self.logo = QLabel("⚡")
        self.logo.setStyleSheet("font-size: 16px; color: #8b5cf6;")
        layout.addWidget(self.logo)

        # App Title
        self.title = QLabel("AI File Search Assistant")
        self.title.setStyleSheet("font-weight: 700; font-size: 13px; color: #ffffff; letter-spacing: 0.2px;")
        layout.addWidget(self.title)

        layout.addStretch()

        # Window Action Controls
        self.btn_min = QPushButton("—")
        self.btn_max = QPushButton("⛶")
        self.btn_close = QPushButton("✕")

        for btn in (self.btn_min, self.btn_max, self.btn_close):
            btn.setFixedSize(32, 32)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setStyleSheet("""
                QPushButton {
                    background-color: transparent;
                    color: #a1a1aa;
                    border: none;
                    border-radius: 6px;
                    font-size: 11px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #27272a;
                    color: #ffffff;
                }
            """)

        self.btn_close.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                color: #a1a1aa;
                border: none;
                border-radius: 6px;
                font-size: 11px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #ef4444;
                color: #ffffff;
            }
        """)

        layout.addWidget(self.btn_min)
        layout.addWidget(self.btn_max)
        layout.addWidget(self.btn_close)

        self.btn_min.clicked.connect(lambda: self.window().showMinimized())
        self.btn_max.clicked.connect(self.toggle_maximized)
        self.btn_close.clicked.connect(lambda: self.window().close())

        self._drag_position = None

    def toggle_maximized(self):
        if self.window().isMaximized():
            self.window().showNormal()
            self.btn_max.setText("⛶")
        else:
            self.window().showMaximized()
            self.btn_max.setText("❐")

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_position = event.globalPosition().toPoint() - self.window().frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        if self._drag_position is not None and event.buttons() == Qt.MouseButton.LeftButton:
            self.window().move(event.globalPosition().toPoint() - self._drag_position)
            event.accept()

    def mouseReleaseEvent(self, event):
        self._drag_position = None
        event.accept()


class ResultCard(QWidget):
    """Custom premium widget card for search result list items."""

    def __init__(self, filename: str, path: str, score: float, size: int, ext: str, modified_time: str, parent=None):
        super().__init__(parent)
        self.setObjectName("resultCard")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(8)

        # Header Row
        header_layout = QHBoxLayout()
        self.lbl_filename = QLabel(filename)
        self.lbl_filename.setStyleSheet("font-weight: 700; font-size: 14px; color: #ffffff;")
        self.lbl_filename.setWordWrap(True)

        # Match Score Badge
        self.lbl_score = QLabel(f"{int(score * 100)}% Match")
        score_color = "#10b981" if score >= 0.75 else "#eab308" if score >= 0.5 else "#f97316"
        self.lbl_score.setStyleSheet(f"""
            background-color: {score_color}1a;
            color: {score_color};
            border: 1px solid {score_color}33;
            border-radius: 6px;
            padding: 3px 8px;
            font-size: 10px;
            font-weight: 700;
        """)

        header_layout.addWidget(self.lbl_filename, stretch=1)
        header_layout.addWidget(self.lbl_score)
        layout.addLayout(header_layout)

        # Path description
        resolved_path = resolve_file_path(path)
        self.lbl_path = QLabel(resolved_path)
        self.lbl_path.setStyleSheet("color: #71717a; font-size: 11px;")
        self.lbl_path.setWordWrap(True)
        layout.addWidget(self.lbl_path)

        # Metadata Footer Row
        footer_layout = QHBoxLayout()
        footer_layout.setSpacing(10)

        # Extension Badge
        clean_ext = ext.replace(".", "").lower()
        self.lbl_ext = QLabel(clean_ext.upper())
        ext_colors = {
            "pdf": ("#ef4444", "#fef2f2"),
            "docx": ("#3b82f6", "#eff6ff"),
            "csv": ("#10b981", "#ecfdf5"),
            "md": ("#a855f7", "#faf5ff"),
            "txt": ("#71717a", "#fafafa"),
            "py": ("#eab308", "#fef9c3"),
            "js": ("#eab308", "#fef9c3"),
            "ts": ("#2563eb", "#eff6ff"),
        }
        fg_color = ext_colors.get(clean_ext, ("#a1a1aa", "#27272a"))[0]
        self.lbl_ext.setStyleSheet(f"""
            background-color: {fg_color}15;
            color: {fg_color};
            border: 1px solid {fg_color}30;
            border-radius: 4px;
            padding: 2px 6px;
            font-size: 9px;
            font-weight: 800;
        """)

        # Size formatting
        size_str = f"{size / 1024:.1f} KB" if size < 1024 * 1024 else f"{size / (1024 * 1024):.1f} MB"
        self.lbl_size = QLabel(f"📁 {size_str}")
        self.lbl_size.setStyleSheet("color: #a1a1aa; font-size: 11px;")

        # Date formatting
        date_str = modified_time.split("T")[0] if "T" in modified_time else modified_time.split(" ")[0]
        self.lbl_date = QLabel(f"🕒 {date_str}")
        self.lbl_date.setStyleSheet("color: #a1a1aa; font-size: 11px;")

        footer_layout.addWidget(self.lbl_ext)
        footer_layout.addWidget(self.lbl_size)
        footer_layout.addWidget(self.lbl_date)
        footer_layout.addStretch()

        layout.addLayout(footer_layout)

        # Component styling
        self.setStyleSheet("""
            #resultCard {
                background-color: #18181b;
                border: 1px solid #27272a;
                border-radius: 8px;
            }
            #resultCard:hover {
                background-color: #202024;
                border-color: #3f3f46;
            }
        """)


# ---------------------------------------------------------------------------
# MainWindow Overhaul
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):

    def __init__(self) -> None:
        super().__init__()
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowSystemMenuHint)
        self.setMinimumSize(1250, 780)

        # Backend Components
        self.indexer = FileIndexer()
        self.search_engine = SearchEngine()

        # State Variables
        self._results: list[dict[str, Any]] = []
        self._index_worker: IndexWorker | None = None
        self._search_worker: SearchWorker | None = None

        self.setup_ui()
        self.connect_signals()
        self.apply_styles()
        self.update_stats()

    def setup_ui(self) -> None:
        """Create and arrange UI components with modern layouts."""
        central = QWidget()
        central.setObjectName("centralWidget")
        self.setCentralWidget(central)

        # Main vertical flow
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # 1. Custom Title bar
        self.title_bar = TitleBar(self)
        root_layout.addWidget(self.title_bar)

        # 2. Main split console container
        self.split_container = QSplitter(Qt.Orientation.Horizontal)
        self.split_container.setObjectName("mainSplitter")
        self.split_container.setHandleWidth(2)

        # Create sidebar and main console content panels
        self.setup_sidebar()
        self.setup_main_console()

        root_layout.addWidget(self.split_container, stretch=1)

        # 3. Status Bar
        self.status_bar = QStatusBar()
        self.status_bar.setObjectName("statusBar")
        self.status_bar.setSizeGripEnabled(True)
        self.status_bar.showMessage("Systems operational. Ready.")
        root_layout.addWidget(self.status_bar)

    def setup_sidebar(self) -> None:
        """Construct the sidebar stats and quick actions panel."""
        sidebar = QWidget()
        sidebar.setObjectName("sidebarPanel")
        sidebar.setMinimumWidth(280)
        sidebar.setMaximumWidth(360)

        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(20, 24, 20, 24)
        layout.setSpacing(20)

        # Section 1: Dashboard Stats Card
        stats_card = QFrame()
        stats_card.setObjectName("statsCard")
        stats_layout = QVBoxLayout(stats_card)
        stats_layout.setContentsMargins(16, 16, 16, 16)
        stats_layout.setSpacing(14)

        lbl_stats_title = QLabel("📊 Database Status")
        lbl_stats_title.setStyleSheet("font-weight: bold; font-size: 12px; color: #a1a1aa; text-transform: uppercase; letter-spacing: 0.5px;")
        stats_layout.addWidget(lbl_stats_title)

        # Files Count
        row_files = QHBoxLayout()
        lbl_files = QLabel("Total Files Indexed")
        lbl_files.setStyleSheet("color: #a1a1aa; font-size: 13px;")
        self.val_files = QLabel("0")
        self.val_files.setStyleSheet("font-weight: bold; color: #ffffff; font-size: 15px;")
        row_files.addWidget(lbl_files)
        row_files.addStretch()
        row_files.addWidget(self.val_files)
        stats_layout.addLayout(row_files)

        # Vectors Count
        row_vectors = QHBoxLayout()
        lbl_vectors = QLabel("Vector Dimensions")
        lbl_vectors.setStyleSheet("color: #a1a1aa; font-size: 13px;")
        self.val_vectors = QLabel("0")
        self.val_vectors.setStyleSheet("font-weight: bold; color: #ffffff; font-size: 15px;")
        row_vectors.addWidget(lbl_vectors)
        row_vectors.addStretch()
        row_vectors.addWidget(self.val_vectors)
        stats_layout.addLayout(row_vectors)

        # Model used info
        row_model = QVBoxLayout()
        lbl_model_title = QLabel("Semantic Encoder Model")
        lbl_model_title.setStyleSheet("color: #71717a; font-size: 11px;")
        lbl_model_val = QLabel("all-MiniLM-L6-v2 (384d)")
        lbl_model_val.setStyleSheet("font-weight: bold; color: #8b5cf6; font-size: 12px;")
        row_model.addWidget(lbl_model_title)
        row_model.addWidget(lbl_model_val)
        stats_layout.addLayout(row_model)

        layout.addWidget(stats_card)

        # Section 2: Command Actions
        actions_card = QFrame()
        actions_card.setObjectName("actionsCard")
        actions_layout = QVBoxLayout(actions_card)
        actions_layout.setContentsMargins(16, 16, 16, 16)
        actions_layout.setSpacing(12)

        lbl_actions_title = QLabel("⚙️ Operations")
        lbl_actions_title.setStyleSheet("font-weight: bold; font-size: 12px; color: #a1a1aa; text-transform: uppercase; letter-spacing: 0.5px;")
        actions_layout.addWidget(lbl_actions_title)

        self.btn_index = QPushButton("⊕  Index Directory")
        self.btn_index.setObjectName("btnPrimary")
        self.btn_index.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_index.setToolTip("Select a local directory to index recursively")

        self.btn_reset_db = QPushButton("🗑️  Reset Database")
        self.btn_reset_db.setObjectName("btnDanger")
        self.btn_reset_db.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_reset_db.setToolTip("Clear all metadata and FAISS index records")

        actions_layout.addWidget(self.btn_index)
        actions_layout.addWidget(self.btn_reset_db)

        layout.addWidget(actions_card)

        # Section 3: Quick Filter Extension Tags
        filter_card = QFrame()
        filter_card.setObjectName("filterCard")
        filter_layout = QVBoxLayout(filter_card)
        filter_layout.setContentsMargins(16, 16, 16, 16)
        filter_layout.setSpacing(12)

        lbl_filter_title = QLabel("🏷️ Quick File Filters")
        lbl_filter_title.setStyleSheet("font-weight: bold; font-size: 12px; color: #a1a1aa; text-transform: uppercase; letter-spacing: 0.5px;")
        filter_layout.addWidget(lbl_filter_title)

        filters = [("All Formats", "*"), ("PDF Files", "pdf"), ("Word Docs", "docx"), ("Spreadsheets", "csv"), ("Scripts / Code", "code"), ("Markdown", "md")]
        self.filter_buttons = []
        for name, ext in filters:
            btn = QPushButton(name)
            btn.setObjectName("btnFilter")
            btn.setCheckable(True)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setProperty("extension", ext)
            filter_layout.addWidget(btn)
            self.filter_buttons.append(btn)

        layout.addWidget(filter_card)
        layout.addStretch()

        self.split_container.addWidget(sidebar)

    def setup_main_console(self) -> None:
        """Construct the search field, results listing, and code preview area."""
        main_console = QWidget()
        main_console.setObjectName("mainConsole")

        layout = QVBoxLayout(main_console)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(20)

        # Search Bar Row
        search_layout = QHBoxLayout()
        search_layout.setSpacing(12)

        self.search_input = QLineEdit()
        self.search_input.setObjectName("searchInput")
        self.search_input.setPlaceholderText("Search documents using natural language... (e.g. 'resume python cover letter')")
        self.search_input.setMinimumHeight(46)

        self.btn_search = QPushButton("🔍  Search")
        self.btn_search.setObjectName("btnSearch")
        self.btn_search.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_search.setMinimumHeight(46)
        self.btn_search.setMinimumWidth(120)

        search_layout.addWidget(self.search_input, stretch=1)
        search_layout.addWidget(self.btn_search)
        layout.addLayout(search_layout)

        # Inner content split container (Results vs Preview Panel)
        self.inner_split = QSplitter(Qt.Orientation.Horizontal)
        self.inner_split.setObjectName("innerSplitter")
        self.inner_split.setHandleWidth(2)

        # Results list area
        results_container = QWidget()
        results_layout = QVBoxLayout(results_container)
        results_layout.setContentsMargins(0, 0, 0, 0)
        results_layout.setSpacing(10)

        lbl_results = QLabel("Matches Found")
        lbl_results.setObjectName("panelLabel")
        results_layout.addWidget(lbl_results)

        self.results_list = QListWidget()
        self.results_list.setObjectName("resultsList")
        self.results_list.setSpacing(6)
        results_layout.addWidget(self.results_list)
        self.inner_split.addWidget(results_container)

        # Text/Document Preview Area
        preview_container = QWidget()
        preview_layout = QVBoxLayout(preview_container)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        preview_layout.setSpacing(10)

        preview_header = QHBoxLayout()
        lbl_preview = QLabel("Document Snippet Preview")
        lbl_preview.setObjectName("panelLabel")
        preview_header.addWidget(lbl_preview)

        preview_header.addStretch()

        self.btn_open_file = QPushButton("↗  Open Native File")
        self.btn_open_file.setObjectName("btnPreviewOpen")
        self.btn_open_file.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_open_file.setEnabled(False)
        preview_header.addWidget(self.btn_open_file)

        preview_layout.addLayout(preview_header)

        self.preview_area = QTextEdit()
        self.preview_area.setObjectName("previewArea")
        self.preview_area.setReadOnly(True)
        self.preview_area.setPlaceholderText("Select any result card on the left to preview extracted content and context here...")
        preview_layout.addWidget(self.preview_area)
        self.inner_split.addWidget(preview_container)

        self.inner_split.setSizes([450, 750])
        layout.addWidget(self.inner_split, stretch=1)

        self.split_container.addWidget(main_console)

    # ------------------------------------------------------------------
    # Event Handlers & Core Methods
    # ------------------------------------------------------------------

    def connect_signals(self) -> None:
        """Wire up user events to trigger logical operations."""
        self.btn_index.clicked.connect(self.select_and_index_folder)
        self.btn_reset_db.clicked.connect(self.reset_entire_database)
        self.btn_search.clicked.connect(self.trigger_search)
        self.search_input.returnPressed.connect(self.trigger_search)
        self.results_list.currentRowChanged.connect(self.update_preview)
        self.results_list.itemDoubleClicked.connect(self.open_file_native)
        self.btn_open_file.clicked.connect(self.open_file_native)

        # Quick filters grouping
        for btn in self.filter_buttons:
            btn.clicked.connect(self.handle_filter_click)

        # Shortcut Ctrl+F focuses search input
        shortcut = QShortcut(QKeySequence("Ctrl+F"), self)
        shortcut.activated.connect(self.search_input.setFocus)

    def update_stats(self) -> None:
        """Query SQLite database records and FAISS indices to display totals."""
        try:
            total_files = self.indexer.db_manager.count_files()
            total_vectors = self.indexer.faiss_manager.get_total_vectors()
            self.val_files.setText(str(total_files))
            self.val_vectors.setText(f"{total_vectors} ({self.indexer.embedding_manager.get_dimension()}d)")
        except Exception as exc:
            logger.error("Failed to query metadata stats: %s", exc)

    def handle_filter_click(self) -> None:
        """Coordinate tag button selections to toggle filter logic."""
        sender = self.sender()
        if not sender:
            return

        # Ensure only one filter button is checked at a time
        for btn in self.filter_buttons:
            if btn != sender:
                btn.setChecked(False)

        ext_filter = sender.property("extension") if sender.isChecked() else "*"
        self.filter_current_results(ext_filter)

    def filter_current_results(self, extension_filter: str) -> None:
        """Instantly filter lists of results based on file extensions."""
        self.results_list.clear()

        code_extensions = {".py", ".js", ".ts", ".java", ".c", ".cpp", ".h", ".hpp", ".cs", ".go", ".rs", ".php", ".rb", ".swift", ".kt"}

        filtered_count = 0
        for i, result in enumerate(self._results):
            ext = result.get("extension", "").replace(".", "").lower()

            match = False
            if extension_filter == "*":
                match = True
            elif extension_filter == "code":
                match = f".{ext}" in code_extensions
            else:
                match = ext == extension_filter

            if match:
                item = QListWidgetItem()
                card = ResultCard(
                    filename=result.get("filename", result.get("file_name", "Unknown")),
                    path=result.get("path", result.get("file_path", "")),
                    score=result.get("similarity_score", 0.0),
                    size=result.get("size", 0),
                    ext=f".{ext}",
                    modified_time=result.get("modified_time", "N/A"),
                )
                item.setSizeHint(card.sizeHint())
                # Save actual index mapping in item user data
                item.setData(Qt.ItemDataRole.UserRole, i)
                self.results_list.addItem(item)
                self.results_list.setItemWidget(item, card)
                filtered_count += 1

        self.status_bar.showMessage(f"Displaying {filtered_count} matching filtered result(s).")

    def select_and_index_folder(self) -> None:
        """Trigger thread-safe recursive directory file text-extraction and embedding indexing."""
        folder_path = QFileDialog.getExistingDirectory(
            self,
            "Select Folder to Index",
            str(Path(__file__).resolve().parent.parent.parent / "Sample_files")
        )
        if not folder_path:
            return

        self.status_bar.showMessage(f"Indexing Folder: {folder_path}...")
        self.btn_index.setEnabled(False)
        self.btn_index.setText("Indexing files...")

        self._index_worker = IndexWorker(self.indexer, folder_path)
        self._index_worker.finished.connect(self._on_indexing_finished)
        self._index_worker.error.connect(self._on_indexing_error)
        self._index_worker.start()

    def _on_indexing_finished(self, summary: dict) -> None:
        self.btn_index.setEnabled(True)
        self.btn_index.setText("⊕  Index Directory")
        self.update_stats()

        msg = (
            f"Successfully indexed: {summary['indexed']} files\n"
            f"Skipped (empty/unsupported): {summary['skipped']}\n"
            f"Failed: {summary['failed']}"
        )
        QMessageBox.information(self, "Indexing Completed", msg)
        self.status_bar.showMessage(f"Indexing completed. {summary['indexed']} files added/updated.")

    def _on_indexing_error(self, error: str) -> None:
        self.btn_index.setEnabled(True)
        self.btn_index.setText("⊕  Index Directory")
        self.status_bar.showMessage("Indexing failed.")
        QMessageBox.critical(self, "Indexing Pipeline Error", f"Failed to run indexing pipeline:\n{error}")

    def reset_entire_database(self) -> None:
        """Clear database and remove all vectors from FAISS."""
        reply = QMessageBox.question(
            self,
            "Reset Database",
            "Are you absolutely sure you want to delete all indexed data? This cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            try:
                self.indexer.db_manager.clear_all()
                self.indexer.faiss_manager.reset()
                self.indexer.faiss_manager.save()
                self.results_list.clear()
                self.preview_area.clear()
                self._results = []
                self.btn_open_file.setEnabled(False)
                self.update_stats()
                self.status_bar.showMessage("Database reset complete. All indices cleared.")
            except Exception as e:
                QMessageBox.critical(self, "Reset Error", f"Failed to reset data:\n{e}")

    def trigger_search(self) -> None:
        """Initiate background thread search using the query string."""
        query = self.search_input.text().strip()
        if not query:
            return

        self.results_list.clear()
        self.preview_area.clear()
        self._results = []
        self.btn_open_file.setEnabled(False)

        self.status_bar.showMessage(f"Searching semantic space for '{query}'...")
        self.btn_search.setEnabled(False)
        self.btn_search.setText("Searching...")

        # Reset quick filters check state
        for btn in self.filter_buttons:
            btn.setChecked(False)

        self._search_worker = SearchWorker(self.search_engine, query, top_k=20)
        self._search_worker.finished.connect(self._on_search_finished)
        self._search_worker.error.connect(self._on_search_error)
        self._search_worker.start()

    def _on_search_finished(self, results: list[dict]) -> None:
        self.btn_search.setEnabled(True)
        self.btn_search.setText("🔍  Search")
        self._results = results

        if not results:
            self.status_bar.showMessage("No matches found.")
            item = QListWidgetItem("No matching records found in local index database.")
            item.setFlags(Qt.ItemFlag.NoItemFlags)
            self.results_list.addItem(item)
            return

        # Populates list with custom Result Cards
        self.filter_current_results("*")

    def _on_search_error(self, error: str) -> None:
        self.btn_search.setEnabled(True)
        self.btn_search.setText("🔍  Search")
        self.status_bar.showMessage("Search failed.")
        QMessageBox.critical(self, "Search Error", f"Search pipeline failed:\n{error}")

    def update_preview(self, row: int) -> None:
        """Display context or extracted text snippets inside preview editor area."""
        if row < 0 or row >= self.results_list.count():
            self.preview_area.clear()
            self.btn_open_file.setEnabled(False)
            return

        item = self.results_list.item(row)
        actual_index = item.data(Qt.ItemDataRole.UserRole)
        if actual_index is None or actual_index >= len(self._results):
            return

        result = self._results[actual_index]
        content = result.get("content", "").strip()

        if content:
            self.preview_area.setPlainText(content)
        else:
            path = result.get("path", result.get("file_path", ""))
            resolved = resolve_file_path(path)
            self.preview_area.setPlainText(f"No textual preview contents available.\n\nFile reference: {resolved}")

        self.btn_open_file.setEnabled(True)

    def open_file_native(self) -> None:
        """Resolve database path relative to current context and open in native OS app."""
        current_row = self.results_list.currentRow()
        if current_row < 0:
            return

        item = self.results_list.item(current_row)
        actual_index = item.data(Qt.ItemDataRole.UserRole)
        if actual_index is None or actual_index >= len(self._results):
            return

        result = self._results[actual_index]
        path = result.get("path", result.get("file_path", ""))
        resolved = resolve_file_path(path)

        if not Path(resolved).exists():
            QMessageBox.warning(self, "File Not Found", f"Cannot find file on filesystem:\n{resolved}")
            return

        try:
            if sys.platform == "win32":
                os.startfile(resolved)
            elif sys.platform == "darwin":
                subprocess.run(["open", resolved], check=True)
            else:
                subprocess.run(["xdg-open", resolved], check=True)
            self.status_bar.showMessage(f"Opened file: {Path(resolved).name}")
        except Exception as exc:
            QMessageBox.critical(self, "Failed to Open File", f"Error launching default application:\n{str(exc)}")

    # ------------------------------------------------------------------
    # Styling System
    # ------------------------------------------------------------------

    def apply_styles(self) -> None:
        """Apply CSS stylesheets simulating modern frameless app shells."""
        self.setStyleSheet("""
            /* --- Global Color System & Resets --- */
            QMainWindow, QWidget#centralWidget {
                background-color: #09090b; /* Zinc 950 base */
                color: #fafafa; /* Zinc 50 primary text */
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
                font-size: 13px;
                border: 1px solid #27272a; /* Thin borders for frameless wrapper */
                border-radius: 8px;
            }

            /* --- Custom Title Bar --- */
            #titleBar {
                background-color: #09090b;
                border-bottom: 1px solid #27272a;
            }

            /* --- Sidebar Styling --- */
            #sidebarPanel {
                background-color: #09090b;
                border-right: 1px solid #27272a;
            }

            #statsCard, #actionsCard, #filterCard {
                background-color: #18181b; /* Zinc 900 */
                border: 1px solid #27272a;
                border-radius: 8px;
            }

            /* --- Buttons --- */
            QPushButton#btnPrimary {
                background-color: #8b5cf6; /* Violet 500 */
                color: #ffffff;
                border: none;
                border-radius: 6px;
                padding: 10px 16px;
                font-weight: 600;
                font-size: 13px;
            }
            QPushButton#btnPrimary:hover {
                background-color: #7c3aed; /* Violet 600 */
            }
            QPushButton#btnPrimary:pressed {
                background-color: #6d28d9; /* Violet 700 */
            }
            QPushButton#btnPrimary:disabled {
                background-color: #27272a;
                color: #52525b;
            }

            QPushButton#btnDanger {
                background-color: transparent;
                color: #f43f5e; /* Rose 500 */
                border: 1px solid #e11d4833;
                border-radius: 6px;
                padding: 9px 16px;
                font-weight: 600;
                font-size: 13px;
            }
            QPushButton#btnDanger:hover {
                background-color: #f43f5e1a;
                border-color: #f43f5e;
            }
            QPushButton#btnDanger:pressed {
                background-color: #f43f5e2d;
            }

            QPushButton#btnSearch {
                background-color: #ffffff;
                color: #09090b;
                border: none;
                border-radius: 6px;
                padding: 10px 20px;
                font-weight: 600;
                font-size: 13px;
            }
            QPushButton#btnSearch:hover {
                background-color: #e4e4e7;
            }
            QPushButton#btnSearch:pressed {
                background-color: #d4d4d8;
            }
            QPushButton#btnSearch:disabled {
                background-color: #27272a;
                color: #71717a;
            }

            QPushButton#btnPreviewOpen {
                background-color: transparent;
                color: #ffffff;
                border: 1px solid #27272a;
                border-radius: 6px;
                padding: 6px 12px;
                font-size: 11px;
                font-weight: 600;
            }
            QPushButton#btnPreviewOpen:hover {
                background-color: #27272a;
            }
            QPushButton#btnPreviewOpen:disabled {
                border-color: transparent;
                color: #52525b;
            }

            /* Filter Tags */
            QPushButton#btnFilter {
                background-color: transparent;
                color: #a1a1aa;
                border: 1px solid transparent;
                border-radius: 6px;
                padding: 8px 12px;
                text-align: left;
                font-weight: 500;
            }
            QPushButton#btnFilter:hover {
                background-color: #27272a;
                color: #ffffff;
            }
            QPushButton#btnFilter:checked {
                background-color: #8b5cf61a;
                color: #c084fc;
                border: 1px solid #8b5cf633;
                font-weight: 600;
            }

            /* --- Main Console --- */
            #mainConsole {
                background-color: #09090b;
            }

            #searchInput {
                background-color: #18181b;
                color: #ffffff;
                border: 1px solid #27272a;
                border-radius: 8px;
                padding: 10px 16px;
                font-size: 14px;
                selection-background-color: #8b5cf63a;
            }
            #searchInput:focus {
                border: 1px solid #8b5cf6; /* Glowing violet border */
                background-color: #18181b;
            }
            #searchInput::placeholder {
                color: #52525b;
            }

            #panelLabel {
                font-size: 11px;
                font-weight: bold;
                color: #a1a1aa;
                letter-spacing: 0.8px;
                text-transform: uppercase;
                padding-bottom: 2px;
            }

            /* --- Results List --- */
            #resultsList {
                background-color: transparent;
                border: none;
                outline: none;
            }
            #resultsList::item {
                background-color: transparent;
                border: none;
                margin-bottom: 2px;
                border-radius: 8px;
            }
            #resultsList::item:selected {
                background-color: #8b5cf60a;
                border: 1px solid #8b5cf650;
                border-radius: 8px;
            }

            /* --- Preview Panel Editor --- */
            #previewArea {
                background-color: #0c0a0f; /* Rich dark-plum theme coding style */
                color: #e4e4e7;
                border: 1px solid #27272a;
                border-radius: 8px;
                padding: 16px;
                font-family: "Cascadia Code", "JetBrains Mono", "Courier New", monospace;
                font-size: 13px;
                line-height: 1.6;
                selection-background-color: #8b5cf630;
                selection-color: #ffffff;
            }

            /* --- Splitter Handles --- */
            #mainSplitter::handle, #innerSplitter::handle {
                background-color: #27272a;
            }

            /* --- Status Bar --- */
            #statusBar {
                background-color: #09090b;
                color: #71717a;
                border-top: 1px solid #27272a;
                font-size: 12px;
                padding-left: 14px;
            }

            /* --- Scrollbars --- */
            QScrollBar:vertical {
                background: transparent;
                width: 8px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background: #27272a;
                border-radius: 4px;
                min-height: 24px;
            }
            QScrollBar::handle:vertical:hover {
                background: #3f3f46;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
            QScrollBar:horizontal {
                background: transparent;
                height: 8px;
                margin: 0px;
            }
            QScrollBar::handle:horizontal {
                background: #27272a;
                border-radius: 4px;
                min-width: 24px;
            }
            QScrollBar::handle:horizontal:hover {
                background: #3f3f46;
            }
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
                width: 0px;
            }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical,
            QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {
                background: none;
            }
        """)