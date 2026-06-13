"""
main_window.py — Main GUI window for AI File Search Assistant.

Implements a three-column split dashboard layout matching the wireframe:
- Column 1: Document database listing, active statistics, and index/reset controls.
- Column 2: Document text preview editor with native file launching.
- Column 3: Context-aware AI chat assistant Q&A.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from PySide6.QtCore import QPoint, Qt, QThread, Signal
from PySide6.QtGui import QAction, QFont, QKeySequence, QShortcut, QIcon, QColor, QTextCharFormat, QTextCursor
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

import qdarktheme

# ---------------------------------------------------------------------------
# Backend imports
# ---------------------------------------------------------------------------

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.indexing.file_indexer import FileIndexer
from app.search.search_engine import SearchEngine
from app.database.db_manager import DatabaseManager
from app.embeddings.embedding_manager import EmbeddingManager
from app.search.faiss_manager import FAISSManager
from app.chat.chat_engine import ChatEngine

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Path Resolver (cross-platform helper)
# ---------------------------------------------------------------------------

def resolve_file_path(path_str: str) -> str:
    """Resolve absolute or relative paths from database to the current machine."""
    path = Path(path_str)
    if path.exists():
        return str(path.resolve())

    parts = path.parts
    for i, part in enumerate(parts):
        if part == "Sample_files":
            rel_path = Path(*parts[i:])
            project_root = Path(__file__).resolve().parent.parent.parent
            resolved = project_root / rel_path
            if resolved.exists():
                return str(resolved.resolve())

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
        self.logo = QLabel("◆")
        self.logo.setStyleSheet("font-size: 14px; color: #14b8a6;")
        layout.addWidget(self.logo)

        # App Title
        self.title = QLabel("AI File Search Workspace")
        self.title.setStyleSheet("font-weight: 700; font-size: 13px; letter-spacing: 0.2px;")
        layout.addWidget(self.title)

        layout.addStretch()

        # Dynamic Action Buttons inside Title Bar
        self.btn_theme = QPushButton("🌙")
        self.btn_theme.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_theme.setFixedHeight(30)
        self.btn_theme.setFixedWidth(34)
        self.btn_theme.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                border: 1px solid rgba(120, 120, 120, 0.2);
                border-radius: 6px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: rgba(120, 120, 120, 0.1);
            }
        """)
        layout.addWidget(self.btn_theme)

        # Separator line
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.VLine)
        sep.setFrameShadow(QFrame.Shadow.Sunken)
        sep.setStyleSheet("color: rgba(120, 120, 120, 0.2);")
        sep.setFixedHeight(20)
        layout.addWidget(sep)

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
                    background-color: rgba(120, 120, 120, 0.1);
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
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(6)

        # Header Row
        header_layout = QHBoxLayout()
        self.lbl_filename = QLabel(filename)
        self.lbl_filename.setObjectName("lblFilename")
        self.lbl_filename.setStyleSheet("font-weight: 700; font-size: 13px;")
        self.lbl_filename.setWordWrap(True)

        # Match Score Badge
        self.lbl_score = QLabel(f"{int(score * 100)}% Match")
        score_color = "#14b8a6" if score >= 0.75 else "#eab308" if score >= 0.5 else "#f97316"
        self.lbl_score.setStyleSheet(f"""
            background-color: {score_color}1a;
            color: {score_color};
            border: 1px solid {score_color}33;
            border-radius: 6px;
            padding: 2px 6px;
            font-size: 9px;
            font-weight: 700;
        """)

        header_layout.addWidget(self.lbl_filename, stretch=1)
        header_layout.addWidget(self.lbl_score)
        layout.addLayout(header_layout)

        # Path description
        resolved_path = resolve_file_path(path)
        self.lbl_path = QLabel(resolved_path)
        self.lbl_path.setObjectName("lblPath")
        self.lbl_path.setStyleSheet("font-size: 11px;")
        self.lbl_path.setWordWrap(True)
        layout.addWidget(self.lbl_path)

        # Metadata Footer Row
        footer_layout = QHBoxLayout()
        footer_layout.setSpacing(8)

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
            padding: 1px 5px;
            font-size: 8px;
            font-weight: 800;
        """)

        size_str = f"{size / 1024:.1f} KB" if size < 1024 * 1024 else f"{size / (1024 * 1024):.1f} MB"
        self.lbl_size = QLabel(f"📁 {size_str}")
        self.lbl_size.setObjectName("lblSize")
        self.lbl_size.setStyleSheet("font-size: 11px;")

        date_str = modified_time.split("T")[0] if "T" in modified_time else modified_time.split(" ")[0]
        self.lbl_date = QLabel(f"🕒 {date_str}")
        self.lbl_date.setObjectName("lblDate")
        self.lbl_date.setStyleSheet("font-size: 11px;")

        footer_layout.addWidget(self.lbl_ext)
        footer_layout.addWidget(self.lbl_size)
        footer_layout.addWidget(self.lbl_date)
        footer_layout.addStretch()

        layout.addLayout(footer_layout)


# ---------------------------------------------------------------------------
# MainWindow Overhaul
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):

    def __init__(self) -> None:
        super().__init__()
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowSystemMenuHint)
        self.setMinimumSize(1250, 780)

        # Backend Components
        self.db_manager = DatabaseManager()
        self.embedding_manager = EmbeddingManager()
        self.faiss_manager = FAISSManager()

        self.indexer = FileIndexer(
            db_manager=self.db_manager,
            embedding_manager=self.embedding_manager,
            faiss_manager=self.faiss_manager,
        )

        self.search_engine = SearchEngine(
            embedding_manager=self.embedding_manager,
            faiss_manager=self.faiss_manager,
            db_manager=self.db_manager,
        )

        self.chat_engine = ChatEngine(
            db_manager=self.db_manager,
            faiss_manager=self.faiss_manager,
        )

        # State Variables
        self._results: list[dict[str, Any]] = []
        self._index_worker: IndexWorker | None = None
        self._search_worker: SearchWorker | None = None
        self.current_theme = "dark"

        self.setup_ui()
        self.connect_signals()
        self.set_theme("dark")  # Initialize global theme
        self.load_all_files()   # Populate left list with all files initially

        # Pre-load embedding model in a background thread to prevent UI freezing
        import threading
        threading.Thread(
            target=self.embedding_manager._ensure_model_loaded,
            daemon=True
        ).start()

    def setup_ui(self) -> None:
        """Construct the 3-column dashboard matching the layout wireframe."""
        central = QWidget()
        central.setObjectName("centralWidget")
        self.setCentralWidget(central)

        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # 1. Custom Title bar
        self.title_bar = TitleBar(self)
        root_layout.addWidget(self.title_bar)

        # Container for content below the custom titlebar
        workspace_container = QWidget()
        workspace_layout = QVBoxLayout(workspace_container)
        workspace_layout.setContentsMargins(20, 16, 20, 16)
        workspace_layout.setSpacing(16)

        # 2. Header Bar — title chip + natural language search + search button,
        #    all grouped inside one soft connected container.
        self.header_bar = QFrame()
        self.header_bar.setObjectName("headerBar")
        header_row = QHBoxLayout(self.header_bar)
        header_row.setContentsMargins(8, 8, 8, 8)
        header_row.setSpacing(10)

        # Left Title Chip
        self.title_block = QFrame()
        self.title_block.setObjectName("titleBlock")
        self.title_block.setFixedWidth(220)
        title_block_layout = QHBoxLayout(self.title_block)
        title_block_layout.setContentsMargins(14, 0, 14, 0)
        title_block_layout.setSpacing(8)

        lbl_title_icon = QLabel("◆")
        lbl_title_icon.setObjectName("lblTitleIcon")

        lbl_title_block = QLabel("AI File Search")
        lbl_title_block.setObjectName("lblTitleBlock")

        title_block_layout.addWidget(lbl_title_icon)
        title_block_layout.addWidget(lbl_title_block)
        title_block_layout.addStretch()

        # Center Search Bar
        self.search_input = QLineEdit()
        self.search_input.setObjectName("searchInput")
        self.search_input.setPlaceholderText('Ask in plain language — e.g. "invoices from last March"')
        self.search_input.setMinimumHeight(44)

        # Right Search Button
        self.btn_search = QPushButton("Search")
        self.btn_search.setObjectName("btnPrimary")
        self.btn_search.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_search.setMinimumHeight(44)
        self.btn_search.setFixedWidth(120)

        header_row.addWidget(self.title_block)
        header_row.addWidget(self.search_input, stretch=1)
        header_row.addWidget(self.btn_search)
        workspace_layout.addWidget(self.header_bar)

        # 3. Main Split Area: Three Vertical Columns
        self.split_container = QSplitter(Qt.Orientation.Horizontal)
        self.split_container.setObjectName("mainSplitter")
        self.split_container.setHandleWidth(2)

        # --- COLUMN 1 (LEFT SIDEBAR): FILES LIST + CONTROL BUTTONS ---
        self.left_panel = QWidget()
        self.left_panel.setObjectName("leftPanel")
        left_layout = QVBoxLayout(self.left_panel)
        left_layout.setContentsMargins(14, 14, 14, 14)
        left_layout.setSpacing(12)

        # Header for the file list
        left_header = QHBoxLayout()
        lbl_left_title = QLabel("All indexed files")
        lbl_left_title.setObjectName("panelLabel")
        left_header.addWidget(lbl_left_title)
        
        self.btn_clear_search = QPushButton("✕ Clear")
        self.btn_clear_search.setObjectName("btnSecondary")
        self.btn_clear_search.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_clear_search.setFixedHeight(22)
        self.btn_clear_search.setFixedWidth(56)
        self.btn_clear_search.setStyleSheet("font-size: 9px; padding: 0;")
        left_header.addWidget(self.btn_clear_search)
        left_layout.addLayout(left_header)

        # Filter buttons
        filters_layout = QHBoxLayout()
        filters_layout.setSpacing(4)
        filters = [("All", "*"), ("PDF", "pdf"), ("Word", "docx"), ("Spreadsheet", "csv"), ("Code", "code")]
        self.filter_buttons = []
        for name, ext in filters:
            btn = QPushButton(name)
            btn.setObjectName("btnFilter")
            btn.setCheckable(True)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setProperty("extension", ext)
            btn.setStyleSheet("""
                QPushButton {
                    background-color: transparent;
                    border: 1px solid rgba(120, 120, 120, 0.15);
                    border-radius: 4px;
                    padding: 3px 6px;
                    font-size: 9px;
                    font-weight: 600;
                }
                QPushButton:hover {
                    background-color: rgba(120, 120, 120, 0.08);
                }
                QPushButton:checked {
                    background-color: rgba(20, 184, 166, 0.16);
                    border-color: #14b8a6;
                    color: #14b8a6;
                }
            """)
            filters_layout.addWidget(btn)
            self.filter_buttons.append(btn)
        left_layout.addLayout(filters_layout)

        # The files list widget
        self.results_list = QListWidget()
        self.results_list.setObjectName("resultsList")
        self.results_list.setSpacing(6)
        left_layout.addWidget(self.results_list)

        # Separator line before operations/info
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFrameShadow(QFrame.Shadow.Sunken)
        sep.setObjectName("panelSep")
        left_layout.addWidget(sep)

        # Info & Operations Footer
        lbl_ops_title = QLabel("indexed information")
        lbl_ops_title.setObjectName("panelLabel")
        left_layout.addWidget(lbl_ops_title)

        self.lbl_stats = QLabel("Total Files: 0 | Vectors: 0")
        self.lbl_stats.setObjectName("lblStats")
        self.lbl_stats.setStyleSheet("font-size: 11px;")
        left_layout.addWidget(self.lbl_stats)

        ops_buttons = QHBoxLayout()
        ops_buttons.setSpacing(8)

        self.btn_index = QPushButton("⊕ Index Folder")
        self.btn_index.setObjectName("btnSecondary")
        self.btn_index.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_index.setFixedHeight(34)
        ops_buttons.addWidget(self.btn_index)

        self.btn_reset_db = QPushButton("🗑️ Reset DB")
        self.btn_reset_db.setObjectName("btnDanger")
        self.btn_reset_db.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_reset_db.setFixedHeight(34)
        ops_buttons.addWidget(self.btn_reset_db)
        left_layout.addLayout(ops_buttons)

        self.split_container.addWidget(self.left_panel)

        # --- COLUMN 2 (MIDDLE PANEL): PREVIEW AREA + NATIVE LAUNCHER ---
        self.middle_panel = QWidget()
        self.middle_panel.setObjectName("middlePanel")
        middle_layout = QVBoxLayout(self.middle_panel)
        middle_layout.setContentsMargins(14, 14, 14, 14)
        middle_layout.setSpacing(12)

        middle_layout.addWidget(QLabel("Document Content Preview", objectName="panelLabel"))

        self.preview_area = QTextEdit()
        self.preview_area.setObjectName("previewArea")
        self.preview_area.setReadOnly(True)
        self.preview_area.setPlaceholderText("Select any file on the left list to load its preview text here...")
        middle_layout.addWidget(self.preview_area)

        self.btn_open_file = QPushButton("Open File")
        self.btn_open_file.setObjectName("btnSecondary")
        self.btn_open_file.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_open_file.setEnabled(False)
        self.btn_open_file.setFixedHeight(38)
        middle_layout.addWidget(self.btn_open_file)

        self.split_container.addWidget(self.middle_panel)

        # --- COLUMN 3 (RIGHT PANEL): CHAT ENGINE PANEL ---
        self.right_panel = QWidget()
        self.right_panel.setObjectName("rightPanel")
        right_layout = QVBoxLayout(self.right_panel)
        right_layout.setContentsMargins(14, 14, 14, 14)
        right_layout.setSpacing(12)

        right_layout.addWidget(QLabel("AI Assistant", objectName="panelLabel"))

        self.chat_output = QTextEdit()
        self.chat_output.setObjectName("chatOutput")
        self.chat_output.setReadOnly(True)
        self.chat_output.setPlaceholderText("AI responses and history stream here...")
        self.chat_output.verticalScrollBar().rangeChanged.connect(
            lambda min_val, max_val: self.chat_output.verticalScrollBar().setValue(max_val)
        )
        right_layout.addWidget(self.chat_output)

        chat_input_row = QHBoxLayout()
        chat_input_row.setSpacing(8)

        self.chat_input = QLineEdit()
        self.chat_input.setObjectName("chatInput")
        self.chat_input.setPlaceholderText("Ask a question about this document...")
        self.chat_input.setMinimumHeight(38)

        self.btn_chat = QPushButton("Ask AI")
        self.btn_chat.setObjectName("btnPrimary")
        self.btn_chat.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_chat.setFixedHeight(38)

        chat_input_row.addWidget(self.chat_input, stretch=1)
        chat_input_row.addWidget(self.btn_chat)
        right_layout.addLayout(chat_input_row)

        self.split_container.addWidget(self.right_panel)

        # Set default proportions: Left 25% | Center 45% | Right 30%
        self.split_container.setSizes([320, 560, 370])
        workspace_layout.addWidget(self.split_container, stretch=1)

        root_layout.addWidget(workspace_container, stretch=1)

        # 4. Status Bar
        self.status_bar = QStatusBar()
        self.status_bar.setObjectName("statusBar")
        self.status_bar.showMessage("Engine ready.")
        root_layout.addWidget(self.status_bar)

    # ------------------------------------------------------------------
    # Styling & Theme Switcher
    # ------------------------------------------------------------------

    def set_theme(self, theme: str) -> None:
        """Switch global PyQtDarkTheme and apply customized layout overrides."""
        self.current_theme = theme
        app = QApplication.instance()
        if app:
            app.setStyleSheet(qdarktheme.load_stylesheet(theme))

        # Configure theme toggle icon and tooltips
        if theme == "dark":
            self.title_bar.btn_theme.setText("🌙")
            self.title_bar.btn_theme.setToolTip("Switch to Light Theme (Ctrl+T)")
            
            # Palette Colors for Dark Mode — Charcoal & Teal
            base_bg = "#09090b"
            card_bg = "#18181b"
            border_color = "#27272a"
            text_color = "#f4f4f5"
            subtext_color = "#a1a1aa"
            preview_bg = "#0a0a0a"
            preview_fg = "#d4d4d8"
            accent_color = "#14b8a6"
            accent_hover = "#0d9488"
            input_bg = "#09090b"
            input_focus_border = "#14b8a6"
            banner_bg = "#1f1f23"
        else:
            self.title_bar.btn_theme.setText("☀️")
            self.title_bar.btn_theme.setToolTip("Switch to Dark Theme (Ctrl+T)")
            
            # Palette Colors for Light Mode — Slate & Teal
            base_bg = "#f8fafc"
            card_bg = "#ffffff"
            border_color = "#e2e8f0"
            text_color = "#0f172a"
            subtext_color = "#64748b"
            preview_bg = "#f1f5f9"
            preview_fg = "#1e293b"
            accent_color = "#0d9488"
            accent_hover = "#0f766e"
            input_bg = "#f8fafc"
            input_focus_border = "#0d9488"
            banner_bg = "#eef2f7"

        # Global stylesheet layer override
        self.setStyleSheet(f"""
            QMainWindow, QWidget#centralWidget {{
                background-color: {base_bg};
                color: {text_color};
                font-family: "Segoe UI", "Ubuntu", "Helvetica Neue", Arial, sans-serif;
                font-size: 13px;
            }}

            #titleBar {{
                background-color: {base_bg};
                border-bottom: 1px solid {border_color};
            }}

            #headerBar {{
                background-color: {card_bg};
                border: 1px solid {border_color};
                border-radius: 14px;
            }}

            #titleBlock {{
                background-color: {banner_bg};
                border-radius: 10px;
            }}

            #lblTitleIcon {{
                color: {accent_color};
                font-size: 14px;
            }}

            #lblTitleBlock {{
                color: {text_color};
                font-weight: 700;
                font-size: 13px;
                letter-spacing: 0.2px;
            }}

            QLineEdit#searchInput {{
                background-color: {base_bg};
                color: {text_color};
                border: 1px solid {border_color};
                border-radius: 10px;
                padding: 10px 16px;
                font-size: 13px;
            }}
            QLineEdit#searchInput:focus {{
                border: 1px solid {input_focus_border};
            }}

            #leftPanel, #middlePanel, #rightPanel {{
                background-color: {card_bg};
                border: 1px solid {border_color};
                border-radius: 12px;
            }}

            QSplitter#mainSplitter::handle {{
                background-color: transparent;
            }}
            QSplitter#mainSplitter::handle:hover {{
                background-color: {accent_color}50;
            }}

            QLineEdit#chatInput {{
                background-color: {input_bg};
                color: {text_color};
                border: 1px solid {border_color};
                border-radius: 8px;
                padding: 6px 12px;
                font-size: 12px;
            }}
            QLineEdit#chatInput:focus {{
                border-color: {accent_color};
            }}

            QPushButton#btnPrimary {{
                background-color: {accent_color};
                color: #ffffff;
                border: none;
                border-radius: 10px;
                padding: 10px 16px;
                font-weight: 600;
                font-size: 12px;
            }}
            QPushButton#btnPrimary:hover {{
                background-color: {accent_hover};
            }}
            QPushButton#btnPrimary:disabled {{
                background-color: {border_color};
                color: {subtext_color};
            }}

            QPushButton#btnSecondary {{
                background-color: transparent;
                color: {text_color};
                border: 1px solid {border_color};
                border-radius: 8px;
                padding: 10px 16px;
                font-weight: 600;
                font-size: 12px;
            }}
            QPushButton#btnSecondary:hover {{
                background-color: {border_color}a0;
            }}
            QPushButton#btnSecondary:disabled {{
                border-color: transparent;
                color: {subtext_color};
            }}

            QPushButton#btnDanger {{
                background-color: transparent;
                color: #f87171;
                border: 1px solid rgba(248, 113, 113, 0.3);
                border-radius: 8px;
                padding: 10px 16px;
                font-weight: 500;
                font-size: 12px;
            }}
            QPushButton#btnDanger:hover {{
                background-color: rgba(248, 113, 113, 0.1);
                border-color: #f87171;
            }}

            #resultsList {{
                background-color: transparent;
                border: none;
                outline: none;
            }}
            #resultsList::item {{
                background-color: transparent;
                border: none;
                border-radius: 8px;
            }}
            #resultsList::item:selected {{
                background-color: {accent_color}15;
                border: 1px solid {accent_color}40;
                border-radius: 8px;
            }}

            #resultCard {{
                background-color: {base_bg};
                border: 1px solid {border_color};
                border-radius: 10px;
            }}
            #resultCard:hover {{
                background-color: {border_color}55;
                border-color: {accent_color}50;
            }}

            #resultCard QLabel#lblFilename {{
                color: {text_color};
            }}
            #resultCard QLabel#lblPath, #resultCard QLabel#lblSize, #resultCard QLabel#lblDate {{
                color: {subtext_color};
            }}

            #previewArea {{
                background-color: {preview_bg};
                color: {preview_fg};
                border: 1px solid {border_color};
                border-radius: 10px;
                padding: 16px;
                font-family: "Cascadia Code", "JetBrains Mono", "Consolas", monospace;
                font-size: 12px;
                line-height: 1.6;
            }}

            #chatOutput {{
                background-color: {preview_bg};
                color: {preview_fg};
                border: 1px solid {border_color};
                border-radius: 10px;
                padding: 12px;
                font-size: 13px;
                line-height: 1.5;
            }}

            #panelLabel {{
                font-size: 11px;
                font-weight: 700;
                color: {subtext_color};
                letter-spacing: 0.6px;
                text-transform: uppercase;
                padding-bottom: 2px;
            }}

            #panelSep {{
                color: {border_color};
            }}
            #lblStats {{
                color: {subtext_color};
            }}
            #statusBar {{
                background-color: {base_bg};
                color: {subtext_color};
                border-top: 1px solid {border_color};
            }}
        """)

        # Re-apply stylesheets to list widgets dynamically to reload styling
        self.results_list.setStyleSheet(self.results_list.styleSheet())

    def toggle_theme(self) -> None:
        if self.current_theme == "dark":
            self.set_theme("light")
        else:
            self.set_theme("dark")

    # ------------------------------------------------------------------
    # Event Handlers & Signals
    # ------------------------------------------------------------------

    def connect_signals(self) -> None:
        """Connect actions, worker signals, shortcuts, and events."""
        # Custom TitleBar signals
        self.title_bar.btn_theme.clicked.connect(self.toggle_theme)

        # Operations buttons
        self.btn_index.clicked.connect(self.select_and_index_folder)
        self.btn_reset_db.clicked.connect(self.reset_entire_database)

        # Search Bar
        self.btn_search.clicked.connect(self.trigger_search)
        self.search_input.returnPressed.connect(self.trigger_search)
        self.btn_clear_search.clicked.connect(self.clear_search)

        # File selections & highlights
        self.results_list.currentRowChanged.connect(self.update_preview)
        self.results_list.itemDoubleClicked.connect(self.open_file_native)
        self.btn_open_file.clicked.connect(self.open_file_native)

        # Chat triggers
        self.btn_chat.clicked.connect(self.ask_ai_about_selected_file)
        self.chat_input.returnPressed.connect(self.ask_ai_about_selected_file)

        # File Filter triggers
        for btn in self.filter_buttons:
            btn.clicked.connect(self.handle_filter_click)

        # Keyboard shortcuts
        self.shortcut_theme = QShortcut(QKeySequence("Ctrl+T"), self)
        self.shortcut_theme.activated.connect(self.toggle_theme)

        self.shortcut_search = QShortcut(QKeySequence("Ctrl+F"), self)
        self.shortcut_search.activated.connect(self.focus_search_bar)

    def focus_search_bar(self) -> None:
        self.search_input.setFocus()

    def update_stats(self) -> None:
        """Update system metrics in status labels."""
        try:
            total_files = self.db_manager.count_files()
            total_vectors = self.faiss_manager.get_total_vectors()
            dim = self.embedding_manager.get_dimension()

            stats_msg = f"Indexed: {total_files} files | Vectors: {total_vectors} ({dim}d)"
            self.lbl_stats.setText(stats_msg)
            self.status_bar.showMessage(stats_msg)
        except Exception as exc:
            logger.exception("Stats retrieval failed: %s", exc)

    def load_all_files(self) -> None:
        """Load all indexed files from SQLite into the results list by default."""
        try:
            files = self.db_manager.get_all_files()
            self._results = []
            for f in files:
                self._results.append({
                    "id": f["id"],
                    "path": f["path"],
                    "filename": f["filename"],
                    "extension": f["extension"],
                    "size": f["size"],
                    "modified_time": f["modified_time"],
                    "content": f["content"],
                    "similarity_score": 1.0  # Default value
                })
            self.filter_current_results("*")
            self.update_stats()
        except Exception as exc:
            logger.exception("load_all_files failed: %s", exc)

    def clear_search(self) -> None:
        """Clear search query and reload all database files."""
        self.search_input.clear()
        self.load_all_files()

    def handle_filter_click(self) -> None:
        """Enforce single checking on tag filters."""
        sender = self.sender()
        if not sender:
            return

        for btn in self.filter_buttons:
            if btn != sender:
                btn.setChecked(False)

        ext_filter = sender.property("extension") if sender.isChecked() else "*"
        self.filter_current_results(ext_filter)

    def filter_current_results(self, extension_filter: str) -> None:
        """Instantly filter list views based on selected extension tags."""
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
                item.setData(Qt.ItemDataRole.UserRole, i)
                self.results_list.addItem(item)
                self.results_list.setItemWidget(item, card)
                filtered_count += 1

        self.status_bar.showMessage(f"Displaying {filtered_count} file(s).")

    # ------------------------------------------------------------------
    # Search & Extraction Orchestration
    # ------------------------------------------------------------------

    def trigger_search(self) -> None:
        """Trigger background QThread search or reload all files if query is empty."""
        query = self.search_input.text().strip()
        if not query:
            self.load_all_files()
            return

        self.results_list.clear()
        self._results = []
        self.preview_area.clear()
        self.chat_output.clear()
        self.chat_input.clear()
        self.btn_open_file.setEnabled(False)

        self.status_bar.showMessage(f"Searching semantic space for '{query}'...")
        self.btn_search.setEnabled(False)
        self.btn_search.setText("Searching...")

        for btn in self.filter_buttons:
            btn.setChecked(False)

        self._search_worker = SearchWorker(self.search_engine, query, top_k=20)
        self._search_worker.finished.connect(self._on_search_finished)
        self._search_worker.error.connect(self._on_search_error)
        self._search_worker.start()

    def _on_search_finished(self, results: list[dict]) -> None:
        self.btn_search.setEnabled(True)
        self.btn_search.setText("Search")
        self._results = results

        if not results:
            self.status_bar.showMessage("No matches found.")
            item = QListWidgetItem("No matching records found in local index.")
            item.setFlags(Qt.ItemFlag.NoItemFlags)
            self.results_list.addItem(item)
            return

        self.filter_current_results("*")

    def _on_search_error(self, error: str) -> None:
        self.btn_search.setEnabled(True)
        self.btn_search.setText("Search")
        self.status_bar.showMessage("Search failed.")
        QMessageBox.critical(self, "Search Error", f"Search pipeline failed:\n{error}")

    def update_preview(self, row: int) -> None:
        """Display text inside preview editors."""
        if row < 0 or row >= self.results_list.count():
            self.preview_area.clear()
            self.chat_output.clear()
            self.chat_input.clear()
            self.btn_open_file.setEnabled(False)
            return

        item = self.results_list.item(row)
        actual_index = item.data(Qt.ItemDataRole.UserRole)
        if actual_index is None or actual_index >= len(self._results):
            return

        result = self._results[actual_index]
        content = result.get("content", "").strip()

        # Reset chat states
        self.chat_engine.reset_chat()
        self.chat_output.clear()
        self.chat_input.clear()

        if content:
            self.preview_area.setPlainText(content)
        else:
            path = result.get("path", result.get("file_path", ""))
            resolved = resolve_file_path(path)
            self.preview_area.setPlainText(f"No textual preview contents available.\n\nFile reference: {resolved}")

        self.btn_open_file.setEnabled(True)

    def select_and_index_folder(self) -> None:
        """Trigger directory scanning and indexing in a background thread."""
        folder_path = QFileDialog.getExistingDirectory(
            self,
            "Select Folder to Index",
            str(Path(__file__).resolve().parent.parent.parent / "Sample_files")
        )
        if not folder_path:
            return

        self.status_bar.showMessage(f"Indexing Folder: {folder_path}...")
        self.btn_index.setEnabled(False)
        self.btn_index.setText("Indexing...")

        self._index_worker = IndexWorker(self.indexer, folder_path)
        self._index_worker.finished.connect(self._on_indexing_finished)
        self._index_worker.error.connect(self._on_indexing_error)
        self._index_worker.start()

    def _on_indexing_finished(self, summary: dict) -> None:
        self.btn_index.setEnabled(True)
        self.btn_index.setText("⊕ Index Folder")
        
        # Reload all database files to include newly indexed entries
        self.load_all_files()

        msg = (
            f"Successfully indexed: {summary['indexed']} files\n"
            f"Skipped (empty/unsupported): {summary['skipped']}\n"
            f"Failed: {summary['failed']}"
        )
        QMessageBox.information(self, "Indexing Completed", msg)
        self.status_bar.showMessage(f"Indexing completed. {summary['indexed']} files added/updated.")

    def _on_indexing_error(self, error: str) -> None:
        self.btn_index.setEnabled(True)
        self.btn_index.setText("⊕ Index Folder")
        self.status_bar.showMessage("Indexing failed.")
        QMessageBox.critical(self, "Indexing Pipeline Error", f"Failed to run indexing pipeline:\n{error}")

    def reset_entire_database(self) -> None:
        """Reset databases and clear indexes."""
        reply = QMessageBox.question(
            self,
            "Reset Database",
            "Are you absolutely sure you want to delete all indexed data? This cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            try:
                self.db_manager.clear_all()
                self.faiss_manager.reset()
                self.faiss_manager.save()
                self.results_list.clear()
                self.preview_area.clear()
                self._results = []
                self.btn_open_file.setEnabled(False)
                self.load_all_files()  # Reloads (will show empty list)
                self.status_bar.showMessage("Database reset complete.")
            except Exception as e:
                QMessageBox.critical(self, "Reset Error", f"Failed to reset data:\n{e}")

    def ask_ai_about_selected_file(self) -> None:
        """Stream conversational Q&A on selected file chunks."""
        current_row = self.results_list.currentRow()

        if current_row < 0:
            QMessageBox.warning(self, "No File Selected", "Please select a file first.")
            return

        question = self.chat_input.text().strip()
        if not question:
            return

        item = self.results_list.item(current_row)
        actual_index = item.data(Qt.ItemDataRole.UserRole)

        if actual_index is None or actual_index >= len(self._results):
            return

        result = self._results[actual_index]
        content = result.get("content", "").strip()

        if not content:
            self.chat_output.setPlainText("Selected file has no readable text content.")
            return

        self.chat_output.setPlainText("Thinking...")
        self.btn_chat.setEnabled(False)
        self.btn_chat.setText("Thinking...")

        file_id = result.get("id")

        try:
            answer = self.chat_engine.ask_about_file(content, question, file_id=file_id)
            self.chat_output.setMarkdown(answer)
            self.chat_input.clear()
        except Exception as e:
            QMessageBox.critical(self, "AI Error", f"Failed to get AI response:\n{str(e)}")
        finally:
            self.btn_chat.setEnabled(True)
            self.btn_chat.setText("Ask AI")

    def open_file_native(self) -> None:
        """Open document in default OS utility."""
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