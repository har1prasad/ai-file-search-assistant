"""
main.py — Entry point for AI File Search Assistant.

Run from the project root:
    python main.py
"""

import sys
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt
from app.ui.main_window import MainWindow



if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setApplicationName("AI File Search Assistant")
    app.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())