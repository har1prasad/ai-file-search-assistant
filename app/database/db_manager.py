"""
db_manager.py — SQLite metadata store for AI File Search Assistant.

Manages all database operations for the `files` table.
Stores file metadata and extracted text content.
Semantic embeddings are handled separately by FAISS.

Usage:
    from app.database import DatabaseManager

    db = DatabaseManager()
    file_id = db.upsert_file(
        path="docs/notes.txt",
        filename="notes.txt",
        extension=".txt",
        size=2048,
        modified_time="2026-05-15 10:00:00",
        content="Some extracted text here.",
    )
    record = db.get_file_by_id(file_id)
    print(record["filename"])
    db.close()
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path


# Logging

logger = logging.getLogger(__name__)


# SQL statements

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS files (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    path          TEXT    UNIQUE NOT NULL,
    filename      TEXT    NOT NULL,
    extension     TEXT,
    size          INTEGER,
    modified_time TEXT,
    content       TEXT,
    indexed_at    TEXT    DEFAULT CURRENT_TIMESTAMP
);
"""

_UPSERT_FILE = """
INSERT INTO files (path, filename, extension, size, modified_time, content)
VALUES (?, ?, ?, ?, ?, ?)
ON CONFLICT(path) DO UPDATE SET
    filename      = excluded.filename,
    extension     = excluded.extension,
    size          = excluded.size,
    modified_time = excluded.modified_time,
    content       = excluded.content,
    indexed_at    = CURRENT_TIMESTAMP;
"""


# DatabaseManager

class DatabaseManager:
    """Manages all SQLite operations for the AI File Search Assistant.

    Creates the database and the `files` table automatically on first use.
    All public methods return plain Python dicts for ease of use downstream.

    Args:
        db_path: Path to the SQLite database file.
                 Defaults to "data/metadata.db".
    """

    def __init__(self, db_path: str = "data/metadata.db") -> None:
        self._db_path = Path(db_path).resolve()
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_db()


    # Connection

    def connect(self) -> sqlite3.Connection:
        """Open and return a new SQLite connection.

        Rows are returned as sqlite3.Row objects, which behave like dicts.
        Use as a context manager so commits and closes are handled automatically.

        Returns:
            An open sqlite3.Connection instance.
        """
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        return conn


    # Initialisation

    def init_db(self) -> None:
        """Create the `files` table if it does not already exist."""
        try:
            with self.connect() as conn:
                conn.execute(_CREATE_TABLE)
        except sqlite3.Error as exc:
            logger.error("Failed to initialise database: %s", exc)


    # Write operations

    def upsert_file(
        self,
        path: str,
        filename: str,
        extension: str,
        size: int,
        modified_time: str,
        content: str,
    ) -> int:
        """Insert a new file record or update it if the path already exists.

        Args:
            path:          Absolute or relative path used as the unique key.
            filename:      File name with extension (e.g. "notes.txt").
            extension:     Lowercase file extension (e.g. ".txt").
            size:          File size in bytes.
            modified_time: Last-modified timestamp as an ISO-format string.
            content:       Extracted text content.

        Returns:
            The row ID of the inserted or updated record.
            Returns -1 if the operation fails.
        """
        try:
            with self.connect() as conn:
                cursor = conn.execute(
                    _UPSERT_FILE,
                    (path, filename, extension, size, modified_time, content),
                )
                # lastrowid is None on UPDATE; fetch the id explicitly.
                if cursor.lastrowid:
                    return cursor.lastrowid
                row = conn.execute(
                    "SELECT id FROM files WHERE path = ?", (path,)
                ).fetchone()
                return row["id"] if row else -1
        except sqlite3.Error as exc:
            logger.error("upsert_file failed for '%s': %s", path, exc)
            return -1

    def delete_file(self, path: str) -> None:
        """Delete a file record by its path.

        Args:
            path: The unique file path to remove.
        """
        try:
            with self.connect() as conn:
                conn.execute("DELETE FROM files WHERE path = ?", (path,))
        except sqlite3.Error as exc:
            logger.error("delete_file failed for '%s': %s", path, exc)

    def clear_all(self) -> None:
        """Delete every record from the `files` table.

        Use with caution — this cannot be undone.
        """
        try:
            with self.connect() as conn:
                conn.execute("DELETE FROM files")
        except sqlite3.Error as exc:
            logger.error("clear_all failed: %s", exc)


    # Read operations

    def get_all_paths(self) -> set[str]:
        """Return all indexed file paths."""
        try:
            with self.connect() as conn:
                rows = conn.execute("SELECT path FROM files").fetchall()
                return {row["path"] for row in rows}
        except sqlite3.Error as exc:
            logger.error("get_all_paths failed: %s", exc)
            return set()


    def get_file_by_id(self, file_id: int) -> dict | None:
        """Retrieve a file record by its primary key.

        Args:
            file_id: The integer ID of the record.

        Returns:
            A dict of column values, or None if not found.
        """
        try:
            with self.connect() as conn:
                row = conn.execute(
                    "SELECT * FROM files WHERE id = ?", (file_id,)
                ).fetchone()
                return dict(row) if row else None
        except sqlite3.Error as exc:
            logger.error("get_file_by_id failed for id=%s: %s", file_id, exc)
            return None
        
    
    def get_file_by_path(self, path: str) -> dict | None:
        """Retrieve a file record by its unique path.

        Args:
            path: The file path used as the unique key.

        Returns:
            A dict of column values, or None if not found.
        """
        try:
            with self.connect() as conn:
                row = conn.execute(
                    "SELECT * FROM files WHERE path = ?", (path,)
                ).fetchone()
                return dict(row) if row else None
        except sqlite3.Error as exc:
            logger.error("get_file_by_path failed for '%s': %s", path, exc)
            return None

    def get_all_files(self) -> list[dict]:
        """Retrieve all file records ordered by indexed_at descending.

        Returns:
            A list of dicts, one per file. Empty list on error or no records.
        """
        try:
            with self.connect() as conn:
                rows = conn.execute(
                    "SELECT * FROM files ORDER BY indexed_at DESC"
                ).fetchall()
                return [dict(row) for row in rows]
        except sqlite3.Error as exc:
            logger.error("get_all_files failed: %s", exc)
            return []

    def count_files(self) -> int:
        """Return the total number of indexed file records.

        Returns:
            Row count as an integer. Returns 0 on error.
        """
        try:
            with self.connect() as conn:
                row = conn.execute("SELECT COUNT(*) FROM files").fetchone()
                return row[0] if row else 0
        except sqlite3.Error as exc:
            logger.error("count_files failed: %s", exc)
            return 0


    # Cleanup

    def close(self) -> None:
        """No-op placeholder for API consistency.

        Connections are opened and closed per-operation via context managers,
        so there is no persistent connection to close. Call this at shutdown
        if you later switch to a persistent connection model.
        """


# Quick smoke-test

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    base = Path(__file__).resolve().parent.parent.parent
    db = DatabaseManager(db_path=str(base / "data" / "metadata.db"))

    file_id = db.upsert_file(
        path="sample_files/cover_letter.pdf",
        filename="cover_letter.pdf",
        extension=".pdf",
        size=20480,
        modified_time="2026-05-15 10:00:00",
        content="This is a sample cover letter for testing purposes.",
    )

    record = db.get_file_by_id(file_id)
    print(f"Inserted record  : {record}")
    print(f"Total files in DB: {db.count_files()}")


# ============================================
# TESTING CHECKLIST - File Operations
# ============================================

# ----- Basic Operations -----
# Test: Insert a new record → verify it returns a valid id (not None, > 0)
# Test: Fetch by id → verify all fields match inserted data
# Test: Fetch by path → verify same fields match inserted data  
# Test: Count after insert → should return 1

# ----- Upsert Behaviour (CRITICAL) -----
# Test: Insert same path again with DIFFERENT content
# Test: Fetch by id → content should be UPDATED, id should REMAIN SAME
# Test: This confirms ON CONFLICT (path) DO UPDATE works correctly

# ----- Edge Cases -----
# Test: get_file_by_id(999) → should return None (not crash/raise exception)
# Test: get_file_by_path("nonexistent/path.txt") → should return None
# Test: get_all_files() on empty table → should return [] (empty list, not None)

# ----- Cleanup -----
# Test: delete_file(path) → count should decrease by 1
# Test: clear_all() → count should return 0 (table fully empty)