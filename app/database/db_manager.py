"""
db_manager.py — SQLite metadata store for AI File Search Assistant.

Manages all database operations for the `files`, `chunks`, and `chunks_fts` tables.
Stores file metadata, extracted text content, text chunks, and FTS5 search index.
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

_CREATE_FILES_TABLE = """
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

_CREATE_CHUNKS_TABLE = """
CREATE TABLE IF NOT EXISTS chunks (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id       INTEGER NOT NULL,
    chunk_index   INTEGER NOT NULL,
    content       TEXT    NOT NULL,
    FOREIGN KEY(file_id) REFERENCES files(id) ON DELETE CASCADE
);
"""

_CREATE_FTS_TABLE = """
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    content,
    tokenize="porter unicode61"
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

    Creates the database and files/chunks tables automatically on first use.
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
        Enables Foreign Key support to automatically clean up chunks on cascade.

        Returns:
            An open sqlite3.Connection instance.
        """
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn


    # Initialisation

    def init_db(self) -> None:
        """Create the tables and virtual FTS indexes if they do not already exist."""
        try:
            with self.connect() as conn:
                conn.execute(_CREATE_FILES_TABLE)
                conn.execute(_CREATE_CHUNKS_TABLE)
                conn.execute(_CREATE_FTS_TABLE)
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
        conn: sqlite3.Connection | None = None,
    ) -> int:
        """Insert a new file record or update it if the path already exists.

        Args:
            path:          Absolute or relative path used as the unique key.
            filename:      File name with extension (e.g. "notes.txt").
            extension:     Lowercase file extension (e.g. ".txt").
            size:          File size in bytes.
            modified_time: Last-modified timestamp as an ISO-format string.
            content:       Extracted text content.
            conn:          Optional active SQLite connection to reuse.

        Returns:
            The row ID of the inserted or updated record.
            Returns -1 if the operation fails.
        """
        def _run(db_conn: sqlite3.Connection) -> int:
            cursor = db_conn.execute(
                _UPSERT_FILE,
                (path, filename, extension, size, modified_time, content),
            )
            if cursor.lastrowid:
                return cursor.lastrowid
            row = db_conn.execute(
                "SELECT id FROM files WHERE path = ?", (path,)
            ).fetchone()
            return row["id"] if row else -1

        if conn is not None:
            return _run(conn)

        try:
            with self.connect() as new_conn:
                return _run(new_conn)
        except sqlite3.Error as exc:
            logger.error("upsert_file failed for '%s': %s", path, exc)
            return -1

    def delete_file(self, path: str, conn: sqlite3.Connection | None = None) -> None:
        """Delete a file record by its path.

        Args:
            path: The unique file path to remove.
            conn: Optional active SQLite connection to reuse.
        """
        def _run(db_conn: sqlite3.Connection) -> None:
            db_conn.execute("DELETE FROM files WHERE path = ?", (path,))

        if conn is not None:
            _run(conn)
            return

        try:
            with self.connect() as new_conn:
                _run(new_conn)
        except sqlite3.Error as exc:
            logger.error("delete_file failed for '%s': %s", path, exc)

    def clear_all(self) -> None:
        """Delete every record from the `files`, `chunks`, and `chunks_fts` tables.

        Use with caution — this cannot be undone.
        """
        try:
            with self.connect() as conn:
                conn.execute("DELETE FROM files")
                conn.execute("DELETE FROM chunks")
                conn.execute("DELETE FROM chunks_fts")
        except sqlite3.Error as exc:
            logger.error("clear_all failed: %s", exc)


    # Chunk operations

    def insert_chunk(
        self,
        file_id: int,
        chunk_index: int,
        content: str,
        conn: sqlite3.Connection | None = None,
    ) -> int:
        """Insert a new text chunk for a file and index it in FTS5.

        Args:
            file_id:     The foreign key mapping to the file's ID.
            chunk_index: The sequential index of this chunk.
            content:     The text content of the chunk.
            conn:        Optional active SQLite connection to reuse.

        Returns:
            The row ID of the inserted chunk record, or -1 on failure.
        """
        def _run(db_conn: sqlite3.Connection) -> int:
            cursor = db_conn.execute(
                "INSERT INTO chunks (file_id, chunk_index, content) VALUES (?, ?, ?)",
                (file_id, chunk_index, content),
            )
            chunk_id = cursor.lastrowid if cursor.lastrowid else -1
            if chunk_id != -1:
                # Add to FTS5 index mapping rowid to chunk_id
                db_conn.execute(
                    "INSERT INTO chunks_fts (rowid, content) VALUES (?, ?)",
                    (chunk_id, content),
                )
            return chunk_id

        if conn is not None:
            return _run(conn)

        try:
            with self.connect() as new_conn:
                return _run(new_conn)
        except sqlite3.Error as exc:
            logger.error("insert_chunk failed for file_id=%d: %s", file_id, exc)
            return -1

    def get_chunks_by_file_id(self, file_id: int, conn: sqlite3.Connection | None = None) -> list[dict]:
        """Retrieve all text chunks associated with a file, ordered by index.

        Args:
            file_id: The ID of the parent file.
            conn:    Optional active SQLite connection to reuse.

        Returns:
            List of dict representation of chunk rows.
        """
        def _run(db_conn: sqlite3.Connection) -> list[dict]:
            rows = db_conn.execute(
                "SELECT * FROM chunks WHERE file_id = ? ORDER BY chunk_index",
                (file_id,),
            ).fetchall()
            return [dict(row) for row in rows]

        if conn is not None:
            return _run(conn)

        try:
            with self.connect() as new_conn:
                return _run(new_conn)
        except sqlite3.Error as exc:
            logger.error("get_chunks_by_file_id failed for file_id=%d: %s", file_id, exc)
            return []

    def delete_chunks_by_file_id(self, file_id: int, conn: sqlite3.Connection | None = None) -> None:
        """Delete all chunks associated with a file, cleaning up FTS indexes.

        Args:
            file_id: The ID of the parent file.
            conn:    Optional active SQLite connection to reuse.
        """
        def _run(db_conn: sqlite3.Connection) -> None:
            # Query chunks to clean up FTS
            rows = db_conn.execute("SELECT id FROM chunks WHERE file_id = ?", (file_id,)).fetchall()
            chunk_ids = [r[0] for r in rows]
            if chunk_ids:
                placeholders = ",".join("?" for _ in chunk_ids)
                db_conn.execute(
                    f"DELETE FROM chunks_fts WHERE rowid IN ({placeholders})",
                    chunk_ids,
                )
            db_conn.execute("DELETE FROM chunks WHERE file_id = ?", (file_id,))

        if conn is not None:
            _run(conn)
            return

        try:
            with self.connect() as new_conn:
                _run(new_conn)
        except sqlite3.Error as exc:
            logger.error("delete_chunks_by_file_id failed for file_id=%d: %s", file_id, exc)

    def get_chunk_by_id(self, chunk_id: int, conn: sqlite3.Connection | None = None) -> dict | None:
        """Retrieve a specific chunk by its ID.

        Args:
            chunk_id: The primary key of the chunk.
            conn:     Optional active SQLite connection to reuse.

        Returns:
            Dict representing the chunk row, or None if not found.
        """
        def _run(db_conn: sqlite3.Connection) -> dict | None:
            row = db_conn.execute("SELECT * FROM chunks WHERE id = ?", (chunk_id,)).fetchone()
            return dict(row) if row else None

        if conn is not None:
            return _run(conn)

        try:
            with self.connect() as new_conn:
                return _run(new_conn)
        except sqlite3.Error as exc:
            logger.error("get_chunk_by_id failed for chunk_id=%d: %s", chunk_id, exc)
            return None


    # Keyword search operations (FTS5)

    def search_fts(self, query: str, limit: int = 30, conn: sqlite3.Connection | None = None) -> list[tuple[int, float]]:
        """Perform exact keyword matching using SQLite FTS5.

        Args:
            query: The search term.
            limit: Maximum number of matches to retrieve.
            conn:  Optional active SQLite connection to reuse.

        Returns:
            List of (chunk_id, bm25_score) matches.
        """
        # Format query for FTS5: wrap individual search terms in double quotes and combine with OR
        words = [f'"{w}"' for w in query.split() if w.strip()]
        cleaned_query = " OR ".join(words)

        if not cleaned_query:
            return []

        def _run(db_conn: sqlite3.Connection) -> list[tuple[int, float]]:
            try:
                # bm25 returns lower values for better matches (usually negative)
                rows = db_conn.execute(
                    "SELECT rowid, bm25(chunks_fts) FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY bm25(chunks_fts) LIMIT ?",
                    (cleaned_query, limit),
                ).fetchall()
                return [(row[0], float(row[1])) for row in rows]
            except sqlite3.Error as exc:
                logger.warning("FTS5 query failed: %s. FTS5 table may be unpopulated.", exc)
                return []

        if conn is not None:
            return _run(conn)

        try:
            with self.connect() as new_conn:
                return _run(new_conn)
        except sqlite3.Error:
            return []


    # Read operations

    def get_all_paths(self, conn: sqlite3.Connection | None = None) -> set[str]:
        """Return all indexed file paths.

        Args:
            conn: Optional active SQLite connection to reuse.
        """
        def _run(db_conn: sqlite3.Connection) -> set[str]:
            rows = db_conn.execute("SELECT path FROM files").fetchall()
            return {row["path"] for row in rows}

        if conn is not None:
            return _run(conn)

        try:
            with self.connect() as new_conn:
                return _run(new_conn)
        except sqlite3.Error as exc:
            logger.error("get_all_paths failed: %s", exc)
            return set()

    def get_file_by_id(self, file_id: int, conn: sqlite3.Connection | None = None) -> dict | None:
        """Retrieve a file record by its primary key.

        Args:
            file_id: The integer ID of the record.
            conn:    Optional active SQLite connection to reuse.

        Returns:
            A dict of column values, or None if not found.
        """
        def _run(db_conn: sqlite3.Connection) -> dict | None:
            row = db_conn.execute(
                "SELECT * FROM files WHERE id = ?", (file_id,)
            ).fetchone()
            return dict(row) if row else None

        if conn is not None:
            return _run(conn)

        try:
            with self.connect() as new_conn:
                return _run(new_conn)
        except sqlite3.Error as exc:
            logger.error("get_file_by_id failed for id=%s: %s", file_id, exc)
            return None

    def get_file_by_path(self, path: str, conn: sqlite3.Connection | None = None) -> dict | None:
        """Retrieve a file record by its unique path.

        Args:
            path: The file path used as the unique key.
            conn: Optional active SQLite connection to reuse.

        Returns:
            A dict of column values, or None if not found.
        """
        def _run(db_conn: sqlite3.Connection) -> dict | None:
            row = db_conn.execute(
                "SELECT * FROM files WHERE path = ?", (path,)
            ).fetchone()
            return dict(row) if row else None

        if conn is not None:
            return _run(conn)

        try:
            with self.connect() as new_conn:
                return _run(new_conn)
        except sqlite3.Error as exc:
            logger.error("get_file_by_path failed for '%s': %s", path, exc)
            return None

    def get_all_files(self, conn: sqlite3.Connection | None = None) -> list[dict]:
        """Retrieve all file records ordered by indexed_at descending.

        Args:
            conn: Optional active SQLite connection to reuse.

        Returns:
            A list of dicts, one per file. Empty list on error or no records.
        """
        def _run(db_conn: sqlite3.Connection) -> list[dict]:
            rows = db_conn.execute(
                "SELECT * FROM files ORDER BY indexed_at DESC"
            ).fetchall()
            return [dict(row) for row in rows]

        if conn is not None:
            return _run(conn)

        try:
            with self.connect() as new_conn:
                return _run(new_conn)
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
        """No-op placeholder for API consistency."""
        pass