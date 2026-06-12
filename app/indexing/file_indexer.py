"""
file_indexer.py — Indexing orchestrator for AI File Search Assistant.

Coordinates the complete indexing pipeline for a given folder:
    1. Recursively scan for files.
    2. Detect and remove records for files that no longer exist on disk.
    3. Extract text using extractor.py.
    4. Store metadata and content in SQLite via db_manager.py.
    5. Generate semantic embeddings via embedding_manager.py.
    6. Add embeddings to FAISS via faiss_manager.py.
    7. Save the FAISS index once all files are processed.

Supports cancellation mid-run and an optional progress callback so a GUI
(e.g. PySide6) can show "current file / total files" while indexing.

Example usage:
    from app.indexing.file_indexer import FileIndexer

    indexer = FileIndexer()
    summary = indexer.index_folder(
        "/home/hari/Documents",
        progress_callback=lambda current, total: print(f"{current}/{total}"),
    )
    print(summary)
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.database.db_manager import DatabaseManager
from app.embeddings.embedding_manager import EmbeddingManager
from app.extraction.extractor import extract_text, get_supported_extensions
from app.search.faiss_manager import FAISSManager

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

# Files larger than this are skipped to avoid huge extraction/embedding times.
MAX_FILE_SIZE_MB = 50


# ---------------------------------------------------------------------------
# FileIndexer
# ---------------------------------------------------------------------------


class FileIndexer:
    """Orchestrates the full file indexing pipeline.

    Scans a folder recursively, extracts text from supported files,
    stores metadata in SQLite, and indexes semantic embeddings in FAISS.
    Also cleans up entries for files that have been deleted from disk,
    and re-indexes files whose content has changed since the last run.

    Args:
        db_manager:        DatabaseManager instance. Created if not provided.
        embedding_manager: EmbeddingManager instance. Created if not provided.
        faiss_manager:     FAISSManager instance. Created if not provided.
    """

    def __init__(
        self,
        db_manager: DatabaseManager | None = None,
        embedding_manager: EmbeddingManager | None = None,
        faiss_manager: FAISSManager | None = None,
    ) -> None:
        self.db_manager = db_manager or DatabaseManager()
        self.embedding_manager = embedding_manager or EmbeddingManager()
        self.faiss_manager = faiss_manager or FAISSManager()
        self._supported = get_supported_extensions()

        # Set to True (via cancel()) to stop index_folder() between files.
        self.cancel_requested = False

        logger.info("FileIndexer initialized.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def cancel(self) -> None:
        """Request that an in-progress index_folder() run stop early.

        The cancellation is checked once per file inside the main loop,
        so the current file finishes processing before the run exits.
        """
        self.cancel_requested = True
        logger.warning("Indexing cancellation requested.")

    def index_folder(
        self,
        folder_path: str | Path,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> dict[str, int]:
        """Recursively index all supported files in a folder.

        Steps performed:
            1. Validate that folder_path exists and is a directory.
            2. Remove DB/FAISS records for files that no longer exist
               on disk (see _cleanup_deleted_files).
            3. Walk every file in the folder, processing each one via
               _process_file (extract, store, embed).
            4. Save the FAISS index once after all files are processed.

        Args:
            folder_path: Path to the folder to index.
            progress_callback: Optional callable invoked as
                progress_callback(current, total) before each file is
                processed, where `current` is the 1-based index of the
                file and `total` is the total number of files found.
                Useful for driving a GUI progress bar.

        Returns:
            A summary dict with keys:
                - total_files : total files discovered
                - indexed     : successfully indexed (new or updated)
                - skipped     : unsupported, unchanged, too large, or empty
                - failed      : exceptions during processing
                - deleted     : DB/FAISS records removed for missing files

        Raises:
            ValueError: If folder_path does not exist or is not a directory.
        """

        folder = Path(folder_path).resolve()

        if not folder.exists():
            raise ValueError(f"Folder does not exist: {folder}")
        if not folder.is_dir():
            raise ValueError(f"Path is not a directory: {folder}")

        logger.info("Indexing started: %s", folder)

        # Collect every file in the folder tree up front so we know the
        # total count (needed for the progress callback).
        all_files = [p for p in sorted(folder.rglob("*")) if p.is_file()]
        total_files = len(all_files)

        indexed = 0
        skipped = 0
        failed = 0
        deleted = 0

        # Cleanup deleted files first, so stale entries don't linger even
        # if the run is cancelled partway through the main loop below.
        deleted = self._cleanup_deleted_files(folder)

        for current, file_path in enumerate(all_files, start=1):
            # Allow the caller to stop the run between files via cancel().
            if self.cancel_requested:
                logger.warning("Indexing cancelled by user.")
                break

            logger.info("[%d/%d] Processing: %s", current, total_files, file_path.name)

            if progress_callback:
                progress_callback(current, total_files)

            try:
                success = self._process_file(file_path)

                if success is True:
                    indexed += 1
                else:
                    skipped += 1

            except Exception as exc:
                logger.error("Failed to process '%s': %s", file_path.name, exc)
                failed += 1

        # Save FAISS index once after all files are processed (or cancelled).
        self.faiss_manager.save()

        summary = {
            "total_files": total_files,
            "indexed": indexed,
            "skipped": skipped,
            "failed": failed,
            "deleted": deleted,
        }

        logger.info("Indexing complete: %s", summary)
        return summary

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def _cleanup_deleted_files(self, folder: Path) -> int:
        """Remove DB and FAISS entries for files that no longer exist.

        Looks at every path previously stored in the database that falls
        under `folder`. If the file is missing from disk, its database
        row and corresponding FAISS embedding are removed.

        Args:
            folder: The folder currently being indexed. Only DB records
                whose path starts with this folder are checked.

        Returns:
            The number of records removed.
        """
        deleted_count = 0

        db_paths = self.db_manager.get_all_paths()

        for path_str in db_paths:
            path = Path(path_str)

            # Only consider records inside the folder being indexed, and
            # only act if the file is actually gone from disk.
            if str(path).startswith(str(folder)) and not path.exists():
                record = self.db_manager.get_file_by_path(path_str)

                if record:
                    file_id = record["id"]
                    self.faiss_manager.remove(file_id)
                    self.db_manager.delete_file(path_str)
                    deleted_count += 1
                    logger.info("Removed deleted file: %s", path.name)

        return deleted_count

    def _collect_metadata(self, file_path: Path) -> dict[str, Any]:
        """Collect file system metadata for a given file.

        Args:
            file_path: Path object pointing to the file.

        Returns:
            A dict containing file_name, file_path, file_type,
            size, and modified_time.
        """
        stat = file_path.stat()

        return {
            "file_name": file_path.name,
            "file_path": str(file_path.resolve()),
            "file_type": file_path.suffix.lower(),
            "size": stat.st_size,
            "modified_time": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        }

    def _process_file(self, file_path: Path) -> bool:
        """Process a single file through the full indexing pipeline.

        Steps:
            1. Skip if the file extension is not supported.
            2. Skip if the file is larger than MAX_FILE_SIZE_MB.
            3. Skip if the file is already in the DB and unchanged
               (same modified_time as last indexed).
            4. Extract text content; skip if extraction returns nothing.
            5. Store metadata + content in SQLite (insert or update).
            6. If this is a re-index of an existing file, remove its old
               FAISS embedding first so it isn't duplicated.
            7. Generate a new embedding and add it to FAISS.

        Args:
            file_path: Path object pointing to the file.

        Returns:
            True if the file was successfully indexed (new or updated).
            False if it was skipped (unsupported type, too large,
            unchanged since last run, empty content, or a DB error).
        """
        ext = file_path.suffix.lower()

        # Skip unsupported extensions.
        if ext not in self._supported:
            logger.debug("Unsupported file type: %s", file_path.name)
            return False

        # Skip files that are too large to process efficiently.
        size_mb = file_path.stat().st_size / (1024 * 1024)

        if size_mb > MAX_FILE_SIZE_MB:
            logger.warning(
                "Skipping large file (%0.2f MB): %s",
                size_mb,
                file_path.name,
            )
            return False

        metadata = self._collect_metadata(file_path)

        # If the file is already indexed and hasn't changed since last
        # time, there's nothing to do.
        existing = self.db_manager.get_file_by_path(metadata["file_path"])

        if existing and existing["modified_time"] == metadata["modified_time"]:
            logger.info("Skipping unchanged file: %s", file_path.name)
            return False

        # Extract text content from the file.
        content = extract_text(str(file_path))

        if not content:
            logger.warning("Empty content: %s", file_path.name)
            return False

        # Store/update metadata + content in SQLite.
        file_id = self.db_manager.upsert_file(
            path=metadata["file_path"],
            filename=metadata["file_name"],
            extension=metadata["file_type"],
            size=metadata["size"],
            modified_time=metadata["modified_time"],
            content=content,
        )

        if file_id == -1:
            logger.error("Database upsert failed: %s", file_path.name)
            return False

        # If this file was already indexed, remove its old embedding
        # before adding the new one to avoid duplicates in FAISS.
        if existing:
            self.faiss_manager.remove(file_id)

        # Generate and store the new embedding.
        embedding = self.embedding_manager.generate_embedding(content)
        self.faiss_manager.add_embedding(file_id, embedding)

        logger.info("Indexed successfully: %s (id=%d)", file_path.name, file_id)
        return True