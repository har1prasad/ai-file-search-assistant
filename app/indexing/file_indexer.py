"""
file_indexer.py — Indexing orchestrator for AI File Search Assistant.

Coordinates the complete indexing pipeline for a given folder:
    1. Recursively scan for files.
    2. Extract text using extractor.py.
    3. Store metadata and content in SQLite via db_manager.py.
    4. Generate semantic embeddings via embedding_manager.py.
    5. Add embeddings to FAISS via faiss_manager.py.
    6. Save the FAISS index once all files are processed.

Example usage:
    from app.indexing.file_indexer import FileIndexer

    indexer = FileIndexer()
    summary = indexer.index_folder("/home/hari/Documents")
    print(summary)
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from app.database.db_manager import DatabaseManager
from app.embeddings.embedding_manager import EmbeddingManager
from app.extraction.extractor import extract_text, get_supported_extensions
from app.search.faiss_manager import FAISSManager

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FileIndexer
# ---------------------------------------------------------------------------


class FileIndexer:
    """Orchestrates the full file indexing pipeline.

    Scans a folder recursively, extracts text from supported files,
    stores metadata in SQLite, and indexes semantic embeddings in FAISS.

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
        logger.info("FileIndexer initialised.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def index_folder(self, folder_path: str | Path) -> dict[str, int]:
        """Recursively index all supported files in a folder.

        Args:
            folder_path: Path to the folder to index.

        Returns:
            A summary dict with keys:
                - total_files : total files discovered
                - indexed     : successfully indexed
                - skipped     : unsupported or empty content
                - failed      : exceptions during processing

        Raises:
            ValueError: If folder_path does not exist or is not a directory.
        """
        folder = Path(folder_path).resolve()

        if not folder.exists():
            raise ValueError(f"Folder does not exist: {folder}")
        if not folder.is_dir():
            raise ValueError(f"Path is not a directory: {folder}")

        logger.info("Indexing started. folder='%s'", folder)

        total_files = 0
        indexed = 0
        skipped = 0
        failed = 0

        for file_path in sorted(folder.rglob("*")):
            if not file_path.is_file():
                continue

            total_files += 1
            logger.info("Processing: %s", file_path.name)

            try:
                success = self._process_file(file_path)
                if success:
                    indexed += 1
                else:
                    skipped += 1
            except Exception as exc:
                logger.error("Failed to process '%s': %s", file_path.name, exc)
                failed += 1

        # Save FAISS index once after all files are processed
        self.faiss_manager.save()

        summary = {
            "total_files": total_files,
            "indexed": indexed,
            "skipped": skipped,
            "failed": failed,
        }

        logger.info(
            "Indexing complete. total=%d indexed=%d skipped=%d failed=%d",
            total_files, indexed, skipped, failed,
        )

        return summary

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

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
            1. Check if file extension is supported.
            2. Extract text content.
            3. Skip if content is empty.
            4. Store metadata + content in SQLite.
            5. Generate embedding.
            6. Add embedding to FAISS.

        Args:
            file_path: Path object pointing to the file.

        Returns:
            True if the file was successfully indexed.
            False if it was skipped due to unsupported type or empty content.
        """
        # Skip unsupported extensions
        if file_path.suffix.lower() not in self._supported:
            logger.warning("Unsupported type, skipping: %s", file_path.name)
            return False

        # Extract text
        content = extract_text(str(file_path))
        if not content:
            logger.warning("Empty content, skipping: %s", file_path.name)
            return False

        # Collect metadata
        metadata = self._collect_metadata(file_path)

        # Store in SQLite
        file_id = self.db_manager.upsert_file(
            path=metadata["file_path"],
            filename=metadata["file_name"],
            extension=metadata["file_type"],
            size=metadata["size"],
            modified_time=metadata["modified_time"],
            content=content,
        )

        if file_id == -1:
            logger.error("DB upsert failed for: %s", file_path.name)
            return False

        # Generate and store embedding
        embedding = self.embedding_manager.generate_embedding(content)
        self.faiss_manager.add_embedding(file_id=file_id, embedding=embedding)

        logger.info("Indexed: %s (file_id=%d)", file_path.name, file_id)
        return True


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    indexer = FileIndexer()
    summary = indexer.index_folder("Sample_files")

    print("\nIndexing Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")