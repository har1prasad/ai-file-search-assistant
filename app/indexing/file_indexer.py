"""
file_indexer.py — Optimized parallel indexing pipeline with chunk-level storage and transactions.
"""

from __future__ import annotations

import logging
import concurrent.futures
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from app.database.db_manager import DatabaseManager
from app.embeddings.embedding_manager import EmbeddingManager
from app.extraction.extractor import extract_with_metadata, chunk_text, get_supported_extensions
from app.search.faiss_manager import FAISSManager

logger = logging.getLogger(__name__)

MAX_FILE_SIZE_MB = 50


class FileIndexer:
    """Indexes files recursively into SQLite + FAISS at the chunk level."""

    def __init__(
        self,
        db_manager: DatabaseManager,
        embedding_manager: EmbeddingManager,
        faiss_manager: FAISSManager,
    ) -> None:
        self.db_manager = db_manager
        self.embedding_manager = embedding_manager
        self.faiss_manager = faiss_manager
        self._supported = get_supported_extensions()
        self.cancel_requested = False

        logger.info("FileIndexer initialized.")

    def cancel(self) -> None:
        self.cancel_requested = True
        logger.warning("Indexing cancellation requested.")

    def index_folder(
        self,
        folder_path: str | Path,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> dict[str, int]:
        folder = Path(folder_path).resolve()

        if not folder.exists():
            raise ValueError(f"Folder does not exist: {folder}")

        if not folder.is_dir():
            raise ValueError(f"Not a directory: {folder}")

        logger.info("Indexing started: %s", folder)
        self.cancel_requested = False

        # 1. Clean up deleted files from DB & FAISS
        deleted = self._cleanup_deleted_files(folder)

        # 2. Gather all files in the folder and determine which ones need indexing
        all_files = [p for p in folder.rglob("*") if p.is_file()]
        
        # Read existing records from DB to check modification timestamps
        existing_files = {r["path"]: r for r in self.db_manager.get_all_files()}
        
        files_to_process = []
        for file_path in all_files:
            ext = file_path.suffix.lower()
            if ext not in self._supported:
                continue

            size_mb = file_path.stat().st_size / (1024 * 1024)
            if size_mb > MAX_FILE_SIZE_MB:
                logger.warning("Skipping large file: %s", file_path.name)
                continue

            metadata = self._collect_metadata(file_path)
            path_str = metadata["file_path"]
            existing = existing_files.get(path_str)

            needs_index = True
            if existing and existing["modified_time"] == metadata["modified_time"]:
                # Check if it already has chunks in the DB (for backward compatibility / upgrade)
                chunks = self.db_manager.get_chunks_by_file_id(existing["id"])
                if len(chunks) > 0:
                    needs_index = False

            if needs_index:
                files_to_process.append((file_path, metadata, existing))

        total_to_process = len(files_to_process)
        if total_to_process == 0:
            logger.info("All files are up to date.")
            return {
                "total_files": len(all_files),
                "indexed": 0,
                "skipped": len(all_files) - deleted,
                "failed": 0,
                "deleted": deleted,
            }

        # 3. Parallel text extraction using a ThreadPoolExecutor
        logger.info("Extracting text from %d files in parallel...", total_to_process)
        extracted_results = []
        failed = 0
        completed = 0

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(extract_with_metadata, str(item[0].resolve()), chunk=True): item
                for item in files_to_process
            }

            for future in concurrent.futures.as_completed(futures):
                if self.cancel_requested:
                    logger.warning("Indexing cancelled during extraction stage.")
                    break

                file_path, metadata, existing = futures[future]
                completed += 1
                if progress_callback:
                    progress_callback(completed, total_to_process)

                try:
                    res = future.result()
                    if res and res.content.strip():
                        extracted_results.append((file_path, metadata, existing, res))
                    else:
                        logger.warning("No text content could be extracted from: %s", file_path.name)
                        failed += 1
                except Exception as exc:
                    logger.exception("Parallel extraction failed for %s: %s", file_path.name, exc)
                    failed += 1

        if self.cancel_requested:
            logger.warning("Indexing aborted.")
            return {
                "total_files": len(all_files),
                "indexed": len(extracted_results),
                "skipped": total_to_process - len(extracted_results),
                "failed": failed,
                "deleted": deleted,
            }

        # 4. Save metadata and chunks inside a single transactional database session
        indexed_count = 0
        all_chunks_to_embed = []

        logger.info("Saving metadata and text chunks to SQLite...")
        conn = self.db_manager.connect()
        try:
            conn.execute("BEGIN TRANSACTION;")

            for file_path, metadata, existing, res in extracted_results:
                if self.cancel_requested:
                    break

                # Remove old chunks from DB and FAISS if modifying an existing file
                if existing:
                    file_id = existing["id"]
                    old_chunks = self.db_manager.get_chunks_by_file_id(file_id, conn=conn)
                    old_chunk_ids = [c["id"] for c in old_chunks]
                    if old_chunk_ids:
                        self.faiss_manager.remove_ids(old_chunk_ids)
                    self.db_manager.delete_chunks_by_file_id(file_id, conn=conn)

                # Upsert file record
                file_id = self.db_manager.upsert_file(
                    path=metadata["file_path"],
                    filename=metadata["file_name"],
                    extension=metadata["file_type"],
                    size=metadata["size"],
                    modified_time=metadata["modified_time"],
                    content=res.content,
                    conn=conn,
                )

                if file_id == -1:
                    logger.error("Failed to upsert file metadata: %s", file_path.name)
                    failed += 1
                    continue

                # Store text chunks
                chunks = res.chunks if res.chunks else chunk_text(res.content)
                if not chunks:
                    chunks = [res.content]  # Fallback to single chunk

                for i, chunk_text_content in enumerate(chunks):
                    chunk_id = self.db_manager.insert_chunk(
                        file_id=file_id,
                        chunk_index=i,
                        content=chunk_text_content,
                        conn=conn,
                    )
                    if chunk_id != -1:
                        all_chunks_to_embed.append((chunk_id, chunk_text_content))

                indexed_count += 1
                logger.info("Indexed document chunks: %s", file_path.name)

            conn.execute("COMMIT;")
        except Exception as exc:
            conn.execute("ROLLBACK;")
            logger.exception("Database transaction failed, rollback triggered: %s", exc)
            raise
        finally:
            conn.close()

        # 5. Batch generate embeddings for chunks and update FAISS index
        if all_chunks_to_embed and not self.cancel_requested:
            logger.info("Generating embeddings for %d chunks in batch...", len(all_chunks_to_embed))
            chunk_ids = [item[0] for item in all_chunks_to_embed]
            chunk_texts = [item[1] for item in all_chunks_to_embed]

            embeddings = self.embedding_manager.generate_embeddings(chunk_texts)

            logger.info("Adding %d chunk embeddings to FAISS index...", len(chunk_ids))
            self.faiss_manager.add_embeddings(chunk_ids, embeddings)
            self.faiss_manager.save()

        summary = {
            "total_files": len(all_files),
            "indexed": indexed_count,
            "skipped": total_to_process - indexed_count,
            "failed": failed,
            "deleted": deleted,
        }

        logger.info("Indexing complete: %s", summary)
        return summary

    def _cleanup_deleted_files(self, folder: Path) -> int:
        deleted_count = 0
        db_paths = self.db_manager.get_all_paths()

        for path_str in db_paths:
            path = Path(path_str)

            if str(path).startswith(str(folder)) and not path.exists():
                record = self.db_manager.get_file_by_path(path_str)

                if record:
                    file_id = record["id"]
                    # Fetch and delete associated chunk vectors from FAISS
                    chunks = self.db_manager.get_chunks_by_file_id(file_id)
                    chunk_ids = [c["id"] for c in chunks]
                    if chunk_ids:
                        self.faiss_manager.remove_ids(chunk_ids)

                    # Delete file from SQLite (which cascade deletes chunks)
                    self.db_manager.delete_file(path_str)
                    deleted_count += 1

        return deleted_count

    def _collect_metadata(self, file_path: Path) -> dict[str, Any]:
        stat = file_path.stat()

        return {
            "file_name": file_path.name,
            "file_path": str(file_path.resolve()),
            "file_type": file_path.suffix.lower(),
            "size": stat.st_size,
            "modified_time": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        }