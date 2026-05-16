"""
faiss_manager.py — FAISS vector index manager for AI File Search Assistant.

Stores and searches semantic embeddings using Facebook AI Similarity Search (FAISS).
Each embedding corresponds to a file indexed by the application, identified by
its SQLite primary key (file_id).

How this fits into the pipeline:
    1. extractor.py         → extracts raw text from files
    2. db_manager.py        → stores metadata and content in SQLite
    3. embedding_manager.py → converts text into embedding vectors
    4. faiss_manager.py     → stores and searches embeddings  ← this module
    5. search/              → ties everything together for the user

Why IndexFlatIP with normalized embeddings:
    EmbeddingManager normalizes all vectors to unit length.
    For unit vectors, inner product (IP) == cosine similarity.
    IndexIDMap wraps the index so vectors can be stored and removed by file_id,
    which maps directly to the SQLite primary key.

Index file:
    data/faiss.index  — persisted to disk after every write operation.

Example usage:
    from app.search.faiss_manager import FAISSManager
    import numpy as np

    fm = FAISSManager()
    embedding = np.random.rand(384).astype("float32")
    fm.add_embedding(file_id=1, embedding=embedding)
    results = fm.search(query_embedding=embedding, top_k=5)
    print(results)  # [(1, 1.0), ...]
"""

from __future__ import annotations

import logging
from pathlib import Path

import faiss
import numpy as np

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FAISSManager
# ---------------------------------------------------------------------------


class FAISSManager:
    """Manages a FAISS vector index for semantic file search.

    Wraps a faiss.IndexIDMap over faiss.IndexFlatIP so that:
    - Vectors are stored and retrieved by file_id (SQLite primary key).
    - Cosine similarity search is performed via inner product on
      pre-normalized embeddings.

    Args:
        dimension:  Embedding dimension. Must match EmbeddingManager output.
                    Defaults to 384 (all-MiniLM-L6-v2).
        index_path: Path where the FAISS index is saved and loaded.
                    Defaults to "data/faiss.index".
    """

    def __init__(
        self,
        dimension: int = 384,
        index_path: str | Path = "data/faiss.index",
    ) -> None:
        self.dimension = dimension
        self.index_path = Path(index_path).resolve()
        self.index_path.parent.mkdir(parents=True, exist_ok=True)

        if self.index_path.exists():
            logger.info("Existing index found. Loading from: %s", self.index_path)
            self.index = self._load_index()
        else:
            logger.info("No index found. Creating new index at: %s", self.index_path)
            self.index = self._create_index()

    # ------------------------------------------------------------------
    # Index lifecycle
    # ------------------------------------------------------------------

    def _create_index(self) -> faiss.Index:
        """Create and return a new empty FAISS IndexIDMap over IndexFlatIP.

        Returns:
            A fresh faiss.IndexIDMap instance.
        """
        flat_index = faiss.IndexFlatIP(self.dimension)
        index = faiss.IndexIDMap(flat_index)
        logger.info("Created new IndexIDMap(IndexFlatIP) with dimension=%d", self.dimension)
        return index

    def _load_index(self) -> faiss.Index:
        """Load and return the FAISS index from disk.

        Returns:
            The loaded faiss.Index instance.

        Raises:
            Exception: Re-raises any file read error after logging it.
        """
        try:
            index = faiss.read_index(str(self.index_path))
            logger.info(
                "Index loaded from '%s' — %d vectors", self.index_path, index.ntotal
            )
            return index
        except Exception as exc:
            logger.error("Failed to load index from '%s': %s", self.index_path, exc)
            raise

    def save(self) -> None:
        """Persist the current FAISS index to disk.

        Raises:
            Exception: Re-raises any file write error after logging it.
        """
        try:
            faiss.write_index(self.index, str(self.index_path))
            logger.info(
                "Index saved to '%s' — %d vectors", self.index_path, self.index.ntotal
            )
        except Exception as exc:
            logger.error("Failed to save index to '%s': %s", self.index_path, exc)
            raise

    def load(self) -> None:
        """Reload the FAISS index from disk into self.index.

        Raises:
            Exception: Re-raises any file read error after logging it.
        """
        self.index = self._load_index()

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def add_embedding(self, file_id: int, embedding: np.ndarray) -> None:
        """Add a single embedding to the index.

        Args:
            file_id:   SQLite primary key of the file.
            embedding: 1-D NumPy array of shape (dimension,).

        Raises:
            ValueError: If embedding dimension does not match self.dimension.
        """
        embedding = np.array(embedding, dtype="float32")

        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)

        if embedding.shape[1] != self.dimension:
            raise ValueError(
                f"Embedding dimension {embedding.shape[1]} "
                f"does not match index dimension {self.dimension}."
            )

        ids = np.array([file_id], dtype=np.int64)
        self.index.add_with_ids(embedding, ids)
        logger.info("Added embedding for file_id=%d. Total vectors: %d", file_id, self.index.ntotal)

    def add_embeddings(self, file_ids: list[int], embeddings: np.ndarray) -> None:
        """Add a batch of embeddings to the index.

        Args:
            file_ids:   List of SQLite primary keys, one per embedding.
            embeddings: 2-D NumPy array of shape (n, dimension).

        Raises:
            ValueError: If file_ids is empty.
            ValueError: If number of IDs does not match number of embeddings.
            ValueError: If embedding dimension does not match self.dimension.
        """
        if not file_ids:
            raise ValueError("file_ids must not be empty.")

        embeddings = np.array(embeddings, dtype="float32")

        if len(file_ids) != embeddings.shape[0]:
            raise ValueError(
                f"Number of file_ids ({len(file_ids)}) does not match "
                f"number of embeddings ({embeddings.shape[0]})."
            )

        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Embedding dimension {embeddings.shape[1]} "
                f"does not match index dimension {self.dimension}."
            )

        ids = np.array(file_ids, dtype=np.int64)
        self.index.add_with_ids(embeddings, ids)
        logger.info(
            "Added %d embeddings. Total vectors: %d", len(file_ids), self.index.ntotal
        )

    def remove(self, file_id: int) -> None:
        """Remove a single vector from the index by file_id.

        Args:
            file_id: SQLite primary key of the file to remove.
        """
        id_selector = faiss.IDSelectorArray(np.array([file_id], dtype=np.int64))
        removed = self.index.remove_ids(id_selector)
        logger.info("Removed file_id=%d. Vectors removed: %d", file_id, removed)

    def rebuild(self, file_ids: list[int], embeddings: np.ndarray) -> None:
        """Replace the current index with a fresh one built from the given data.

        Useful after many removals to reclaim memory and compact the index.

        Args:
            file_ids:   List of SQLite primary keys.
            embeddings: 2-D NumPy array of shape (n, dimension).
        """
        logger.info("Rebuilding index with %d vectors.", len(file_ids))
        self.index = self._create_index()
        self.add_embeddings(file_ids, embeddings)
        self.save()
        logger.info("Index rebuild complete.")

    def reset(self) -> None:
        """Replace the current index with a new empty index.

        Does not delete the saved file on disk until save() is called.
        """
        self.index = self._create_index()
        logger.info("Index reset to empty.")

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self, query_embedding: np.ndarray, top_k: int = 10
    ) -> list[tuple[int, float]]:
        """Search the index for the most semantically similar vectors.

        Args:
            query_embedding: 1-D NumPy array of shape (dimension,).
                             Must be normalized (EmbeddingManager does this).
            top_k:           Maximum number of results to return.

        Returns:
            A list of (file_id, score) tuples ordered by descending similarity.
            Entries with file_id == -1 (FAISS padding) are excluded.

        Raises:
            ValueError: If query embedding dimension does not match index.
        """
        query = np.array(query_embedding, dtype="float32")

        if query.ndim == 1:
            query = query.reshape(1, -1)

        if query.shape[1] != self.dimension:
            raise ValueError(
                f"Query dimension {query.shape[1]} "
                f"does not match index dimension {self.dimension}."
            )

        effective_k = min(top_k, self.index.ntotal)
        if effective_k == 0:
            logger.warning("Search called on empty index.")
            return []

        scores, ids = self.index.search(query, effective_k)
        logger.info("Search complete. top_k=%d requested, %d returned.", top_k, effective_k)

        return [
            (int(file_id), float(score))
            for file_id, score in zip(ids[0], scores[0])
            if file_id != -1
        ]

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_total_vectors(self) -> int:
        """Return the total number of vectors currently stored in the index.

        Returns:
            Integer count of indexed vectors.
        """
        return self.index.ntotal

    def is_empty(self) -> bool:
        """Return True if the index contains no vectors.

        Returns:
            True if empty, False otherwise.
        """
        return self.index.ntotal == 0

    def __len__(self) -> int:
        """Support len(faiss_manager) to return total vector count.

        Returns:
            Total number of vectors in the index.
        """
        return self.index.ntotal


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    fm = FAISSManager()

    # Generate 3 random normalized vectors of shape (3, 384)
    rng = np.random.default_rng(seed=42)
    raw = rng.random((3, 384)).astype("float32")
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    vectors = raw / norms  # normalize to unit length

    # Add with file IDs matching SQLite primary keys
    fm.add_embeddings(file_ids=[1, 2, 3], embeddings=vectors)
    fm.save()

    # Search using the first vector — should return file_id=1 with score ≈ 1.0
    results = fm.search(query_embedding=vectors[0], top_k=3)
    print(f"\nSearch results (file_id, score):")
    for file_id, score in results:
        print(f"  file_id={file_id}  score={score:.6f}")

    print(f"\nTotal vectors in index: {len(fm)}")