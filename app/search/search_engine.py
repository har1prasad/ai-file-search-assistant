"""
search_engine.py — Semantic search engine for AI File Search Assistant.

Ties together all three core components of the pipeline:
    - EmbeddingManager  : converts a natural language query into a vector
    - FAISSManager      : finds the most similar file vectors in the index
    - DatabaseManager   : fetches full file metadata for each matched file_id

The SearchEngine is the single entry point the UI layer interacts with.
Everything below it is an implementation detail.

Pipeline for a single search:
    user query (str)
        → EmbeddingManager.generate_embedding()   → query vector (384,)
        → FAISSManager.search()                    → [(file_id, score), ...]
        → DatabaseManager.get_file_by_id()         → [{ metadata + score }, ...]
        → returned to caller

Example usage:
    from app.search.search_engine import SearchEngine

    engine = SearchEngine()
    results = engine.search("Python classes and inheritance", top_k=5)
    for r in results:
        print(r["filename"], r["similarity_score"])
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import logging
from typing import Any

from app.database.db_manager import DatabaseManager
from app.embeddings.embedding_manager import EmbeddingManager
from app.search.faiss_manager import FAISSManager

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SearchEngine
# ---------------------------------------------------------------------------


class SearchEngine:
    """Orchestrates semantic search across indexed files.

    Accepts a natural language query, converts it to an embedding,
    finds the closest matches in the FAISS index, and enriches each
    result with file metadata from SQLite.

    Args:
        embedding_manager: An initialised EmbeddingManager instance.
                           If None, a default instance is created.
        faiss_manager:     An initialised FAISSManager instance.
                           If None, a default instance is created.
        db_manager:        An initialised DatabaseManager instance.
                           If None, a default instance is created.
    """

    def __init__(
        self,
        embedding_manager: EmbeddingManager | None = None,
        faiss_manager: FAISSManager | None = None,
        db_manager: DatabaseManager | None = None,
    ) -> None:
        self.embedding_manager = embedding_manager or EmbeddingManager()
        self.faiss_manager = faiss_manager or FAISSManager()
        self.db_manager = db_manager or DatabaseManager()
        logger.info("SearchEngine initialised.")

    # ------------------------------------------------------------------
    # Core search
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        """Search indexed files using a natural language query.

        Steps:
            1. Validate query and top_k.
            2. Generate a query embedding.
            3. Search the FAISS index for the closest vectors.
            4. Fetch metadata from SQLite for each matched file_id.
            5. Attach similarity_score to each metadata dict.
            6. Return the ranked result list.

        Args:
            query:  Natural language search query.
            top_k:  Maximum number of results to return. Must be > 0.

        Returns:
            A list of metadata dicts, each containing all SQLite columns
            plus a "similarity_score" key (float, 0.0 – 1.0).
            Results are ordered by descending similarity score.

        Raises:
            ValueError: If query is empty or whitespace-only.
            ValueError: If top_k is not a positive integer.
        """
        # --- Validation ---
        if not query or not query.strip():
            raise ValueError("Search query must not be empty.")
        if top_k <= 0:
            raise ValueError(f"top_k must be a positive integer, got {top_k}.")

        logger.info("Search started. query='%s'  top_k=%d", query, top_k)

        # --- Embed the query ---
        query_embedding = self.embedding_manager.generate_embedding(query)

        # --- Search FAISS ---
        vector_matches = self.faiss_manager.search(query_embedding, top_k=top_k)
        logger.info("Vector matches returned: %d", len(vector_matches))

        # --- Enrich with metadata ---
        results: list[dict[str, Any]] = []
        for file_id, score in vector_matches:
            metadata = self.db_manager.get_file_by_id(file_id)
            if metadata:
                metadata["similarity_score"] = score
                results.append(metadata)

        logger.info("Final results with metadata: %d", len(results))
        return results

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    def search_paths(self, query: str, top_k: int = 10) -> list[str]:
        """Return only the file paths from search results.

        Args:
            query:  Natural language search query.
            top_k:  Maximum number of results.

        Returns:
            A list of file path strings ordered by descending similarity.
        """
        results = self.search(query, top_k=top_k)
        return [r["path"] for r in results if "path" in r]

    def search_with_scores(
        self, query: str, top_k: int = 10
    ) -> list[tuple[str, float]]:
        """Return (file_path, similarity_score) tuples from search results.

        Args:
            query:  Natural language search query.
            top_k:  Maximum number of results.

        Returns:
            A list of (path, score) tuples ordered by descending similarity.
        """
        results = self.search(query, top_k=top_k)
        return [
            (r["path"], r["similarity_score"])
            for r in results
            if "path" in r
        ]


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Ensure project root is on the path when run directly
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

    logging.basicConfig(level=logging.INFO)

    engine = SearchEngine()
    query = input("Search query: ").strip()

    if not query:
        print("No query entered. Exiting.")
        sys.exit(0)

    results = engine.search(query, top_k=5)

    if not results:
        print("No results found.")
    else:
        print("\nResults:")
        for i, result in enumerate(results, start=1):
            print(
                f"{i}. {result.get('filename', 'Unknown')} "
                f"(score={result['similarity_score']:.4f})"
            )
            print(f"   {result.get('path', 'N/A')}")