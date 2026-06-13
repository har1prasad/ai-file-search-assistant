"""
search_engine.py — Optimized hybrid semantic & keyword search engine.
"""

from __future__ import annotations

import logging
from typing import Any

from app.database.db_manager import DatabaseManager
from app.embeddings.embedding_manager import EmbeddingManager
from app.search.faiss_manager import FAISSManager

logger = logging.getLogger(__name__)


class SearchEngine:
    """Hybrid search over indexed files (Semantic Vector + FTS5 Keywords)."""

    def __init__(
        self,
        embedding_manager: EmbeddingManager,
        faiss_manager: FAISSManager,
        db_manager: DatabaseManager,
    ) -> None:
        self.embedding_manager = embedding_manager
        self.faiss_manager = faiss_manager
        self.db_manager = db_manager
        logger.info("SearchEngine initialized.")

    def search(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        """Search indexed files using a hybrid model (FAISS + SQLite FTS5).

        Matches concepts semantically and exact keywords, blending scores.
        """
        query = query.strip()

        if not query:
            raise ValueError("Search query cannot be empty.")

        if top_k <= 0:
            raise ValueError("top_k must be > 0")

        # 1. Semantic retrieval (FAISS)
        vector_matches = []
        if not self.faiss_manager.is_empty():
            query_embedding = self.embedding_manager.generate_embedding(query)
            vector_matches = self.faiss_manager.search(query_embedding, top_k * 3)

        # 2. Exact keyword retrieval (SQLite FTS5)
        fts_matches = self.db_manager.search_fts(query, limit=top_k * 3)

        # 3. Score blending (Hybrid RRF / Weighted combination)
        combined_scores: dict[int, dict[str, float]] = {}

        # Parse vector scores (cosine similarity range is typically 0.0 to 1.0)
        for chunk_id, sem_score in vector_matches:
            clamped_sem = max(0.0, min(1.0, float(sem_score)))
            combined_scores[chunk_id] = {
                "semantic": clamped_sem,
                "keyword": 0.0
            }

        # Parse FTS5 BM25 scores (range: lower is better, usually negative)
        if fts_matches:
            bm25_scores = [score for _, score in fts_matches]
            min_score = min(bm25_scores)
            max_score = max(bm25_scores)
            score_range = max_score - min_score

            for chunk_id, bm25_score in fts_matches:
                # Normalize BM25 score from 0.0 (worst) to 1.0 (best)
                norm_keyword = 1.0
                if score_range > 0.0001:
                    norm_keyword = 1.0 - ((bm25_score - min_score) / score_range)

                if chunk_id in combined_scores:
                    combined_scores[chunk_id]["keyword"] = norm_keyword
                else:
                    combined_scores[chunk_id] = {
                        "semantic": 0.0,
                        "keyword": norm_keyword
                    }

        # 4. Resolve chunks and aggregate by parent file ID
        results_map = {}
        for chunk_id, scores in combined_scores.items():
            sem = scores["semantic"]
            kw = scores["keyword"]

            # Blending logic
            if sem > 0.0 and kw > 0.0:
                hybrid_score = 0.7 * sem + 0.3 * kw
            elif sem > 0.0:
                hybrid_score = sem
            else:
                hybrid_score = 0.5 * kw  # Keyword-only match slightly discounted

            chunk = self.db_manager.get_chunk_by_id(chunk_id)
            if chunk:
                file_id = chunk["file_id"]
                if file_id not in results_map:
                    metadata = self.db_manager.get_file_by_id(file_id)
                    if metadata:
                        metadata["similarity_score"] = hybrid_score
                        metadata["matching_chunks"] = [chunk["content"]]
                        results_map[file_id] = metadata
                else:
                    # Keep the highest match score, accumulate matching chunks
                    if hybrid_score > results_map[file_id]["similarity_score"]:
                        results_map[file_id]["similarity_score"] = hybrid_score
                    results_map[file_id]["matching_chunks"].append(chunk["content"])

        # Sort the aggregated unique file results by combined similarity score descending
        results = sorted(results_map.values(), key=lambda x: x["similarity_score"], reverse=True)[:top_k]

        logger.info("Hybrid search yielded %d unique file matches.", len(results))
        return results

    def search_paths(self, query: str, top_k: int = 10) -> list[str]:
        results = self.search(query, top_k)
        return [r["path"] for r in results if "path" in r]

    def search_with_scores(
        self,
        query: str,
        top_k: int = 10
    ) -> list[tuple[str, float]]:
        results = self.search(query, top_k)
        return [
            (r["path"], r["similarity_score"])
            for r in results
            if "path" in r
        ]