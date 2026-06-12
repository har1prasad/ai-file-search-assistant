"""
embedding_manager.py — Semantic embedding module for AI File Search Assistant.

Converts extracted file text into fixed-size numerical vectors (embeddings)
using the Sentence Transformers library. These embeddings capture the semantic
meaning of text, enabling similarity-based search rather than keyword matching.

How embeddings are used in this project:
    1. extractor.py          → extracts raw text from files
    2. db_manager.py         → stores metadata and content in SQLite
    3. embedding_manager.py  → converts text into embedding vectors
    4. faiss_manager.py      → stores and searches embeddings using FAISS

Model:
    sentence-transformers/all-MiniLM-L6-v2
    - Embedding dimension : 384
    - Downloaded once
    - Stored locally in ./models/
    - Runs fully offline after first download
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# EmbeddingManager
# ---------------------------------------------------------------------------

class EmbeddingManager:
    """Generates semantic embeddings using a Sentence Transformer model."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    ) -> None:
        self.model_name = model_name
        self.local_model_path = Path("models/all-MiniLM-L6-v2")

        logger.info("Initializing embedding model...")

        try:
            if self.local_model_path.exists():
                logger.info("Loading model from local folder...")
                self.model = SentenceTransformer(str(self.local_model_path))
                logger.info("Local model loaded successfully.")
            else:
                logger.info("Local model not found.")
                logger.info("Downloading model from Hugging Face...")

                self.model = SentenceTransformer(model_name)

                self.local_model_path.parent.mkdir(parents=True, exist_ok=True)
                self.model.save(str(self.local_model_path))

                logger.info(
                    "Model downloaded and saved to: %s",
                    self.local_model_path
                )

        except Exception as exc:
            logger.error("Failed to load model '%s': %s", model_name, exc)
            raise

    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a single text."""

        if not text or not text.strip():
            text = "[EMPTY]"

        try:
            embedding = self.model.encode(
                text,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            return embedding

        except Exception as exc:
            logger.error("generate_embedding failed: %s", exc)
            raise

    def generate_embeddings(self, texts: list[str]) -> np.ndarray:
        """Generate embeddings for multiple texts."""

        cleaned = [t if t and t.strip() else "[EMPTY]" for t in texts]

        try:
            embeddings = self.model.encode(
                cleaned,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            return embeddings

        except Exception as exc:
            logger.error("generate_embeddings failed: %s", exc)
            raise

    def get_dimension(self) -> int:
        """Return embedding dimension."""
        sample = self.model.encode(
            "dimension check",
            convert_to_numpy=True
        )
        return len(sample)


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    em = EmbeddingManager()

    vector = em.generate_embedding(
        "Python notes about classes and inheritance."
    )
    print(f"Embedding shape     : {vector.shape}")
    print(f"Embedding dimension : {em.get_dimension()}")
    print(f"First 5 values      : {vector[:5]}")

    batch = em.generate_embeddings(
        [
            "Python programming notes",
            "Database normalization concepts",
            "Machine learning introduction",
        ]
    )
    print(f"Batch shape         : {batch.shape}")