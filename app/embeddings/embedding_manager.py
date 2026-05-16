"""
embedding_manager.py — Semantic embedding module for AI File Search Assistant.

Converts extracted file text into fixed-size numerical vectors (embeddings)
using the Sentence Transformers library. These embeddings capture the semantic
meaning of text, enabling similarity-based search rather than keyword matching.

How embeddings are used in this project:
    1. extractor.py     → extracts raw text from files
    2. db_manager.py    → stores metadata and content in SQLite
    3. embedding_manager.py → converts text into embedding vectors  ← this module
    4. faiss_manager.py → stores and searches embeddings using FAISS

Why normalization is enabled:
    Setting normalize_embeddings=True scales every vector to unit length.
    This allows FAISS to use dot-product operations as cosine similarity,
    which measures semantic relatedness regardless of text length.

Model: all-MiniLM-L6-v2
    - Embedding dimension : 384
    - Runs fully offline after first download
    - Fast and accurate for general-purpose semantic search

Example usage:
    from app.embeddings import EmbeddingManager

    em = EmbeddingManager()
    vector = em.generate_embedding("Python classes and inheritance")
    print(vector.shape)  # (384,)

    batch = em.generate_embeddings(["Python notes", "Database concepts"])
    print(batch.shape)   # (2, 384)
"""



from __future__ import annotations

# Force the Hugging Face libraries to work fully offline.
# Prevents network calls to huggingface.co on every model load.
# The model must already be downloaded and cached locally before this takes effect.
import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

import logging

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
    """Generates semantic embeddings using a Sentence Transformer model.

    Loads the model once at construction time and exposes methods for
    single-text and batch embedding generation.

    Args:
        model_name: Hugging Face model identifier.
                    Defaults to "all-MiniLM-L6-v2".
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        logger.info("Loading embedding model: %s", model_name)
        try:
            self.model = SentenceTransformer(model_name)
            logger.info("Model loaded successfully: %s", model_name)
        except Exception as exc:
            logger.error("Failed to load model '%s': %s", model_name, exc)
            raise

    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate a single semantic embedding vector for the given text.

        Empty or whitespace-only strings are replaced with "[EMPTY]" to
        ensure the model always receives valid input.

        Args:
            text: Input string to embed.

        Returns:
            A NumPy array of shape (384,) representing the embedding.

        Raises:
            Exception: Re-raises any model inference error after logging it.
        """
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
        """Generate semantic embeddings for a batch of texts.

        Empty or whitespace-only strings in the list are replaced with
        "[EMPTY]" before encoding.

        Args:
            texts: List of input strings to embed.

        Returns:
            A NumPy array of shape (n, 384) where n = len(texts).

        Raises:
            Exception: Re-raises any model inference error after logging it.
        """
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
        """Return the embedding dimension produced by the loaded model.

        Determined dynamically by encoding a sample string, so it stays
        accurate even if the model is swapped out.

        Returns:
            Integer dimension of the embedding vectors (384 for all-MiniLM-L6-v2).
        """
        sample = self.model.encode("dimension check", convert_to_numpy=True)
        return len(sample)


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    em = EmbeddingManager()

    # Single embedding
    vector = em.generate_embedding("Python notes about classes and inheritance.")
    print(f"Embedding shape     : {vector.shape}")
    print(f"Embedding dimension : {em.get_dimension()}")
    print(f"First 5 values      : {vector[:5]}")

    # Batch embeddings
    batch = em.generate_embeddings(
        [
            "Python programming notes",
            "Database normalization concepts",
            "Machine learning introduction",
        ]
    )
    print(f"Batch shape         : {batch.shape}")