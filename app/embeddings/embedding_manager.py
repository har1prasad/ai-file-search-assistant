"""
embedding_manager.py — Optimized semantic embedding manager.

Improvements:
- Lazy model loading (faster app startup)
- Shared singleton model across all instances
- Thread-safe initialization
- GPU acceleration support (CUDA auto-detection)
- No unnecessary inference for dimension lookup
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """Generate semantic embeddings using Sentence Transformers."""

    _shared_model = None
    _model_lock = threading.Lock()

    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    MODEL_PATH = Path("models/all-MiniLM-L6-v2")
    DIMENSION = 384

    def __init__(self) -> None:
        logger.info("EmbeddingManager initialized (lazy mode).")

    def _ensure_model_loaded(self) -> SentenceTransformer:
        """Load model only once globally, using GPU if available."""
        if EmbeddingManager._shared_model is not None:
            return EmbeddingManager._shared_model

        with EmbeddingManager._model_lock:
            if EmbeddingManager._shared_model is not None:
                return EmbeddingManager._shared_model

            logger.info("Loading embedding model...")

            try:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                logger.info("Embedding model target device: %s", device)

                if self.MODEL_PATH.exists():
                    logger.info("Loading local model from disk...")
                    model = SentenceTransformer(str(self.MODEL_PATH), device=device)
                else:
                    logger.info("Downloading model...")
                    model = SentenceTransformer(self.MODEL_NAME, device=device)

                    self.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
                    model.save(str(self.MODEL_PATH))

                EmbeddingManager._shared_model = model
                logger.info("Embedding model ready on %s.", device)
                return model

            except Exception as exc:
                logger.exception("Model loading failed: %s", exc)
                raise

    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for one text."""
        if not text or not text.strip():
            text = "[EMPTY]"

        model = self._ensure_model_loaded()

        embedding = model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        return embedding.astype("float32")

    def generate_embeddings(self, texts: list[str]) -> np.ndarray:
        """Generate embeddings for multiple texts."""
        cleaned = [t if t and t.strip() else "[EMPTY]" for t in texts]
        model = self._ensure_model_loaded()

        embeddings = model.encode(
            cleaned,
            convert_to_numpy=True,
            normalize_embeddings=True,
            batch_size=32,
            show_progress_bar=False,
        )

        return embeddings.astype("float32")

    def get_dimension(self) -> int:
        """Return embedding dimension."""
        return self.DIMENSION