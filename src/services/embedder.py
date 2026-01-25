"""
Embedding service using fastembed (ONNX-based, lightweight).

Generates vector embeddings for text using the BAAI/bge-small-en-v1.5 model
which produces 384-dimensional vectors. Uses ONNX runtime instead of PyTorch
for smaller footprint and faster inference.
"""

import asyncio
from typing import Optional

import numpy as np


class Embedder:
    """
    Text embedding service using fastembed (ONNX-based).

    Uses BAAI/bge-small-en-v1.5 by default which has:
    - 512 token max context window
    - 384 dimensional output vectors
    - Fast inference via ONNX runtime
    - ~50x smaller than PyTorch-based sentence-transformers
    """

    # Model name mapping for backward compatibility
    MODEL_MAP = {
        "all-MiniLM-L6-v2": "BAAI/bge-small-en-v1.5",  # Similar size/performance
    }

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        """
        Initialize the embedder.

        Args:
            model_name: Name of the fastembed model to use
        """
        # Map old model names to fastembed equivalents
        self.model_name = self.MODEL_MAP.get(model_name, model_name)
        self._model = None
        self._dimension: Optional[int] = 384  # bge-small-en-v1.5 dimension

    def _load_model(self):
        """Lazy load the model on first use."""
        if self._model is None:
            from fastembed import TextEmbedding
            self._model = TextEmbedding(model_name=self.model_name)

    @property
    def dimension(self) -> int:
        """Get the embedding dimension."""
        return self._dimension

    async def embed_text(self, text: str) -> np.ndarray:
        """
        Embed a single text string.

        Args:
            text: Text to embed

        Returns:
            Numpy array of shape (dimension,)
        """
        self._load_model()

        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(
            None,
            lambda: list(self._model.embed([text]))[0]
        )

        return np.array(embedding, dtype=np.float32)

    async def embed_texts(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """
        Embed multiple texts.

        Args:
            texts: List of texts to embed
            batch_size: Batch size for encoding

        Returns:
            Numpy array of shape (n_texts, dimension)
        """
        self._load_model()

        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None,
            lambda: list(self._model.embed(texts, batch_size=batch_size))
        )

        return np.array(embeddings, dtype=np.float32)

    def get_max_tokens(self) -> int:
        """Get the maximum token limit for the model."""
        # bge-small-en-v1.5 has 512 token max
        return 512
