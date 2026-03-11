"""
Embedding service using fastembed (ONNX-based, lightweight).

Generates vector embeddings for text using the BAAI/bge-small-en-v1.5 model
which produces 384-dimensional vectors. Uses ONNX runtime instead of PyTorch
for smaller footprint and faster inference.

Supports hardware acceleration via ONNX execution providers:
- Apple Silicon: CoreMLExecutionProvider (install onnxruntime-silicon or default onnxruntime on arm64 macOS)
- NVIDIA GPU: CUDAExecutionProvider (install onnxruntime-gpu)
- CPU fallback: CPUExecutionProvider (always available)

Set ONNX_PROVIDER=cpu to force CPU-only mode. Otherwise, the best available provider is auto-detected.
"""

import asyncio
import logging
import os
from typing import Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

# Type alias matching fastembed's OnnxProvider
OnnxProvider = str | tuple[str, dict]


def _detect_providers() -> list[OnnxProvider]:
    """Detect the best available ONNX execution providers at runtime.

    Returns providers in priority order with CPUExecutionProvider as fallback.
    Respects ONNX_PROVIDER env var to force a specific mode:
        cpu   - CPU only
        coreml - CoreML (Apple Silicon)
        cuda  - CUDA (NVIDIA GPU)
    """
    import onnxruntime as ort

    available = set(ort.get_available_providers())
    forced = os.environ.get("ONNX_PROVIDER", "").lower().strip()

    if forced == "cpu":
        logger.info("ONNX provider forced to CPU via ONNX_PROVIDER env var")
        return ["CPUExecutionProvider"]
    elif forced == "coreml":
        if "CoreMLExecutionProvider" in available:
            logger.info("ONNX provider forced to CoreML via ONNX_PROVIDER env var")
            return ["CoreMLExecutionProvider", "CPUExecutionProvider"]
        logger.warning("CoreML requested but not available, falling back to CPU")
        return ["CPUExecutionProvider"]
    elif forced == "cuda":
        if "CUDAExecutionProvider" in available:
            logger.info("ONNX provider forced to CUDA via ONNX_PROVIDER env var")
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        logger.warning("CUDA requested but not available, falling back to CPU")
        return ["CPUExecutionProvider"]

    # Auto-detect: prefer GPU/accelerator over CPU
    providers: list[OnnxProvider] = []
    if "CUDAExecutionProvider" in available:
        providers.append("CUDAExecutionProvider")
    if "CoreMLExecutionProvider" in available:
        providers.append("CoreMLExecutionProvider")
    providers.append("CPUExecutionProvider")

    logger.info("Auto-detected ONNX providers: %s", providers)
    return providers


# Default batch sizes tuned per provider type
_BATCH_SIZES = {
    "CUDAExecutionProvider": 128,
    "CoreMLExecutionProvider": 64,
    "CPUExecutionProvider": 8,
}


class Embedder:
    """
    Text embedding service using fastembed (ONNX-based).

    Uses BAAI/bge-small-en-v1.5 by default which has:
    - 512 token max context window
    - 384 dimensional output vectors
    - Fast inference via ONNX runtime
    - ~50x smaller than PyTorch-based sentence-transformers

    Automatically selects the best available hardware accelerator.
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
        self._providers: Optional[list[OnnxProvider]] = None

    def _load_model(self):
        """Lazy load the model on first use with the best available provider."""
        if self._model is None:
            from fastembed import TextEmbedding

            self._providers = _detect_providers()
            self._model = TextEmbedding(
                model_name=self.model_name,
                providers=self._providers,
            )

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

    @property
    def default_batch_size(self) -> int:
        """Return a batch size tuned for the active provider."""
        if self._providers:
            for p in self._providers:
                name = p if isinstance(p, str) else p[0]
                if name in _BATCH_SIZES and name != "CPUExecutionProvider":
                    return _BATCH_SIZES[name]
        return _BATCH_SIZES["CPUExecutionProvider"]

    @property
    def active_providers(self) -> list[str]:
        """Return the list of active provider names (after model load)."""
        self._load_model()
        return [p if isinstance(p, str) else p[0] for p in (self._providers or [])]

    async def embed_texts(self, texts: list[str], batch_size: int | None = None) -> np.ndarray:
        """
        Embed multiple texts.

        Args:
            texts: List of texts to embed
            batch_size: Batch size for encoding (auto-tuned per provider if None)

        Returns:
            Numpy array of shape (n_texts, dimension)
        """
        self._load_model()

        if batch_size is None:
            batch_size = self.default_batch_size

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
