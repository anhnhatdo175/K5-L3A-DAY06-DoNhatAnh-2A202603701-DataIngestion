"""Local text embeddings using FastEmbed and ONNX Runtime."""
import math
import os
from collections.abc import Sequence
from typing import Any, Protocol

from fastembed import TextEmbedding


class Embedder(Protocol):
    model_name: str
    dimensions: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per input text, in the same order."""
        ...

    def embed_query(self, question: str) -> list[float]:
        """Return a query vector compatible with the stored document vectors."""
        ...


class FastEmbedder:
    """Generate local English embeddings with BAAI/bge-small-en-v1.5."""

    DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
    DEFAULT_DIMENSIONS = 384

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        model: Any | None = None,
        batch_size: int = 64,
    ) -> None:
        if model_name != self.DEFAULT_MODEL:
            raise ValueError(f"Unsupported embedding model: {model_name}")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.model_name = model_name
        self.dimensions = self.DEFAULT_DIMENSIONS
        self.batch_size = batch_size
        cache_dir = os.getenv("FASTEMBED_CACHE_PATH")
        self._model = (
            model
            if model is not None
            else TextEmbedding(model_name=model_name, cache_dir=cache_dir)
        )

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if any(not text.strip() for text in texts):
            raise ValueError("Embedding inputs must not be empty")

        vectors: list[list[float]] = []
        for offset in range(0, len(texts), self.batch_size):
            batch = list(texts[offset : offset + self.batch_size])
            encoded = self._model.embed(batch)
            batch_vectors = [[float(value) for value in vector] for vector in encoded]
            if len(batch_vectors) != len(batch):
                raise RuntimeError(
                    f"Embedding model returned {len(batch_vectors)} vectors "
                    f"for {len(batch)} inputs"
                )
            for vector in batch_vectors:
                if len(vector) != self.dimensions:
                    raise ValueError(
                        f"Expected {self.dimensions} dimensions, got {len(vector)}"
                    )
                if not all(math.isfinite(value) for value in vector):
                    raise ValueError("Embedding model returned a non-finite vector value")
                vectors.append(vector)
        return vectors

    def embed_query(self, question: str) -> list[float]:
        if not question.strip():
            raise ValueError("Query must not be empty")
        return self.embed_documents([question])[0]
