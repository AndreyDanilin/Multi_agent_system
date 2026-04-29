"""Local embedding adapters.

The deterministic model is intentionally simple and dependency-free. It gives
tests and offline demos stable vector behavior while the production dependency
set can swap in Sentence Transformers or another embedding provider.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable
from typing import Any, Protocol

from .tokenization import tokenize

BGE_SMALL_EN_V15 = "BAAI/bge-small-en-v1.5"
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages:"


class EmbeddingModel(Protocol):
    dimensions: int

    def embed(self, text: str) -> list[float]:
        ...

    def embed_query(self, text: str) -> list[float]:
        ...


class DeterministicEmbeddingModel:
    """Hashing-vector embedding model for local fallback and tests."""

    def __init__(self, dimensions: int = 128) -> None:
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")
        self.dimensions = dimensions

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        tokens = tokenize(text)
        if not tokens:
            return vector

        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]

    def embed_query(self, text: str) -> list[float]:
        return self.embed(text)


class SentenceTransformerEmbeddingModel:
    """Sentence Transformers embedding backend using BGE small v1.5 by default."""

    def __init__(
        self,
        model_name: str = BGE_SMALL_EN_V15,
        *,
        dimensions: int = 384,
        model_factory: Callable[[str], Any] | None = None,
    ) -> None:
        self.model_name = model_name
        self.dimensions = dimensions
        self._model_factory = model_factory
        self._model: Any | None = None

    @property
    def model(self) -> Any:
        if self._model is None:
            if self._model_factory is None:
                from sentence_transformers import SentenceTransformer

                self._model_factory = SentenceTransformer
            self._model = self._model_factory(self.model_name)
        return self._model

    def embed(self, text: str) -> list[float]:
        return self._encode(text)

    def embed_query(self, text: str) -> list[float]:
        return self._encode(f"{BGE_QUERY_INSTRUCTION} {text}")

    def _encode(self, text: str) -> list[float]:
        encoded = self.model.encode([text], normalize_embeddings=True)
        vector = encoded[0]
        if hasattr(vector, "tolist"):
            return vector.tolist()
        return [float(value) for value in vector]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    return sum(a * b for a, b in zip(left, right, strict=True))
