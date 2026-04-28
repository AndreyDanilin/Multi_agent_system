"""Local embedding adapters.

The deterministic model is intentionally simple and dependency-free. It gives
tests and offline demos stable vector behavior while the production dependency
set can swap in Sentence Transformers or another embedding provider.
"""

from __future__ import annotations

import hashlib
import math
from typing import Protocol

from .tokenization import tokenize


class EmbeddingModel(Protocol):
    dimensions: int

    def embed(self, text: str) -> list[float]:
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


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    return sum(a * b for a, b in zip(left, right, strict=True))
