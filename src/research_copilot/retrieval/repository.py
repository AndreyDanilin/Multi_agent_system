"""LanceDB-compatible local repository.

The class keeps a dependency-light in-process implementation for tests and
offline demos. When LanceDB is installed, this boundary is where a production
table adapter can be added without changing graph or API contracts.
"""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from typing import Any

from research_copilot.types import RetrievalMode, RetrievedChunk

from .embeddings import DeterministicEmbeddingModel, EmbeddingModel, cosine_similarity
from .tokenization import tokenize


class LanceDBRepository:
    """Local-first retrieval repository with lexical, vector and hybrid modes."""

    def __init__(
        self,
        path: str | Path,
        embedding_model: EmbeddingModel | None = None,
    ) -> None:
        self.path = Path(path)
        self.embedding_model = embedding_model or DeterministicEmbeddingModel()
        self._records: list[dict[str, Any]] = []
        self._loaded = False

    @property
    def storage_file(self) -> Path:
        return self.path / "chunks.json"

    def upsert_chunks(self, chunks: list[RetrievedChunk]) -> None:
        self.path.mkdir(parents=True, exist_ok=True)
        self._records = []
        for chunk in chunks:
            text_for_embedding = f"{chunk.title}\n{chunk.text}"
            self._records.append(
                {
                    "chunk": chunk.model_dump(mode="json"),
                    "tokens": tokenize(text_for_embedding),
                    "embedding": self.embedding_model.embed(text_for_embedding),
                }
            )
        self.storage_file.write_text(json.dumps(self._records, indent=2), encoding="utf-8")
        self._loaded = True

    def count(self) -> int:
        self._ensure_loaded()
        return len(self._records)

    def search(
        self,
        query: str,
        *,
        mode: RetrievalMode = "hybrid_rerank",
        limit: int = 5,
    ) -> tuple[list[RetrievedChunk], float]:
        if limit <= 0:
            return [], 0.0
        if mode not in {"lexical", "vector", "hybrid", "hybrid_rerank"}:
            raise ValueError(f"Unsupported retrieval mode: {mode}")

        self._ensure_loaded()
        started = perf_counter()
        query_tokens = tokenize(query)
        query_embedding = self.embedding_model.embed(query)

        scored: list[RetrievedChunk] = []
        lexical_values: list[float] = []
        vector_values: list[float] = []

        raw_scores: list[tuple[dict[str, Any], float, float]] = []
        for record in self._records:
            lexical = self._lexical_score(query_tokens, record["tokens"])
            vector = max(0.0, cosine_similarity(query_embedding, record["embedding"]))
            lexical_values.append(lexical)
            vector_values.append(vector)
            raw_scores.append((record, lexical, vector))

        max_lexical = max(lexical_values, default=0.0) or 1.0
        max_vector = max(vector_values, default=0.0) or 1.0

        for record, lexical, vector in raw_scores:
            lexical_norm = lexical / max_lexical
            vector_norm = vector / max_vector
            if mode == "lexical":
                score = lexical_norm
            elif mode == "vector":
                score = vector_norm
            else:
                score = 0.55 * lexical_norm + 0.45 * vector_norm

            chunk = RetrievedChunk(**record["chunk"])
            chunk.lexical_score = round(lexical_norm, 6)
            chunk.vector_score = round(vector_norm, 6)
            chunk.hybrid_score = round(score, 6)
            chunk.metadata["retrieval_mode"] = mode
            scored.append(chunk)

        if mode == "hybrid_rerank":
            scored = self._rerank(query_tokens, scored)

        scored.sort(
            key=lambda chunk: (
                chunk.rerank_score if chunk.rerank_score is not None else chunk.hybrid_score
            ),
            reverse=True,
        )
        latency_ms = (perf_counter() - started) * 1000
        return scored[:limit], latency_ms

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        if not self.storage_file.exists():
            self._records = []
            self._loaded = True
            return
        self._records = json.loads(self.storage_file.read_text(encoding="utf-8"))
        self._loaded = True

    @staticmethod
    def _lexical_score(query_tokens: list[str], document_tokens: list[str]) -> float:
        if not query_tokens or not document_tokens:
            return 0.0
        doc_counts: dict[str, int] = {}
        for token in document_tokens:
            doc_counts[token] = doc_counts.get(token, 0) + 1
        score = 0.0
        doc_len = len(document_tokens)
        avg_len = 80.0
        k1 = 1.2
        b = 0.75
        for token in query_tokens:
            tf = doc_counts.get(token, 0)
            if tf == 0:
                continue
            score += ((k1 + 1) * tf) / (k1 * (1 - b + b * doc_len / avg_len) + tf)
        return score

    @staticmethod
    def _rerank(query_tokens: list[str], chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        query_set = set(query_tokens)
        for chunk in chunks:
            chunk_tokens = set(tokenize(f"{chunk.title} {chunk.text}"))
            coverage = len(query_set & chunk_tokens) / max(1, len(query_set))
            title_overlap = len(query_set & set(tokenize(chunk.title))) / max(1, len(query_set))
            rerank = 0.75 * chunk.hybrid_score + 0.2 * coverage + 0.05 * title_overlap
            chunk.rerank_score = round(min(1.0, rerank), 6)
        return chunks
