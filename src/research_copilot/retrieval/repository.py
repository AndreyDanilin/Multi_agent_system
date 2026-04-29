"""Retrieval repositories for local tests and LanceDB-backed runtime."""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from typing import Any

from research_copilot.types import RetrievalMode, RetrievedChunk

from .embeddings import EmbeddingModel, SentenceTransformerEmbeddingModel, cosine_similarity
from .tokenization import tokenize


class HybridScoringMixin:
    """Shared lexical/vector/hybrid scoring used by repository backends."""

    embedding_model: EmbeddingModel
    backend_name: str

    def _score_records(
        self,
        records: list[dict[str, Any]],
        query: str,
        *,
        mode: RetrievalMode,
        limit: int,
    ) -> list[RetrievedChunk]:
        query_tokens = tokenize(query)
        query_embedding = self.embedding_model.embed_query(query)
        lexical_values: list[float] = []
        vector_values: list[float] = []
        raw_scores: list[tuple[dict[str, Any], float, float]] = []

        for record in records:
            lexical = self._lexical_score(query_tokens, record["tokens"])
            vector = max(0.0, cosine_similarity(query_embedding, record["embedding"]))
            lexical_values.append(lexical)
            vector_values.append(vector)
            raw_scores.append((record, lexical, vector))

        max_lexical = max(lexical_values, default=0.0) or 1.0
        max_vector = max(vector_values, default=0.0) or 1.0
        scored: list[RetrievedChunk] = []

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
            chunk.metadata["retrieval_backend"] = self.backend_name
            scored.append(chunk)

        if mode == "hybrid_rerank":
            scored = self._rerank(query_tokens, scored)

        scored.sort(
            key=lambda chunk: (
                chunk.rerank_score if chunk.rerank_score is not None else chunk.hybrid_score
            ),
            reverse=True,
        )
        return scored[:limit]

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


class InMemoryRetrievalRepository(HybridScoringMixin):
    """Lightweight deterministic repository for unit tests and no-storage checks."""

    backend_name = "memory"

    def __init__(self, embedding_model: EmbeddingModel) -> None:
        self.embedding_model = embedding_model
        self._records: list[dict[str, Any]] = []

    def upsert_chunks(self, chunks: list[RetrievedChunk]) -> None:
        self._records = [self._record_from_chunk(chunk) for chunk in chunks]

    def count(self) -> int:
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
        _validate_mode(mode)
        started = perf_counter()
        chunks = self._score_records(self._records, query, mode=mode, limit=limit)
        return chunks, (perf_counter() - started) * 1000

    def _record_from_chunk(self, chunk: RetrievedChunk) -> dict[str, Any]:
        text_for_embedding = f"{chunk.title}\n{chunk.text}"
        return {
            "chunk": chunk.model_dump(mode="json"),
            "tokens": tokenize(text_for_embedding),
            "embedding": self.embedding_model.embed(text_for_embedding),
        }


class LanceDBRepository(HybridScoringMixin):
    """LanceDB-backed repository for runtime retrieval over stored chunk vectors."""

    backend_name = "lancedb"
    table_name = "chunks"

    def __init__(
        self,
        path: str | Path,
        embedding_model: EmbeddingModel | None = None,
    ) -> None:
        self.path = Path(path)
        self.embedding_model = embedding_model or SentenceTransformerEmbeddingModel()
        self._records: list[dict[str, Any]] = []
        self._loaded = False
        self._db: Any | None = None

    def upsert_chunks(self, chunks: list[RetrievedChunk]) -> None:
        self.path.mkdir(parents=True, exist_ok=True)
        self._records = [self._record_from_chunk(chunk) for chunk in chunks]
        rows = [
            {
                "chunk_id": record["chunk"]["chunk_id"],
                "document_id": record["chunk"]["document_id"],
                "title": record["chunk"]["title"],
                "text": record["chunk"]["text"],
                "source": record["chunk"]["source"],
                "chunk_json": json.dumps(record["chunk"], separators=(",", ":")),
                "tokens_json": json.dumps(record["tokens"], separators=(",", ":")),
                "vector": record["embedding"],
            }
            for record in self._records
        ]
        db = self._connect()
        if rows:
            db.create_table(self.table_name, data=rows, mode="overwrite")
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
        _validate_mode(mode)
        self._ensure_loaded()
        started = perf_counter()
        chunks = self._score_records(self._records, query, mode=mode, limit=limit)
        return chunks, (perf_counter() - started) * 1000

    def _record_from_chunk(self, chunk: RetrievedChunk) -> dict[str, Any]:
        text_for_embedding = f"{chunk.title}\n{chunk.text}"
        return {
            "chunk": chunk.model_dump(mode="json"),
            "tokens": tokenize(text_for_embedding),
            "embedding": self.embedding_model.embed(text_for_embedding),
        }

    def _connect(self) -> Any:
        if self._db is None:
            try:
                import lancedb
            except ImportError as exc:
                raise RuntimeError(
                    "LanceDBRepository requires the default 'lancedb' dependency to be installed."
                ) from exc
            self._db = lancedb.connect(str(self.path))
        return self._db

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        db = self._connect()
        try:
            table = db.open_table(self.table_name)
        except Exception:
            self._records = []
            self._loaded = True
            return
        rows = table.to_arrow().to_pylist()
        self._records = [
            {
                "chunk": json.loads(row["chunk_json"]),
                "tokens": json.loads(row["tokens_json"]),
                "embedding": [float(value) for value in row["vector"]],
            }
            for row in rows
        ]
        self._loaded = True


def _validate_mode(mode: RetrievalMode) -> None:
    if mode not in {"lexical", "vector", "hybrid", "hybrid_rerank"}:
        raise ValueError(f"Unsupported retrieval mode: {mode}")
