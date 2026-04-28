"""High-level retrieval service used by tools, graph and API."""

from __future__ import annotations

from research_copilot.types import RetrievalMode, RetrievedChunk, SourceDocument

from .chunking import chunk_documents
from .repository import LanceDBRepository


class RetrievalService:
    def __init__(
        self,
        repository: LanceDBRepository,
        *,
        chunk_size_words: int = 120,
        chunk_overlap_words: int = 24,
    ) -> None:
        self.repository = repository
        self.chunk_size_words = chunk_size_words
        self.chunk_overlap_words = chunk_overlap_words
        self.last_latency_ms = 0.0

    def ingest_documents(self, documents: list[SourceDocument]) -> int:
        chunks = chunk_documents(
            documents,
            max_words=self.chunk_size_words,
            overlap_words=min(self.chunk_overlap_words, self.chunk_size_words - 1),
        )
        self.repository.upsert_chunks(chunks)
        return len(chunks)

    def search(
        self,
        query: str,
        *,
        mode: RetrievalMode = "hybrid_rerank",
        limit: int = 5,
    ) -> list[RetrievedChunk]:
        chunks, latency = self.repository.search(query, mode=mode, limit=limit)
        self.last_latency_ms = latency
        return chunks
