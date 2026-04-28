"""Document chunking."""

from __future__ import annotations

from research_copilot.types import RetrievedChunk, SourceDocument


def chunk_documents(
    documents: list[SourceDocument],
    *,
    max_words: int = 120,
    overlap_words: int = 24,
) -> list[RetrievedChunk]:
    """Split documents into overlapping chunks with stable provenance."""

    if max_words <= 0:
        raise ValueError("max_words must be positive")
    if overlap_words < 0:
        raise ValueError("overlap_words cannot be negative")
    if overlap_words >= max_words:
        raise ValueError("overlap_words must be smaller than max_words")

    chunks: list[RetrievedChunk] = []
    for document in documents:
        words = document.text.split()
        if not words:
            continue

        start = 0
        chunk_index = 0
        step = max_words - overlap_words
        while start < len(words):
            end = min(start + max_words, len(words))
            text = " ".join(words[start:end])
            chunks.append(
                RetrievedChunk(
                    chunk_id=f"{document.document_id}#{chunk_index}",
                    document_id=document.document_id,
                    title=document.title,
                    text=text,
                    source=document.source,
                    metadata={
                        "chunk_index": chunk_index,
                        "word_start": start,
                        "word_end": end,
                        "answer": document.answer,
                        **document.metadata,
                    },
                )
            )
            if end == len(words):
                break
            start += step
            chunk_index += 1

    return chunks
