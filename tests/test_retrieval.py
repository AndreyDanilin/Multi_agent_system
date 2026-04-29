import pytest

from research_copilot.data.fixtures import load_sample_techqa_documents
from research_copilot.retrieval.chunking import chunk_documents
from research_copilot.retrieval.embeddings import (
    BGE_SMALL_EN_V15,
    DeterministicEmbeddingModel,
    SentenceTransformerEmbeddingModel,
)
from research_copilot.retrieval.repository import InMemoryRetrievalRepository, LanceDBRepository
from research_copilot.retrieval.service import RetrievalService


def test_chunking_preserves_dataset_attribution_and_source_ids():
    docs = load_sample_techqa_documents()
    chunks = chunk_documents(docs, max_words=40, overlap_words=8)

    assert chunks
    assert chunks[0].document_id == docs[0].document_id
    assert chunks[0].source == "galileo-ai/ragbench/techqa"
    assert chunks[0].chunk_id.startswith(f"{docs[0].document_id}#")


def test_hybrid_rerank_finds_relevant_dns_chunk():
    docs = load_sample_techqa_documents()
    repository = InMemoryRetrievalRepository(
        embedding_model=DeterministicEmbeddingModel(dimensions=64),
    )
    service = RetrievalService(repository=repository)
    service.ingest_documents(docs)

    results = service.search(
        "DNS lookup fails after VPN connection",
        mode="hybrid_rerank",
        limit=3,
    )

    assert results
    assert results[0].document_id == "techqa-dns-vpn"
    assert results[0].hybrid_score > 0
    assert results[0].metadata["retrieval_mode"] == "hybrid_rerank"


def test_retrieval_modes_are_available():
    repository = InMemoryRetrievalRepository(
        embedding_model=DeterministicEmbeddingModel(dimensions=32),
    )
    service = RetrievalService(repository=repository)
    service.ingest_documents(load_sample_techqa_documents())

    modes = {"lexical", "vector", "hybrid", "hybrid_rerank"}
    observed = {mode for mode in modes if service.search("certificate renewal", mode=mode, limit=2)}

    assert observed == modes


def test_sentence_transformer_embedding_model_uses_bge_query_instruction_without_loading_model():
    captured = {}

    class FakeSentenceTransformer:
        def __init__(self, model_name: str):
            captured["model_name"] = model_name

        def encode(self, texts, normalize_embeddings):
            captured["texts"] = texts
            captured["normalize_embeddings"] = normalize_embeddings
            return [[0.1, 0.2, 0.3]]

    model = SentenceTransformerEmbeddingModel(model_factory=FakeSentenceTransformer)

    vector = model.embed_query("DNS fails after VPN")

    assert model.model_name == BGE_SMALL_EN_V15
    assert vector == [0.1, 0.2, 0.3]
    assert captured["texts"] == [
        "Represent this sentence for searching relevant passages: DNS fails after VPN"
    ]
    assert captured["normalize_embeddings"] is True


def test_lancedb_repository_persists_chunks_in_real_table(tmp_path):
    pytest.importorskip("lancedb")

    class FakeSentenceTransformer:
        def __init__(self, model_name: str):
            self.model_name = model_name

        def encode(self, texts, normalize_embeddings):
            return [[0.5, 0.25, 0.125] for _ in texts]

    docs = load_sample_techqa_documents()
    repository = LanceDBRepository(
        path=tmp_path / "lancedb",
        embedding_model=SentenceTransformerEmbeddingModel(model_factory=FakeSentenceTransformer),
    )
    service = RetrievalService(repository=repository)
    service.ingest_documents(docs)

    assert repository.table_name == "chunks"
    assert repository.count() > 0
    assert (tmp_path / "lancedb").exists()

    results = service.search("DNS VPN lookup", mode="hybrid", limit=2)

    assert results
    assert results[0].metadata["retrieval_backend"] == "lancedb"


def test_in_memory_repository_keeps_lightweight_unit_test_backend():
    repository = InMemoryRetrievalRepository(
        embedding_model=DeterministicEmbeddingModel(dimensions=16)
    )
    service = RetrievalService(repository=repository)
    service.ingest_documents(load_sample_techqa_documents())

    results = service.search("certificate renewal", mode="hybrid_rerank", limit=2)

    assert results
    assert results[0].metadata["retrieval_backend"] == "memory"
