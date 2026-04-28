from research_copilot.data.fixtures import load_sample_techqa_documents
from research_copilot.retrieval.chunking import chunk_documents
from research_copilot.retrieval.embeddings import DeterministicEmbeddingModel
from research_copilot.retrieval.repository import LanceDBRepository
from research_copilot.retrieval.service import RetrievalService


def test_chunking_preserves_dataset_attribution_and_source_ids():
    docs = load_sample_techqa_documents()
    chunks = chunk_documents(docs, max_words=40, overlap_words=8)

    assert chunks
    assert chunks[0].document_id == docs[0].document_id
    assert chunks[0].source == "galileo-ai/ragbench/techqa"
    assert chunks[0].chunk_id.startswith(f"{docs[0].document_id}#")


def test_hybrid_rerank_finds_relevant_dns_chunk(tmp_path):
    docs = load_sample_techqa_documents()
    repository = LanceDBRepository(
        path=tmp_path / "lancedb",
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


def test_retrieval_modes_are_available(tmp_path):
    repository = LanceDBRepository(
        path=tmp_path / "lancedb",
        embedding_model=DeterministicEmbeddingModel(dimensions=32),
    )
    service = RetrievalService(repository=repository)
    service.ingest_documents(load_sample_techqa_documents())

    modes = {"lexical", "vector", "hybrid", "hybrid_rerank"}
    observed = {mode for mode in modes if service.search("certificate renewal", mode=mode, limit=2)}

    assert observed == modes
