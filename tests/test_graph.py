from research_copilot.data.fixtures import load_sample_techqa_documents
from research_copilot.graph import ResearchCopilotGraph
from research_copilot.retrieval.embeddings import DeterministicEmbeddingModel
from research_copilot.retrieval.repository import LanceDBRepository
from research_copilot.retrieval.service import RetrievalService


def test_graph_invokes_rag_as_a_tool_and_returns_cited_answer(tmp_path):
    repository = LanceDBRepository(
        path=tmp_path / "lancedb",
        embedding_model=DeterministicEmbeddingModel(dimensions=64),
    )
    retrieval = RetrievalService(repository=repository)
    retrieval.ingest_documents(load_sample_techqa_documents())
    graph = ResearchCopilotGraph(retrieval_service=retrieval)

    response = graph.query(
        "Why does DNS fail after connecting to a VPN?",
        retrieval_mode="hybrid_rerank",
    )

    assert response.citations
    assert "DNS" in response.answer
    assert [event.node for event in response.trace] == [
        "router",
        "planner",
        "tool_executor",
        "rag_tool",
        "answer_synthesizer",
        "critic",
        "finalizer",
    ]


def test_graph_marks_answer_as_needing_evidence_when_retrieval_is_empty(tmp_path):
    repository = LanceDBRepository(
        path=tmp_path / "empty-lancedb",
        embedding_model=DeterministicEmbeddingModel(dimensions=64),
    )
    graph = ResearchCopilotGraph(retrieval_service=RetrievalService(repository=repository))

    response = graph.query("What is the incident response policy?", retrieval_mode="hybrid")

    assert response.citations == []
    assert response.metadata["assessment"] == "needs_evidence"
