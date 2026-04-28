from research_copilot.types import (
    AgentEvent,
    AgentState,
    AnswerResponse,
    Citation,
    RetrievedChunk,
    ToolCall,
    ToolResult,
)


def test_answer_response_exposes_citations_and_trace():
    chunk = RetrievedChunk(
        chunk_id="techqa-1#0",
        document_id="techqa-1",
        title="Network troubleshooting",
        text="Restarting the network adapter can restore DNS resolution.",
        source="galileo-ai/ragbench/techqa",
        lexical_score=0.7,
        vector_score=0.4,
        hybrid_score=0.62,
    )
    citation = Citation(
        chunk_id=chunk.chunk_id,
        document_id=chunk.document_id,
        title=chunk.title,
        quote="restore DNS resolution",
        source=chunk.source,
        score=chunk.hybrid_score,
    )
    event = AgentEvent(node="rag_tool", message="Retrieved context", metadata={"k": 1})
    response = AnswerResponse(
        answer="Restart the network adapter and retry DNS resolution.",
        citations=[citation],
        trace=[event],
        confidence=0.62,
        retrieval_mode="hybrid_rerank",
    )

    assert response.citations[0].chunk_id == "techqa-1#0"
    assert response.trace[0].node == "rag_tool"
    assert response.retrieval_mode == "hybrid_rerank"


def test_agent_state_tracks_tool_calls_and_results():
    state = AgentState(question="How do I restore DNS resolution?")
    call = ToolCall(tool_name="rag_search", arguments={"query": state.question})
    result = ToolResult(tool_name="rag_search", status="ok", output={"chunks": 2})

    state.tool_calls.append(call)
    state.tool_results.append(result)

    assert state.tool_calls[0].tool_name == "rag_search"
    assert state.tool_results[0].output["chunks"] == 2
