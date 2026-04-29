from research_copilot.types import (
    AgentEvent,
    AgentState,
    AnswerResponse,
    Citation,
    FunctionCall,
    FunctionCallOutput,
    FunctionTool,
    RetrievedChunk,
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


def test_agent_state_tracks_openai_function_calls_and_outputs():
    state = AgentState(question="How do I restore DNS resolution?")
    call = FunctionCall(name="rag_search", arguments='{"query":"How do I restore DNS resolution?"}')
    output = FunctionCallOutput(call_id=call.call_id, output='{"chunks":2}')

    state.function_calls.append(call)
    state.function_call_outputs.append(output)

    assert state.function_calls[0].model_dump()["type"] == "function_call"
    assert state.function_calls[0].name == "rag_search"
    assert state.function_call_outputs[0].model_dump()["type"] == "function_call_output"
    assert state.function_call_outputs[0].call_id == call.call_id


def test_function_tool_schema_uses_strict_openai_shape():
    tool = FunctionTool(
        name="rag_search",
        description="Search indexed technical documentation.",
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
            "additionalProperties": False,
        },
    )

    assert tool.model_dump() == {
        "type": "function",
        "name": "rag_search",
        "description": "Search indexed technical documentation.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
            "additionalProperties": False,
        },
        "strict": True,
    }
