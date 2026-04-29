"""Agent node specifications for the research copilot graph."""

from __future__ import annotations

from pydantic import BaseModel, Field


class AgentSpec(BaseModel):
    """Instruction contract for a single graph node."""

    name: str
    role: str
    inputs: list[str] = Field(default_factory=list)
    outputs: list[str] = Field(default_factory=list)
    instructions: str


AGENT_SEQUENCE = (
    "router",
    "planner",
    "tool_executor",
    "rag_tool",
    "answer_synthesizer",
    "critic",
    "finalizer",
)


AGENT_SPECS: dict[str, AgentSpec] = {
    "router": AgentSpec(
        name="router",
        role="Classify the request and route it into the research workflow.",
        inputs=["question"],
        outputs=["route", "trace event"],
        instructions="Keep routing deterministic until additional workflows exist.",
    ),
    "planner": AgentSpec(
        name="planner",
        role="Create a short research plan and request retrieval as an OpenAI function call.",
        inputs=["question", "retrieval_mode", "limit"],
        outputs=["plan", "function_calls"],
        instructions="Emit rag_search calls with JSON-string arguments matching TOOLS.md.",
    ),
    "tool_executor": AgentSpec(
        name="tool_executor",
        role="Prepare registered tools for execution and expose available tool names.",
        inputs=["function_calls", "registered tools"],
        outputs=["trace event"],
        instructions="Do not execute retrieval here; execution belongs to the rag_tool node.",
    ),
    "rag_tool": AgentSpec(
        name="rag_tool",
        role="Execute the rag_search function call against the retrieval service.",
        inputs=["latest function_call", "retrieval service"],
        outputs=["retrieved_chunks", "function_call_outputs", "trace event"],
        instructions="Return structured JSON output and preserve the originating call_id.",
    ),
    "answer_synthesizer": AgentSpec(
        name="answer_synthesizer",
        role="Generate a cited answer from retrieved context.",
        inputs=["question", "retrieved_chunks"],
        outputs=["answer", "citations", "confidence", "trace event"],
        instructions="Use only retrieved chunks as evidence and keep citations attached.",
    ),
    "critic": AgentSpec(
        name="critic",
        role="Check whether the answer has enough cited evidence.",
        inputs=["answer", "citations"],
        outputs=["assessment", "trace event"],
        instructions="Mark answers without citations as needs_evidence.",
    ),
    "finalizer": AgentSpec(
        name="finalizer",
        role="Normalize the final response for API, CLI and UI consumers.",
        inputs=["assessment", "answer", "citations"],
        outputs=["final answer", "confidence", "trace event"],
        instructions="Return a no-evidence response when the critic reports needs_evidence.",
    ),
}
