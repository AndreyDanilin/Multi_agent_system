"""Agent graph runtime for the Research Copilot."""

from __future__ import annotations

from typing import Any

from research_copilot.models import DeterministicLLM, LLMAdapter
from research_copilot.retrieval.service import RetrievalService
from research_copilot.tools import ToolRegistry
from research_copilot.types import (
    AgentEvent,
    AgentState,
    AnswerResponse,
    Citation,
    RetrievalMode,
    ToolCall,
    ToolResult,
)


class ResearchCopilotGraph:
    """A compact LangGraph-compatible orchestration facade.

    The deterministic local runner keeps the portfolio demo reproducible, while
    node boundaries map directly to a LangGraph StateGraph in environments where
    the dependency is installed.
    """

    def __init__(
        self,
        retrieval_service: RetrievalService,
        *,
        llm: LLMAdapter | None = None,
    ) -> None:
        self.retrieval_service = retrieval_service
        self.llm = llm or DeterministicLLM()
        self.tools = ToolRegistry()
        self.tools.register("rag_search", self._rag_search_tool)
        self.langgraph_available = self._detect_langgraph()
        self._compiled_graph = self._compile_langgraph()

    def query(
        self,
        question: str,
        *,
        retrieval_mode: RetrievalMode = "hybrid_rerank",
        limit: int = 5,
    ) -> AnswerResponse:
        state = AgentState(
            question=question,
            retrieval_mode=retrieval_mode,
            metadata={"limit": limit, "langgraph_available": self.langgraph_available},
        )
        if self._compiled_graph is not None:
            try:
                state = AgentState.model_validate(self._compiled_graph.invoke(state.model_dump()))
            except Exception as exc:
                state.metadata["langgraph_fallback_reason"] = str(exc)
                state = self._run_local(state)
        else:
            state = self._run_local(state)
        return AnswerResponse(
            answer=state.answer or "",
            citations=state.citations,
            trace=state.events,
            confidence=state.confidence,
            retrieval_mode=state.retrieval_mode,
            metadata={
                "assessment": state.assessment,
                "tool_calls": [call.model_dump(mode="json") for call in state.tool_calls],
                "tool_results": [result.model_dump(mode="json") for result in state.tool_results],
                **state.metadata,
            },
        )

    def _run_local(self, state: AgentState) -> AgentState:
        for node in self._node_sequence():
            state = node(state)
        return state

    def _node_sequence(self):
        return (
            self._router,
            self._planner,
            self._tool_executor,
            self._rag_tool,
            self._answer_synthesizer,
            self._critic,
            self._finalizer,
        )

    def _router(self, state: AgentState) -> AgentState:
        state.route = "research"
        state.events.append(
            AgentEvent(node="router", message="Routed question to research workflow")
        )
        return state

    def _planner(self, state: AgentState) -> AgentState:
        state.plan = [
            "Retrieve relevant technical context",
            "Synthesize a grounded answer",
            "Verify that citations support the answer",
        ]
        state.tool_calls.append(
            ToolCall(
                tool_name="rag_search",
                arguments={
                    "query": state.question,
                    "mode": state.retrieval_mode,
                    "limit": state.metadata.get("limit", 5),
                },
            )
        )
        state.events.append(AgentEvent(node="planner", message="Created tool-using research plan"))
        return state

    def _tool_executor(self, state: AgentState) -> AgentState:
        state.events.append(
            AgentEvent(
                node="tool_executor",
                message="Prepared tool execution",
                metadata={"tools": self.tools.names},
            )
        )
        return state

    def _rag_tool(self, state: AgentState) -> AgentState:
        call = state.tool_calls[-1]
        output = self.tools.call(call.tool_name, **call.arguments)
        state.retrieved_chunks = output["chunks"]
        state.tool_results.append(ToolResult(tool_name=call.tool_name, status="ok", output=output))
        state.events.append(
            AgentEvent(
                node="rag_tool",
                message="Retrieved context",
                metadata={
                    "chunks": len(state.retrieved_chunks),
                    "retrieval_mode": state.retrieval_mode,
                    "latency_ms": round(self.retrieval_service.last_latency_ms, 3),
                },
            )
        )
        return state

    def _answer_synthesizer(self, state: AgentState) -> AgentState:
        contexts = [chunk.text for chunk in state.retrieved_chunks[:3]]
        state.answer = self.llm.answer(state.question, contexts)
        state.citations = [self._citation_from_chunk(chunk) for chunk in state.retrieved_chunks[:3]]
        if state.citations:
            state.confidence = round(
                sum(citation.score for citation in state.citations) / len(state.citations),
                3,
            )
        state.events.append(
            AgentEvent(
                node="answer_synthesizer",
                message="Synthesized grounded answer",
                metadata={"citations": len(state.citations)},
            )
        )
        return state

    def _critic(self, state: AgentState) -> AgentState:
        state.assessment = "complete" if state.citations else "needs_evidence"
        state.events.append(
            AgentEvent(
                node="critic",
                message="Checked citation coverage",
                metadata={"assessment": state.assessment},
            )
        )
        return state

    def _finalizer(self, state: AgentState) -> AgentState:
        if state.assessment == "needs_evidence":
            state.answer = (
                "I could not find enough cited evidence in the indexed corpus to answer "
                "confidently. Ingest more documents or try a broader retrieval query."
            )
            state.confidence = 0.0
        state.events.append(AgentEvent(node="finalizer", message="Finalized response"))
        return state

    def _rag_search_tool(
        self,
        *,
        query: str,
        mode: RetrievalMode,
        limit: int = 5,
    ) -> dict[str, Any]:
        chunks = self.retrieval_service.search(query, mode=mode, limit=limit)
        return {
            "chunks": chunks,
            "latency_ms": self.retrieval_service.last_latency_ms,
        }

    @staticmethod
    def _citation_from_chunk(chunk: Any) -> Citation:
        quote = chunk.text[:180].strip()
        if len(chunk.text) > 180:
            quote += "..."
        score = chunk.rerank_score if chunk.rerank_score is not None else chunk.hybrid_score
        return Citation(
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            title=chunk.title,
            quote=quote,
            source=chunk.source,
            score=score,
        )

    @staticmethod
    def _detect_langgraph() -> bool:
        try:
            import langgraph  # noqa: F401
        except Exception:
            return False
        return True

    def _compile_langgraph(self):
        try:
            from langgraph.graph import END, StateGraph
        except Exception:
            return None

        workflow = StateGraph(dict)
        nodes = {
            "router": self._router,
            "planner": self._planner,
            "tool_executor": self._tool_executor,
            "rag_tool": self._rag_tool,
            "answer_synthesizer": self._answer_synthesizer,
            "critic": self._critic,
            "finalizer": self._finalizer,
        }

        def wrap(node):
            def invoke(payload):
                state = (
                    payload
                    if isinstance(payload, AgentState)
                    else AgentState.model_validate(payload)
                )
                return node(state).model_dump()

            return invoke

        for name, node in nodes.items():
            workflow.add_node(name, wrap(node))
        workflow.set_entry_point("router")
        workflow.add_edge("router", "planner")
        workflow.add_edge("planner", "tool_executor")
        workflow.add_edge("tool_executor", "rag_tool")
        workflow.add_edge("rag_tool", "answer_synthesizer")
        workflow.add_edge("answer_synthesizer", "critic")
        workflow.add_edge("critic", "finalizer")
        workflow.add_edge("finalizer", END)
        return workflow.compile()
