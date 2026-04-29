# Project Overview

Agentic Research Copilot is a multi-agent research assistant that treats retrieval as one tool inside an explicit agent workflow. The project is designed as a portfolio-grade backend: typed contracts, traceable graph nodes, local vector search, reproducible evaluation and thin API/CLI/UI entrypoints.

## Architecture

```text
User question
  -> router
  -> planner
  -> tool_executor
  -> rag_tool
  -> answer_synthesizer
  -> critic
  -> finalizer
  -> AnswerResponse
```

The node order and node contracts live in `src/research_copilot/agents.py`. `ResearchCopilotGraph` can run with the deterministic local runner and can map the same nodes into LangGraph when the optional graph extra is installed.

## Public Contracts

Public contracts live in `src/research_copilot/types.py`.

Tool use follows OpenAI Responses-style items:

- `FunctionTool` declares strict function schemas.
- `FunctionCall` stores a requested function call with JSON-string arguments.
- `FunctionCallOutput` stores the tool output string with the same `call_id`.

`AnswerResponse` is the shared API, CLI and UI response shape. It includes the final answer, citations, trace events, confidence, retrieval mode and metadata.

## Retrieval Runtime

The runtime retrieval path uses:

- LanceDB for local vector table storage.
- `BAAI/bge-small-en-v1.5` through Sentence Transformers for embeddings.
- A BGE query instruction for user queries: `Represent this sentence for searching relevant passages: ...`.
- Plain passage text for document/chunk embeddings.

The retrieval service supports four modes:

- `lexical`: BM25-like token score.
- `vector`: cosine similarity over embeddings.
- `hybrid`: weighted lexical/vector score.
- `hybrid_rerank`: hybrid score plus query-token coverage and title overlap.

`InMemoryRetrievalRepository` remains available for lightweight unit tests. It should not be presented as the production retrieval backend.

## Application Surfaces

- CLI: `research-copilot ingest-sample`, `query`, and `evaluate`.
- API: Litestar routes for health, sample ingestion, query, evaluation runs and run lookup.
- UI: Streamlit demo for chat, trace and evaluation views.

The service layer owns ingestion, query execution, evaluation and in-memory run lookup.

## Evaluation

Evaluation is deterministic by default. It compares retrieval modes with:

- hit@k
- mean reciprocal rank
- citation coverage
- average retrieval latency

This keeps retrieval quality visible before adding optional LLM-as-judge evaluation.

## Development Notes

- Use `AGENTS.md` for agent/coding-assistant operating instructions.
- Use `TOOLS.md` for function tool schema rules.
- Keep heavy runtime dependencies in default dependencies because real local RAG is part of the core project.
- Keep API, UI and full dataset ingestion as extras.
