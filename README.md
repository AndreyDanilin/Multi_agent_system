# Agentic Research Copilot

A modern multi-agent research assistant where RAG is one tool in a broader agent workflow.

The demo is built around the Hugging Face dataset [`galileo-ai/ragbench`](https://huggingface.co/datasets/galileo-ai/ragbench), subset `techqa`. The repository commits a small attributed fixture so tests and demos run without network access or API keys.

## What This Demonstrates

- Agent orchestration with explicit graph nodes: router, planner, tool executor, RAG tool, answer synthesizer, critic and finalizer.
- RAG as a typed tool call, not the whole application.
- Hybrid retrieval modes: `lexical`, `vector`, `hybrid`, `hybrid_rerank`.
- Cited answers with trace events and confidence.
- Deterministic evaluation: hit@k, MRR, citation coverage and latency.
- Modern-stable stack choices documented in `docs/adr/`.

## Architecture

```text
User question
  -> router
  -> planner
  -> tool_executor
  -> rag_tool: hybrid retrieval over RAGBench techqa
  -> answer_synthesizer
  -> critic: citation coverage check
  -> finalizer
  -> AnswerResponse(answer, citations, trace, metrics)
```

## Technology Decisions

- LangGraph is the target runtime for stateful agent workflows.
- Litestar is the API framework for typed OpenAPI-ready services.
- LanceDB is the local-first vector store target for hybrid retrieval and reranking.
- Streamlit is used as a focused demo console for chat, trace and evaluation views.
- A provider-neutral model adapter keeps hosted and local LLMs replaceable.

See the ADRs in `docs/adr/` for the rationale.

## Quick Start

```bash
uv run --extra test pytest
uv run research-copilot ingest-sample
uv run research-copilot query "DNS lookup fails after I connect to VPN. What should I check?"
uv run research-copilot evaluate
```

Run the API with the API extra:

```bash
uv run --extra api litestar --app research_copilot.api:app run
```

Run the Streamlit demo:

```bash
uv run --extra ui streamlit run src/research_copilot/ui/streamlit_app.py
```

Install the full showcase stack:

```bash
uv sync --extra all --extra dev
```

## Public Contracts

The main public types are:

- `AgentState`
- `AgentEvent`
- `ToolCall`
- `ToolResult`
- `RetrievedChunk`
- `Citation`
- `AnswerResponse`
- `EvaluationReport`

## Project Layout

```text
src/research_copilot/
  api.py                 Litestar-compatible API facade
  graph.py               Agent graph runtime
  service.py             Application service layer
  types.py               Public Pydantic contracts
  data/                  RAGBench techqa fixture
  models/                Provider-neutral model adapters
  retrieval/             Chunking, embeddings, repository, retrieval service
  tools/                 Typed tool registry
  ui/                    Streamlit demo
tests/                   Contract and regression tests
docs/adr/                Technology decision records
```

## Evaluation

The default evaluation is deterministic and reproducible:

- hit@k
- mean reciprocal rank
- citation coverage
- average retrieval latency

This keeps the quality story concrete. Optional LLM judge evaluation can be added later behind the provider-neutral model layer.

## License

See `LICENSE`.

