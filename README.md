# Agentic Research Copilot

A multi-agent research assistant where RAG is one OpenAI-style function tool in a broader agent workflow.

The demo is built around the Hugging Face dataset `galileo-ai/ragbench`, subset `techqa`. The repository commits a small attributed fixture so tests can run without external API keys.

## What This Demonstrates

- Explicit agent graph nodes: router, planner, tool executor, RAG tool, answer synthesizer, critic and finalizer.
- OpenAI Responses-style function call contracts for tool use.
- Real local retrieval runtime with LanceDB and `BAAI/bge-small-en-v1.5` embeddings.
- Hybrid retrieval modes: `lexical`, `vector`, `hybrid`, `hybrid_rerank`.
- Cited answers with trace events, confidence and deterministic evaluation metrics.

## Quick Start

```bash
uv sync --extra dev
uv run --extra dev pytest
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

## Important Files

- `PROJECT.md` explains the architecture, runtime choices and public contracts.
- `AGENTS.md` is the operating manual for future coding agents working in this repository.
- `TOOLS.md` defines the OpenAI-style function tool contract and schema discipline.

## Public Contracts

The main public types are:

- `AgentState`
- `AgentEvent`
- `FunctionCall`
- `FunctionCallOutput`
- `FunctionTool`
- `RetrievedChunk`
- `Citation`
- `AnswerResponse`
- `EvaluationReport`

## Project Layout

```text
src/research_copilot/
  agents.py              Agent node specs and graph order
  api.py                 Litestar-compatible API facade
  graph.py               Agent graph runtime
  service.py             Application service layer
  types.py               Public Pydantic contracts
  data/                  RAGBench techqa fixture
  models/                OpenAI Responses-style model client boundary
  retrieval/             Chunking, embeddings, LanceDB repository and retrieval service
  tools/                 OpenAI function tool registry
  ui/                    Streamlit demo
tests/                   Contract and regression tests
```

## Evaluation

The default evaluation compares retrieval modes with hit@k, mean reciprocal rank, citation coverage and average retrieval latency. This keeps retrieval quality visible before adding optional model-judge evaluation.

## License

See `LICENSE`.
