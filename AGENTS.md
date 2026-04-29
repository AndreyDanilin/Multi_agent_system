# Agent Instructions

This file is for coding agents and AI assistants working in this repository.

## Operating Rules

- Work on the active feature branch or worktree, not on `main`, unless the user explicitly asks.
- Preserve the public response contracts in `src/research_copilot/types.py`.
- Use tests first for behavior changes. Keep graph, retrieval and API tests focused on their own layer.
- Do not introduce generic model-provider abstractions. Tool and trace contracts should follow OpenAI Responses-style items.
- Keep root docs in sync when changing public contracts: `README.md`, `PROJECT.md`, `AGENTS.md`, `TOOLS.md`.

## Agent Graph Order

The graph order is defined in `src/research_copilot/agents.py` and consumed by `ResearchCopilotGraph`:

```text
router -> planner -> tool_executor -> rag_tool -> answer_synthesizer -> critic -> finalizer
```

Do not reorder these nodes without updating:

- `AGENT_SEQUENCE`
- `AGENT_SPECS`
- graph tests
- any trace-order expectations in API/UI code

## Node Responsibilities

- `router`: deterministically routes a question into the research workflow.
- `planner`: creates a short plan and emits a `FunctionCall` for `rag_search`.
- `tool_executor`: exposes registered tools and their schemas in trace metadata.
- `rag_tool`: executes the latest function call, stores retrieved chunks and emits `FunctionCallOutput`.
- `answer_synthesizer`: creates a cited answer from retrieved chunks only.
- `critic`: marks answers without citations as `needs_evidence`.
- `finalizer`: normalizes the final response for API, CLI and UI consumers.

## Testing Expectations

- Use `uv run --extra dev pytest` for full verification after dependency sync.
- Use `uv run --no-sync --extra dev pytest` only when network or dependency sync is blocked; report skips clearly.
- Retrieval unit tests should use `InMemoryRetrievalRepository` unless the test specifically covers LanceDB.
- LanceDB tests may use fake Sentence Transformer objects to avoid downloading model weights during unit tests.
