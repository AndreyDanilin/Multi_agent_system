# ADR 0001: Use LangGraph for Agent Runtime

## Status
Accepted

## Context
The previous project used a hand-written loop to coordinate planner, executor and critic agents. That loop was useful as a learning artifact, but it hid workflow transitions in imperative control flow and made retries, tracing and future checkpointing harder to reason about.

## Decision
Use LangGraph as the target runtime for stateful agent orchestration. The v1 implementation exposes deterministic node boundaries that map to a LangGraph workflow: router, planner, tool executor, RAG tool, answer synthesizer, critic and finalizer.

## Consequences
- Agent behavior is modeled as explicit state transitions rather than scattered conditionals.
- Trace events can be surfaced in API and Streamlit UI.
- The local deterministic runner keeps tests fast and offline-friendly.
- A full LangGraph compiled graph can replace the local runner without changing public API contracts.
