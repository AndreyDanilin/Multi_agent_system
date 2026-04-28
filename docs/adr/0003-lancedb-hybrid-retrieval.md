# ADR 0003: Use LanceDB for Hybrid Retrieval

## Status
Accepted

## Context
The project demonstrates RAG as one tool in a broader agent system. The retrieval layer therefore needs to be strong enough to evaluate, compare and explain retrieval quality, while staying easy to run locally.

## Decision
Use LanceDB as the target local-first vector database and retrieval backend. The system supports four retrieval modes: lexical, vector, hybrid and hybrid rerank. The committed implementation includes a dependency-light LanceDB-compatible repository for tests and offline demos, with the adapter boundary ready for native LanceDB tables and rerankers.

## Consequences
- The demo can compare retrieval quality across modes.
- The project avoids Docker as a default requirement.
- The storage boundary keeps production LanceDB integration isolated from graph/API code.
