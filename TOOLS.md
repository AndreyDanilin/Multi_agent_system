# Tool Instructions

This file defines tool contracts for agents working in this repository.

## Contract Style

Tools use OpenAI Responses-style function declarations:

```json
{
  "type": "function",
  "name": "tool_name",
  "description": "What the tool does.",
  "parameters": {
    "type": "object",
    "properties": {},
    "required": [],
    "additionalProperties": false
  },
  "strict": true
}
```

Every tool schema must:

- use `type: "function"`;
- set `strict: true`;
- include `additionalProperties: false`;
- keep all runtime arguments JSON-serializable;
- return output that can be serialized into a `FunctionCallOutput.output` string.

## `rag_search`

`rag_search` is the only registered runtime tool today. It searches indexed technical documentation and returns cited chunks.

Schema:

```json
{
  "type": "function",
  "name": "rag_search",
  "description": "Search indexed technical documentation and return cited chunks.",
  "parameters": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "Natural-language retrieval query."
      },
      "mode": {
        "type": "string",
        "enum": ["lexical", "vector", "hybrid", "hybrid_rerank"],
        "description": "Retrieval strategy to use."
      },
      "limit": {
        "type": "integer",
        "minimum": 1,
        "maximum": 20,
        "description": "Maximum number of chunks to return."
      }
    },
    "required": ["query", "mode", "limit"],
    "additionalProperties": false
  },
  "strict": true
}
```

The graph stores the request as a `FunctionCall`:

```json
{
  "type": "function_call",
  "call_id": "...",
  "name": "rag_search",
  "arguments": "{\"query\":\"DNS VPN lookup\",\"mode\":\"hybrid_rerank\",\"limit\":5}"
}
```

The graph stores the result as a `FunctionCallOutput` with the same `call_id`.

## Adding Tools

When adding a tool:

1. Register the handler and schema in `ToolRegistry`.
2. Add or update tests for schema shape and graph metadata.
3. Update this file with the schema and output shape.
4. Keep tool execution deterministic in tests.
