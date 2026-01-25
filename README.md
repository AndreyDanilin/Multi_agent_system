# Multi-Agent System

Production-grade multi-agent orchestration system demonstrating advanced agent architecture, structured communication, and scalable design patterns.

## Architecture Overview

This system implements a multi-agent architecture with clear separation of concerns, typed interfaces, and async-first design. The architecture follows production best practices:

- **Modular Design**: Each component (agents, tools, memory) is independently testable and extensible
- **Type Safety**: Full Pydantic v2 typing throughout the codebase
- **Async-First**: All I/O operations are asynchronous for scalability
- **Observability**: Structured logging with traceable agent decisions
- **Pluggable Components**: Memory and tools can be swapped without code changes

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                    ManagerAgent (Orchestrator)              │
│  - Controls workflow                                         │
│  - Routes messages                                           │
│  - Manages state transitions                                 │
└──────────────┬──────────────────────────────────────────────┘
               │
       ┌───────┴───────┬───────────────┬──────────────┐
       │               │               │              │
┌──────▼──────┐ ┌──────▼──────┐ ┌─────▼─────┐ ┌─────▼─────┐
│ PlannerAgent│ │ExecutorAgent│ │CriticAgent│ │  Memory   │
│             │ │             │ │           │ │           │
│ - Analyzes  │ │ - Executes  │ │ - Reviews │ │ - Short-  │
│   queries   │ │   steps     │ │   results │ │   term    │
│ - Creates   │ │ - Calls     │ │ - Detects │ │ - Long-   │
│   plans     │ │   tools     │ │   errors  │ │   term    │
└──────┬──────┘ └──────┬──────┘ └─────┬─────┘ └─────┬─────┘
       │               │               │              │
       └───────────────┴───────────────┴──────────────┘
                       │
              ┌────────┴────────┐
              │      Tools       │
              │ - Calculator     │
              │ - TextAnalyzer   │
              └──────────────────┘
```

## Agent Responsibilities

### PlannerAgent
Analyzes user input and decomposes complex tasks into structured execution plans. Each plan consists of sequential steps with associated tools and parameters.

**Key Features:**
- LLM-powered plan generation (with heuristic fallback)
- Tool-aware planning
- Conversation context integration

### ExecutorAgent
Executes individual steps from the plan by invoking appropriate tools. Handles tool execution, error management, and result formatting.

**Key Features:**
- Structured tool calling via schemas
- Error handling and recovery
- Intermediate result reporting

### CriticAgent
Reviews execution results to ensure correctness and completeness. Detects logical errors, missing steps, and hallucinations.

**Key Features:**
- Result validation
- Completeness checking
- Refinement recommendations

### ManagerAgent (Orchestrator)
Controls the overall workflow, routing messages between agents and managing state transitions. Implements the orchestration loop and termination conditions.

**Key Features:**
- State management
- Message routing
- Iteration control
- Retry logic

## Design Decisions

### Why This Architecture?

1. **Separation of Concerns**: Each agent has a single, well-defined responsibility. This makes the system easier to test, debug, and extend.

2. **Structured Communication**: All agent communication uses typed Pydantic models (`AgentMessage`), ensuring type safety and enabling validation.

3. **Pluggable Memory**: The memory system uses an abstract base class, allowing easy swapping between implementations (in-memory, vector stores, databases).

4. **Tool Abstraction**: Tools implement a common interface (`ToolBase`) with JSON schema definitions, enabling dynamic tool discovery and LLM function calling.

5. **Explicit State Management**: State is managed explicitly via `AgentState`, not hidden in prompts. This enables:
   - State inspection and debugging
   - State persistence
   - State rollback and recovery

6. **Async-First Design**: All operations are async, enabling:
   - Concurrent tool execution (future enhancement)
   - Non-blocking I/O
   - Scalability for production workloads

7. **Observability**: Structured logging with `structlog` provides:
   - Traceable agent decisions
   - Debugging-friendly output
   - Production-ready monitoring hooks

### Production Readiness

This system demonstrates production-grade patterns:

- **Error Handling**: Comprehensive error handling at each layer
- **Type Safety**: Full Pydantic v2 typing prevents runtime errors
- **Logging**: Structured logging for observability
- **Modularity**: Components can be tested and deployed independently
- **Extensibility**: Easy to add new agents, tools, or memory backends
- **No Hardcoded Prompts**: Prompts are separated into dedicated modules
- **Fallback Mechanisms**: System works without LLM (heuristic fallbacks)

## Project Structure

```
Multi_agent_system/
├── src/
│   ├── agents/          # Agent implementations
│   │   ├── base.py      # Base agent class
│   │   ├── planner.py   # PlannerAgent
│   │   ├── executor.py  # ExecutorAgent
│   │   ├── critic.py    # CriticAgent
│   │   └── manager.py   # ManagerAgent (orchestrator)
│   ├── tools/           # Tool implementations
│   │   ├── base.py      # Base tool interface
│   │   ├── calculator.py
│   │   └── text_analyzer.py
│   ├── memory/          # Memory management
│   │   ├── base.py      # Abstract memory interface
│   │   ├── short_term.py # Conversation context
│   │   └── long_term.py  # Vector store interface (pluggable)
│   ├── schemas/         # Pydantic models
│   │   ├── messages.py  # AgentMessage, MessageRole, etc.
│   │   └── state.py     # AgentState, Plan, ExecutionStep
│   ├── prompts/         # Prompt templates
│   │   ├── planner.py
│   │   ├── executor.py
│   │   └── critic.py
│   └── utils/           # Utilities
│       └── logger.py     # Structured logging
├── main.py              # Entry point
├── requirements.txt     # Dependencies
└── README.md
```

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables (optional, for LLM features)
export OPENAI_API_KEY=your_api_key_here
# Or create .env file:
# OPENAI_API_KEY=your_api_key_here
```

## Usage

```bash
python main.py
```

The system will run example queries demonstrating the full agent interaction flow. All agent decisions and state transitions are logged for observability.

### Example Output

```
✓ LLM initialized (OpenAI)
✓ Initialized 2 tools: ['calculator', 'text_analyzer']
✓ All agents initialized

================================================================================
Example 1: Calculate 15 multiplied by 23, then add 100 to the result
================================================================================

📋 Final Result:
   Task completed. Result: Step 1: 345.0
Step 2: 445.0

📊 Execution Summary:
   Iterations: 3
   Steps completed: 2/2
   Messages exchanged: 6

📝 Execution Plan:
   ✓ Step 1: Perform multiplication calculation
      Result: {'tool': 'calculator', 'tool_args': {...}, 'result': {'result': 345.0, ...}}
   ✓ Step 2: Perform addition calculation
      Result: {'tool': 'calculator', 'tool_args': {...}, 'result': {'result': 445.0, ...}}
```

## Extending the System

### Adding a New Agent

1. Create a new agent class inheriting from `AgentBase`:
```python
from src.agents.base import AgentBase
from src.schemas.messages import MessageRole, MessageType

class MyAgent(AgentBase):
    name = "MyAgent"
    role = MessageRole.MANAGER  # Choose appropriate role
    
    async def process(self, state, **kwargs):
        # Implement agent logic
        return self.create_message(...)
```

2. Register it in `ManagerAgent` if needed.

### Adding a New Tool

1. Create a tool class inheriting from `ToolBase`:
```python
from src.tools.base import ToolBase

class MyTool(ToolBase):
    name = "my_tool"
    description = "Tool description"
    
    async def execute(self, **kwargs):
        # Implement tool logic
        return {"result": ...}
    
    def _get_parameters_schema(self):
        return {...}  # JSON schema
```

2. Register it in `main.py` and pass to agents.

### Adding Long-Term Memory

The `LongTermMemory` class is a placeholder for vector store integration. To implement:

1. Choose a vector store (Pinecone, Weaviate, Chroma)
2. Implement embedding generation
3. Override `LongTermMemory` methods:
   - `add()`: Generate embeddings and store in vector DB
   - `search()`: Perform semantic similarity search

### Customizing Prompts

Edit prompt templates in `src/prompts/` without modifying business logic.

## Testing

The modular architecture enables easy unit testing:

```python
# Test individual agents
async def test_planner():
    planner = PlannerAgent(tools=[...])
    state = AgentState(user_query="...")
    result = await planner.process(state)
    assert result.message_type == MessageType.PLAN

# Test tools
async def test_calculator():
    tool = CalculatorTool()
    result = await tool.execute(operation="add", a=5, b=3)
    assert result["result"] == 8
```

## Performance Considerations

- **Async Operations**: All I/O is async for non-blocking execution
- **Memory Management**: Short-term memory is in-memory (suitable for single sessions)
- **LLM Calls**: Optional LLM usage with fallback heuristics
- **State Size**: State is kept in memory; for large-scale deployments, consider state persistence

## Future Enhancements

- **Vector Store Integration**: Implement semantic search in `LongTermMemory`
- **Parallel Execution**: Execute independent steps concurrently
- **State Persistence**: Save/restore state for long-running tasks
- **Agent Specialization**: Domain-specific agent variants
- **Streaming Responses**: Real-time result streaming
- **Retry Logic**: Enhanced retry mechanisms with exponential backoff
- **Metrics**: Prometheus/StatsD integration for production monitoring

## License

See LICENSE file.

## Author

Designed as a portfolio project demonstrating production-grade multi-agent system architecture.

