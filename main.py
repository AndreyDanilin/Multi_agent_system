"""Main entry point for the multi-agent system."""

import asyncio
import os
from typing import List

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from src.agents.critic import CriticAgent
from src.agents.executor import ExecutorAgent
from src.agents.manager import ManagerAgent
from src.agents.planner import PlannerAgent
from src.schemas.state import AgentState
from src.tools.calculator import CalculatorTool
from src.tools.text_analyzer import TextAnalyzerTool
from src.utils.logger import setup_logging

# Load environment variables
load_dotenv()

# Setup logging
setup_logging(log_level="INFO")


async def main():
    """Run the multi-agent system with example queries."""
    # Initialize LLM (optional - system works without it using fallbacks)
    llm = None
    if os.getenv("OPENAI_API_KEY"):
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.1,
        )
        print("✓ LLM initialized (OpenAI)")
    else:
        print("⚠ LLM not configured (using fallback heuristics)")
        print("  Set OPENAI_API_KEY environment variable to enable LLM features")

    # Initialize tools
    calculator = CalculatorTool()
    text_analyzer = TextAnalyzerTool()
    tools: List = [calculator, text_analyzer]
    tools_dict = {tool.name: tool for tool in tools}

    print(f"✓ Initialized {len(tools)} tools: {[t.name for t in tools]}")

    # Initialize agents
    planner = PlannerAgent(
        tools=tools,
        llm=llm,
    )

    executor = ExecutorAgent(
        tools=tools_dict,
        llm=llm,
    )

    critic = CriticAgent(llm=llm)

    manager = ManagerAgent(
        planner=planner,
        executor=executor,
        critic=critic,
        max_iterations=10,
    )

    print("✓ All agents initialized\n")

    # Example queries
    example_queries = [
        "Calculate 15 multiplied by 23, then add 100 to the result",
        "Analyze the text: 'The quick brown fox jumps over the lazy dog' and count the words",
    ]

    for i, query in enumerate(example_queries, 1):
        print("=" * 80)
        print(f"Example {i}: {query}")
        print("=" * 80)

        # Create initial state
        state = AgentState(
            user_query=query,
            max_iterations=10,
        )

        # Run orchestration
        try:
            result_message = await manager.process(state)

            print(f"\n📋 Final Result:")
            print(f"   {result_message.content}")
            print(f"\n📊 Execution Summary:")
            print(f"   Iterations: {state.iteration_count}")
            print(f"   Steps completed: {state.current_step_index}/{len(state.plan.steps) if state.plan else 0}")
            print(f"   Messages exchanged: {len(state.messages)}")

            if state.plan:
                print(f"\n📝 Execution Plan:")
                for step in state.plan.steps:
                    status_icon = "✓" if step.status == "completed" else "✗" if step.status == "failed" else "○"
                    print(f"   {status_icon} Step {step.step_id}: {step.description}")
                    if step.result:
                        print(f"      Result: {step.result}")

        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()

        print("\n")


if __name__ == "__main__":
    asyncio.run(main())

