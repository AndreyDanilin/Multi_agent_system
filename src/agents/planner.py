"""PlannerAgent implementation."""

import json
from typing import Any, Dict, List, Optional

from langchain_openai import ChatOpenAI
from pydantic import Field

from ..prompts.planner import PLANNER_PROMPT
from ..schemas.messages import AgentMessage, MessageRole, MessageType
from ..schemas.state import AgentState, ExecutionStep, Plan
from ..tools.base import ToolBase
from .base import AgentBase


class PlannerAgent(AgentBase):
    """Agent responsible for creating execution plans from user queries."""

    name: str = "PlannerAgent"
    role: MessageRole = MessageRole.PLANNER
    tools: List[ToolBase] = Field(default_factory=list)
    llm: Optional[ChatOpenAI] = None

    async def process(
        self, state: AgentState, **kwargs: Any
    ) -> AgentMessage:
        """Analyze user query and create execution plan.

        Args:
            state: Current system state
            **kwargs: Additional context

        Returns:
            AgentMessage containing the execution plan
        """
        self.log_decision("Creating execution plan", {"user_query": state.user_query})

        # Build tools description
        tools_description = self._format_tools_description()

        # Get conversation history
        conversation_history = self._format_conversation_history(state.messages)

        # Create prompt
        prompt = PLANNER_PROMPT.format(
            tools_description=tools_description,
            user_query=state.user_query,
            conversation_history=conversation_history,
        )

        # Get LLM response
        if self.llm is None:
            # Fallback: create a simple plan if LLM is not available
            plan = self._create_simple_plan(state.user_query)
        else:
            try:
                response = await self.llm.ainvoke(prompt)
                plan = self._parse_llm_response(response.content, state.user_query)
            except Exception as e:
                self.logger.error("LLM error in planner", error=str(e))
                plan = self._create_simple_plan(state.user_query)

        # Create message
        message = self.create_message(
            content=f"Created execution plan with {len(plan.steps)} steps",
            message_type=MessageType.PLAN,
            metadata={"plan": plan.model_dump()},
        )

        self.log_decision("Plan created", {"step_count": len(plan.steps)})

        return message

    def _format_tools_description(self) -> str:
        """Format available tools for prompt."""
        if not self.tools:
            return "No tools available."

        descriptions = []
        for tool in self.tools:
            schema = tool.get_schema()
            descriptions.append(
                f"- {tool.name}: {tool.description}\n"
                f"  Parameters: {json.dumps(schema['function']['parameters'], indent=2)}"
            )
        return "\n".join(descriptions)

    def _format_conversation_history(self, messages: List[AgentMessage]) -> str:
        """Format conversation history for prompt."""
        if not messages:
            return "No previous conversation."

        formatted = []
        for msg in messages[-5:]:  # Last 5 messages
            formatted.append(f"{msg.role.value}: {msg.content}")
        return "\n".join(formatted)

    def _parse_llm_response(self, response: str, user_query: str) -> Plan:
        """Parse LLM response into structured Plan.

        Args:
            response: LLM response text
            user_query: Original user query

        Returns:
            Parsed Plan object
        """
        # Try to extract JSON from response
        try:
            # Look for JSON block in response
            if "```json" in response:
                json_start = response.find("```json") + 6
                json_end = response.find("```", json_start)
                json_str = response[json_start:json_end].strip()
            elif "```" in response:
                json_start = response.find("```") + 3
                json_end = response.find("```", json_start)
                json_str = response[json_start:json_end].strip()
            else:
                json_str = response

            plan_data = json.loads(json_str)
            steps = [
                ExecutionStep(**step) if isinstance(step, dict) else step
                for step in plan_data.get("steps", [])
            ]
            return Plan(goal=user_query, steps=steps)
        except (json.JSONDecodeError, KeyError, TypeError):
            # Fallback: create plan from text analysis
            return self._create_simple_plan(user_query)

    def _create_simple_plan(self, user_query: str) -> Plan:
        """Create a simple plan when LLM is unavailable or parsing fails.

        Args:
            user_query: User query

        Returns:
            Simple Plan object
        """
        import re

        # Heuristic-based plan creation
        query_lower = user_query.lower()

        steps = []
        step_id = 1

        # Extract numbers from query
        numbers = [float(n) for n in re.findall(r'-?\d+\.?\d*', user_query)]

        # Detect calculator operations
        if any(op in query_lower for op in ["calculate", "compute", "add", "multiply", "divide", "subtract", "power"]):
            # Detect operation type
            if "multiply" in query_lower or "multiplied" in query_lower or "*" in user_query:
                if len(numbers) >= 2:
                    steps.append(
                        ExecutionStep(
                            step_id=step_id,
                            description=f"Multiply {numbers[0]} by {numbers[1]}",
                            tool_name="calculator",
                            tool_args={"operation": "multiply", "a": numbers[0], "b": numbers[1]},
                        )
                    )
                    step_id += 1
                    # If there are more operations, use result from previous step
                    if len(numbers) > 2:
                        # Store intermediate result reference
                        steps[-1].metadata = {"next_operation": "add", "next_value": numbers[2]}

            if "add" in query_lower or "+" in user_query or "then add" in query_lower:
                if len(numbers) >= 2:
                    # Check if this is a continuation of previous step
                    if steps and steps[-1].tool_name == "calculator":
                        # This is a chained operation
                        steps.append(
                            ExecutionStep(
                                step_id=step_id,
                                description=f"Add {numbers[-1] if len(numbers) > 2 else numbers[1]} to previous result",
                                tool_name="calculator",
                                tool_args={"operation": "add", "a": 0, "b": numbers[-1] if len(numbers) > 2 else numbers[1]},
                            )
                        )
                    else:
                        steps.append(
                            ExecutionStep(
                                step_id=step_id,
                                description=f"Add {numbers[0]} and {numbers[1]}",
                                tool_name="calculator",
                                tool_args={"operation": "add", "a": numbers[0], "b": numbers[1]},
                            )
                        )
                    step_id += 1

            if "divide" in query_lower or "/" in user_query:
                if len(numbers) >= 2:
                    steps.append(
                        ExecutionStep(
                            step_id=step_id,
                            description=f"Divide {numbers[0]} by {numbers[1]}",
                            tool_name="calculator",
                            tool_args={"operation": "divide", "a": numbers[0], "b": numbers[1]},
                        )
                    )
                    step_id += 1

            if "subtract" in query_lower or "-" in user_query:
                if len(numbers) >= 2:
                    steps.append(
                        ExecutionStep(
                            step_id=step_id,
                            description=f"Subtract {numbers[1]} from {numbers[0]}",
                            tool_name="calculator",
                            tool_args={"operation": "subtract", "a": numbers[0], "b": numbers[1]},
                        )
                    )
                    step_id += 1

        # Detect text analysis
        if any(word in query_lower for word in ["analyze", "text", "count", "word"]):
            # Try to extract text from query (between quotes or after "text:")
            text_match = re.search(r'["\']([^"\']+)["\']|text:\s*([^,\.]+)', user_query, re.IGNORECASE)
            text_to_analyze = text_match.group(1) or text_match.group(2) if text_match else ""
            
            if not text_to_analyze:
                # Try to find text after "analyze"
                analyze_match = re.search(r'analyze[:\s]+(.+?)(?:\.|$)', user_query, re.IGNORECASE)
                text_to_analyze = analyze_match.group(1).strip() if analyze_match else ""

            steps.append(
                ExecutionStep(
                    step_id=step_id,
                    description=f"Analyze text: {text_to_analyze[:50]}..." if len(text_to_analyze) > 50 else f"Analyze text: {text_to_analyze}",
                    tool_name="text_analyzer",
                    tool_args={"text": text_to_analyze, "analysis_type": "full"},
                )
            )
            step_id += 1

        # Default: single step if nothing detected
        if not steps:
            steps.append(
                ExecutionStep(
                    step_id=1,
                    description=f"Process user request: {user_query}",
                    tool_name=None,
                    tool_args={},
                )
            )

        return Plan(goal=user_query, steps=steps)

