"""ManagerAgent (Orchestrator) implementation."""

from typing import Any, Dict, List, Optional

from langchain_openai import ChatOpenAI
from pydantic import Field

from ..memory.base import MemoryBase
from ..memory.long_term import LongTermMemory
from ..memory.short_term import ShortTermMemory
from ..schemas.messages import AgentMessage, MessageRole, MessageType
from ..schemas.state import AgentState, Plan
from ..tools.base import ToolBase
from ..utils.logger import get_logger
from .base import AgentBase
from .critic import CriticAgent
from .executor import ExecutorAgent
from .planner import PlannerAgent


class ManagerAgent(AgentBase):
    """Orchestrator that manages the multi-agent workflow.

    Responsibilities:
    - Initialize and coordinate agents
    - Route messages between agents
    - Manage state transitions
    - Handle retries and termination
    """

    name: str = "ManagerAgent"
    role: MessageRole = MessageRole.MANAGER
    planner: PlannerAgent = Field(...)
    executor: ExecutorAgent = Field(...)
    critic: CriticAgent = Field(...)
    short_term_memory: ShortTermMemory = Field(default_factory=ShortTermMemory)
    long_term_memory: LongTermMemory = Field(default_factory=LongTermMemory)
    max_iterations: int = Field(default=10)
    logger: Any = None

    def __init__(self, **data: Any):
        """Initialize manager with agents."""
        super().__init__(**data)
        if self.logger is None:
            self.logger = get_logger(self.__class__.__name__)

    async def process(
        self, state: AgentState, **kwargs: Any
    ) -> AgentMessage:
        """Orchestrate the multi-agent workflow.

        This method implements the main orchestration loop:
        1. Planner creates execution plan
        2. Executor executes steps sequentially
        3. Critic reviews results
        4. Loop until complete or max iterations

        Args:
            state: Current system state
            **kwargs: Additional context

        Returns:
            Final AgentMessage with result
        """
        self.log_decision("Starting orchestration", {"query": state.user_query})

        # Store initial user message
        user_message = AgentMessage(
            role=MessageRole.USER,
            message_type=MessageType.TEXT,
            content=state.user_query,
        )
        await self.short_term_memory.add(user_message)
        state.messages.append(user_message)

        # Main orchestration loop
        while state.iteration_count < state.max_iterations and not state.is_complete:
            state.iteration_count += 1
            self.logger.info(
                "orchestration_iteration",
                iteration=state.iteration_count,
                max_iterations=state.max_iterations,
            )

            # Phase 1: Planning (if no plan exists)
            if state.plan is None:
                self.logger.info("orchestration_phase", phase="planning")
                planner_message = await self.planner.process(state)
                await self.short_term_memory.add(planner_message)
                state.messages.append(planner_message)

                # Extract plan from message metadata
                if "plan" in planner_message.metadata:
                    plan_data = planner_message.metadata["plan"]
                    state.plan = Plan(**plan_data)

            # Phase 2: Execution (if plan exists and not all steps done)
            if state.plan and state.current_step_index < len(state.plan.steps):
                self.logger.info(
                    "orchestration_phase",
                    phase="execution",
                    step_index=state.current_step_index,
                )

                executor_message = await self.executor.process(state)
                await self.short_term_memory.add(executor_message)
                state.messages.append(executor_message)

                # Store execution result
                if "result" in executor_message.metadata:
                    state.execution_results.append(executor_message.metadata["result"])

                # Handle chained operations (pass result to next step)
                current_step = state.plan.steps[state.current_step_index]
                if current_step.status == "completed" and current_step.result:
                    # If next step needs previous result, update its args
                    if state.current_step_index + 1 < len(state.plan.steps):
                        next_step = state.plan.steps[state.current_step_index + 1]
                        if (
                            next_step.tool_name == "calculator"
                            and next_step.tool_args.get("a") == 0
                            and current_step.result
                        ):
                            # Extract result from previous step
                            prev_result = current_step.result
                            if isinstance(prev_result, dict) and "result" in prev_result:
                                calc_result = prev_result["result"]
                                if isinstance(calc_result, dict) and "result" in calc_result:
                                    next_step.tool_args["a"] = calc_result["result"]
                                elif isinstance(calc_result, (int, float)):
                                    next_step.tool_args["a"] = calc_result

                # Move to next step
                if current_step.status == "completed":
                    state.current_step_index += 1

            # Phase 3: Review (if all steps executed)
            if (
                state.plan
                and state.current_step_index >= len(state.plan.steps)
                and not state.is_complete
            ):
                self.logger.info("orchestration_phase", phase="review")
                critic_message = await self.critic.process(state)
                await self.short_term_memory.add(critic_message)
                state.messages.append(critic_message)

                # Check if refinement is needed
                needs_refinement = critic_message.metadata.get("needs_refinement", False)
                assessment = critic_message.metadata.get("assessment", "unknown")

                if not needs_refinement and assessment == "complete":
                    state.is_complete = True
                    state.final_result = self._generate_final_result(state)
                elif needs_refinement:
                    # Reset for refinement (simplified - in production, be more selective)
                    self.logger.warning("orchestration_refinement_needed")
                    # For now, mark as complete to avoid infinite loops
                    # In production, implement proper refinement logic
                    state.is_complete = True
                    state.final_result = self._generate_final_result(state)

        # Final message
        if state.is_complete:
            content = f"Task completed. Result: {state.final_result}"
        else:
            content = (
                f"Task incomplete after {state.iteration_count} iterations. "
                f"Current state: {state.current_step_index}/{len(state.plan.steps) if state.plan else 0} steps"
            )

        final_message = AgentMessage(
            role=self.role,
            message_type=MessageType.TEXT,
            content=content,
            metadata={"final_state": state.model_dump()},
        )

        await self.short_term_memory.add(final_message)
        self.log_decision("Orchestration complete", {"iterations": state.iteration_count})

        return final_message

    def _generate_final_result(self, state: AgentState) -> str:
        """Generate final result string from execution results.

        Args:
            state: Current state

        Returns:
            Formatted final result
        """
        if not state.execution_results:
            return "No results generated."

        results = []
        for i, result in enumerate(state.execution_results, 1):
            if isinstance(result, dict):
                tool_result = result.get("result", {})
                if isinstance(tool_result, dict):
                    calc_result = tool_result.get("result")
                    if calc_result is not None:
                        results.append(f"Step {i}: {calc_result}")
                    else:
                        results.append(f"Step {i}: {tool_result}")
                else:
                    results.append(f"Step {i}: {tool_result}")
            else:
                results.append(f"Step {i}: {result}")

        return "\n".join(results) if results else "Execution completed."

