"""ExecutorAgent implementation."""

from typing import Any, Dict, Optional

from langchain_openai import ChatOpenAI
from pydantic import Field

from ..prompts.executor import EXECUTOR_PROMPT
from ..schemas.messages import AgentMessage, MessageRole, MessageType
from ..schemas.state import AgentState, ExecutionStep
from ..tools.base import ToolBase
from .base import AgentBase


class ExecutorAgent(AgentBase):
    """Agent responsible for executing individual plan steps."""

    name: str = "ExecutorAgent"
    role: MessageRole = MessageRole.EXECUTOR
    tools: Dict[str, ToolBase] = Field(default_factory=dict)
    llm: Optional[ChatOpenAI] = None

    async def process(
        self, state: AgentState, **kwargs: Any
    ) -> AgentMessage:
        """Execute current step from the plan.

        Args:
            state: Current system state
            **kwargs: Additional context

        Returns:
            AgentMessage with execution result
        """
        if not state.plan or not state.plan.steps:
            return self.create_message(
                content="No plan or steps available for execution",
                message_type=MessageType.ERROR,
            )

        current_step = state.plan.steps[state.current_step_index]
        self.log_decision(
            "Executing step",
            {
                "step_id": current_step.step_id,
                "tool": current_step.tool_name,
            },
        )

        # Update step status
        current_step.status = "executing"

        # Execute the step
        try:
            if current_step.tool_name:
                result = await self._execute_tool(current_step)
            else:
                result = {"status": "completed", "message": current_step.description}

            current_step.status = "completed"
            current_step.result = result

            content = f"Step {current_step.step_id} completed successfully"
            message_type = MessageType.EXECUTION_RESULT

        except Exception as e:
            current_step.status = "failed"
            current_step.error = str(e)
            result = {"status": "failed", "error": str(e)}
            content = f"Step {current_step.step_id} failed: {str(e)}"
            message_type = MessageType.ERROR
            self.logger.error("Step execution failed", step_id=current_step.step_id, error=str(e))

        message = self.create_message(
            content=content,
            message_type=message_type,
            metadata={
                "step_id": current_step.step_id,
                "result": result,
            },
        )

        self.log_decision("Step execution completed", {"step_id": current_step.step_id})

        return message

    async def _execute_tool(self, step: ExecutionStep) -> Dict[str, Any]:
        """Execute tool for the given step.

        Args:
            step: Execution step with tool information

        Returns:
            Tool execution result

        Raises:
            ValueError: If tool is not found or execution fails
        """
        if not step.tool_name:
            raise ValueError("No tool specified for step")

        tool = self.tools.get(step.tool_name)
        if not tool:
            raise ValueError(f"Tool '{step.tool_name}' not found")

        # Merge tool_args from step with any additional context
        tool_args = step.tool_args.copy()

        # Handle chained operations (use previous result as input)
        if step.tool_name == "calculator" and tool_args.get("a") == 0:
            # This might be a chained operation - check if we have previous results
            # In a more sophisticated implementation, we'd look at state.execution_results
            # For now, we'll use the provided value
            pass

        # Execute tool
        result = await tool.execute(**tool_args)

        return {
            "tool": step.tool_name,
            "tool_args": tool_args,
            "result": result,
        }

