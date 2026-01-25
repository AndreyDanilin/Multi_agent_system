"""CriticAgent implementation."""

from typing import Any, Optional

from langchain_openai import ChatOpenAI
from pydantic import Field

from ..prompts.critic import CRITIC_PROMPT
from ..schemas.messages import AgentMessage, MessageRole, MessageType
from ..schemas.state import AgentState
from .base import AgentBase


class CriticAgent(AgentBase):
    """Agent responsible for reviewing execution results and detecting issues."""

    name: str = "CriticAgent"
    role: MessageRole = MessageRole.CRITIC
    llm: Optional[ChatOpenAI] = None

    async def process(
        self, state: AgentState, **kwargs: Any
    ) -> AgentMessage:
        """Review execution results and provide critique.

        Args:
            state: Current system state
            **kwargs: Additional context

        Returns:
            AgentMessage with critique
        """
        self.log_decision("Reviewing execution results")

        if not state.plan:
            return self.create_message(
                content="No plan available for review",
                message_type=MessageType.ERROR,
            )

        # Format execution results
        execution_results = self._format_execution_results(state)

        # Create prompt
        prompt = CRITIC_PROMPT.format(
            plan=state.plan.model_dump_json(indent=2),
            execution_results=execution_results,
            user_query=state.user_query,
        )

        # Get LLM response
        assessment = "complete"
        issues = []
        needs_refinement = False

        if self.llm is not None:
            try:
                response = await self.llm.ainvoke(prompt)
                # Parse response (simplified - in production, use structured output)
                content = response.content.lower()
                if "incomplete" in content or "needs refinement" in content:
                    needs_refinement = True
                    assessment = "needs_refinement"
                if "issue" in content or "error" in content:
                    issues.append("Issues detected in execution")
            except Exception as e:
                self.logger.error("LLM error in critic", error=str(e))
                # Fallback: basic validation
                assessment = self._basic_validation(state)

        # Determine if task is complete
        all_steps_completed = all(
            step.status == "completed"
            for step in state.plan.steps
        )
        all_steps_present = len(state.execution_results) >= len(state.plan.steps)

        if not all_steps_completed or not all_steps_present:
            needs_refinement = True
            assessment = "incomplete"

        content = (
            f"Review complete. Assessment: {assessment}. "
            f"Issues found: {len(issues)}. "
            f"Needs refinement: {needs_refinement}"
        )

        message = self.create_message(
            content=content,
            message_type=MessageType.CRITIQUE,
            metadata={
                "assessment": assessment,
                "issues": issues,
                "needs_refinement": needs_refinement,
                "all_steps_completed": all_steps_completed,
            },
        )

        self.log_decision(
            "Review completed",
            {"assessment": assessment, "needs_refinement": needs_refinement},
        )

        return message

    def _format_execution_results(self, state: AgentState) -> str:
        """Format execution results for prompt."""
        if not state.execution_results:
            return "No execution results yet."

        formatted = []
        for i, result in enumerate(state.execution_results):
            formatted.append(f"Step {i+1}: {result}")
        return "\n".join(formatted)

    def _basic_validation(self, state: AgentState) -> str:
        """Perform basic validation without LLM.

        Args:
            state: Current state

        Returns:
            Assessment string
        """
        if not state.plan:
            return "incomplete"

        completed_steps = sum(
            1 for step in state.plan.steps if step.status == "completed"
        )
        total_steps = len(state.plan.steps)

        if completed_steps == total_steps:
            return "complete"
        elif completed_steps > 0:
            return "partial"
        else:
            return "incomplete"

