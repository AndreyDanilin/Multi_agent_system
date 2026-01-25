"""State schemas for agent orchestration."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from .messages import AgentMessage


class ExecutionStep(BaseModel):
    """A single step in the execution plan."""

    step_id: int = Field(..., description="Unique step identifier")
    description: str = Field(..., description="Human-readable step description")
    tool_name: Optional[str] = Field(
        default=None, description="Tool to execute this step"
    )
    tool_args: Dict[str, Any] = Field(
        default_factory=dict, description="Arguments for the tool"
    )
    status: str = Field(
        default="pending", description="Step status: pending, executing, completed, failed"
    )
    result: Optional[Any] = Field(default=None, description="Step execution result")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional step metadata"
    )


class Plan(BaseModel):
    """Execution plan created by PlannerAgent."""

    goal: str = Field(..., description="Original user goal")
    steps: List[ExecutionStep] = Field(..., description="Ordered list of execution steps")
    created_at: Optional[float] = Field(default=None, description="Plan creation timestamp")


class AgentState(BaseModel):
    """Global state for the multi-agent system."""

    user_query: str = Field(..., description="Original user query")
    messages: List[AgentMessage] = Field(
        default_factory=list, description="Conversation history"
    )
    plan: Optional[Plan] = Field(default=None, description="Current execution plan")
    current_step_index: int = Field(
        default=0, description="Index of currently executing step"
    )
    execution_results: List[Dict[str, Any]] = Field(
        default_factory=list, description="Accumulated execution results"
    )
    iteration_count: int = Field(
        default=0, description="Number of orchestration iterations"
    )
    max_iterations: int = Field(
        default=10, description="Maximum allowed iterations"
    )
    is_complete: bool = Field(
        default=False, description="Whether the task is complete"
    )
    final_result: Optional[str] = Field(
        default=None, description="Final result to return to user"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional state metadata"
    )

    class Config:
        """Pydantic config."""

        arbitrary_types_allowed = True

