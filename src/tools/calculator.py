"""Calculator tool for mathematical operations."""

from typing import Any, Dict

from .base import ToolBase


class CalculatorTool(ToolBase):
    """Tool for performing mathematical calculations.

    Supports basic arithmetic operations: addition, subtraction,
    multiplication, division, and power.
    """

    name: str = "calculator"
    description: str = (
        "Performs mathematical calculations. Supports operations: "
        "add, subtract, multiply, divide, power. "
        "Example: {'operation': 'add', 'a': 5, 'b': 3} returns 8"
    )

    async def execute(self, **kwargs: Any) -> Dict[str, Any]:
        """Execute calculation.

        Args:
            **kwargs: Must contain 'operation', 'a', and optionally 'b'

        Returns:
            Dictionary with 'result' key containing the calculation result

        Raises:
            ValueError: If operation or arguments are invalid
        """
        operation = kwargs.get("operation")
        a = kwargs.get("a")
        b = kwargs.get("b")

        if operation is None or a is None:
            raise ValueError("Missing required arguments: operation, a")

        try:
            a = float(a)
            if b is not None:
                b = float(b)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid numeric arguments: {e}")

        operations = {
            "add": lambda x, y: x + y,
            "subtract": lambda x, y: x - y,
            "multiply": lambda x, y: x * y,
            "divide": lambda x, y: x / y if y != 0 else float("inf"),
            "power": lambda x, y: x ** y,
        }

        if operation not in operations:
            raise ValueError(
                f"Unknown operation: {operation}. "
                f"Supported: {list(operations.keys())}"
            )

        if operation in ["divide", "subtract", "multiply", "power"] and b is None:
            raise ValueError(f"Operation '{operation}' requires argument 'b'")

        if operation == "add" and b is None:
            # For addition, if b is not provided, return a as-is
            result = a
        else:
            result = operations[operation](a, b)

        return {
            "result": result,
            "operation": operation,
            "inputs": {"a": a, "b": b} if b is not None else {"a": a},
        }

    def _get_parameters_schema(self) -> Dict[str, Any]:
        """Get JSON schema for calculator parameters."""
        return {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["add", "subtract", "multiply", "divide", "power"],
                    "description": "Mathematical operation to perform",
                },
                "a": {
                    "type": "number",
                    "description": "First number",
                },
                "b": {
                    "type": "number",
                    "description": "Second number (required for all operations except add)",
                },
            },
            "required": ["operation", "a"],
        }

