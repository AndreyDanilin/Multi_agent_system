"""Text analysis tool for processing and analyzing text content."""

from typing import Any, Dict

from .base import ToolBase


class TextAnalyzerTool(ToolBase):
    """Tool for analyzing text content.

    Provides basic text analysis capabilities: word count, character count,
    and keyword extraction. Extensible for more advanced NLP features.
    """

    name: str = "text_analyzer"
    description: str = (
        "Analyzes text content. Provides word count, character count, "
        "and basic keyword extraction. "
        "Example: {'text': 'Hello world', 'analysis_type': 'word_count'} "
        "returns word count"
    )

    async def execute(self, **kwargs: Any) -> Dict[str, Any]:
        """Execute text analysis.

        Args:
            **kwargs: Must contain 'text' and optionally 'analysis_type'

        Returns:
            Dictionary with analysis results

        Raises:
            ValueError: If text is missing or invalid
        """
        text = kwargs.get("text")
        analysis_type = kwargs.get("analysis_type", "full")

        if not text or not isinstance(text, str):
            raise ValueError("Missing or invalid 'text' argument")

        results: Dict[str, Any] = {
            "text_length": len(text),
            "character_count": len(text),
            "character_count_no_spaces": len(text.replace(" ", "")),
        }

        if analysis_type in ["word_count", "full"]:
            words = text.split()
            results["word_count"] = len(words)
            results["sentence_count"] = text.count(".") + text.count("!") + text.count("?")

        if analysis_type in ["keywords", "full"]:
            # Simple keyword extraction: words longer than 4 characters
            words = text.lower().split()
            keywords = [
                word.strip(".,!?;:")
                for word in words
                if len(word.strip(".,!?;:")) > 4
            ]
            # Get unique keywords, limit to top 10
            unique_keywords = list(dict.fromkeys(keywords))[:10]
            results["keywords"] = unique_keywords

        if analysis_type == "full":
            results["analysis_type"] = "full"
        else:
            results["analysis_type"] = analysis_type

        return results

    def _get_parameters_schema(self) -> Dict[str, Any]:
        """Get JSON schema for text analyzer parameters."""
        return {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Text to analyze",
                },
                "analysis_type": {
                    "type": "string",
                    "enum": ["word_count", "keywords", "full"],
                    "description": "Type of analysis to perform. 'full' includes all metrics.",
                    "default": "full",
                },
            },
            "required": ["text"],
        }

