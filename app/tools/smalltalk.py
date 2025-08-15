from typing import Optional

from app.tools.llm import generate_smalltalk_opening


def smalltalk_opening(user_text: str, language: str = "tr") -> str:
    """Return a brief, on-task small talk opening in Turkish.

    Uses LLM generation when available, falls back to a simple rule-based greeting.
    """
    return generate_smalltalk_opening(user_text, language)


__all__ = ["smalltalk_opening"]
