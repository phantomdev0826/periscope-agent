from __future__ import annotations

import os
from functools import lru_cache

from langchain_anthropic import ChatAnthropic

from agent.core.config import settings


@lru_cache(maxsize=1)
def chat_model(*, temperature: float = 0.2, max_tokens: int = 1024) -> ChatAnthropic:
    """Build a LangChain Claude client. LangSmith picks up tracing automatically
    from the LANGCHAIN_TRACING_V2 / LANGCHAIN_API_KEY env vars we mirror below."""
    if settings.langsmith_enabled:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key
        os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project

    return ChatAnthropic(
        model=settings.anthropic_model,  # type: ignore[call-arg]
        anthropic_api_key=settings.anthropic_api_key,  # type: ignore[arg-type]
        temperature=temperature,
        max_tokens=max_tokens,
    )
