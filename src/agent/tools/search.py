from __future__ import annotations

from typing import Protocol

from agent.schemas import SearchResult


class SearchTool(Protocol):
    """Search backend interface. All implementations return SearchResult objects so
    the LangGraph nodes never branch on provider type."""

    name: str

    async def search(self, query: str, *, limit: int) -> list[SearchResult]: ...


def build_search_tool() -> SearchTool:
    from agent.core.config import settings
    from agent.tools.ddg import DDGSearchTool
    from agent.tools.mock import MockSearchTool
    from agent.tools.tavily import TavilySearchTool

    provider = settings.effective_search_provider
    if provider == "tavily":
        return TavilySearchTool()
    if provider == "ddg":
        return DDGSearchTool()
    return MockSearchTool()
