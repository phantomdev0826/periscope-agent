from __future__ import annotations

import asyncio

from tavily import TavilyClient
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from agent.core.config import settings
from agent.core.logging import get_logger
from agent.schemas import SearchResult

log = get_logger(__name__)


class TavilySearchTool:
    name = "tavily"

    def __init__(self) -> None:
        if not settings.tavily_api_key:
            raise RuntimeError("TAVILY_API_KEY not set")
        self._client = TavilyClient(api_key=settings.tavily_api_key)

    async def search(self, query: str, *, limit: int) -> list[SearchResult]:
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            retry=retry_if_exception_type((TimeoutError, ConnectionError)),
            reraise=True,
        ):
            with attempt:
                raw = await asyncio.to_thread(
                    self._client.search,
                    query=query,
                    max_results=limit,
                    search_depth="basic",
                )

        results: list[SearchResult] = []
        for r in raw.get("results", []):
            results.append(
                SearchResult(
                    url=r.get("url", ""),
                    title=r.get("title", ""),
                    snippet=r.get("content", ""),
                    source_quality=float(r.get("score", 0.5)),
                )
            )
        log.info("tavily_search", query=query[:80], hits=len(results))
        return results
