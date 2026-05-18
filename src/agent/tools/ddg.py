from __future__ import annotations

from urllib.parse import quote_plus

import httpx
from bs4 import BeautifulSoup
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from agent.core.logging import get_logger
from agent.schemas import SearchResult

log = get_logger(__name__)


class DDGSearchTool:
    """No-key DuckDuckGo HTML scrape. Suitable for development/demos but rate-limited;
    not recommended for production. If DDG blocks, callers should fall back to mock."""

    name = "ddg"
    base_url = "https://html.duckduckgo.com/html/"

    async def search(self, query: str, *, limit: int) -> list[SearchResult]:
        params = {"q": query}
        headers = {"User-Agent": "Mozilla/5.0 (compatible; research-agent/0.1)"}
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=1, max=8),
            retry=retry_if_exception_type((httpx.HTTPError,)),
            reraise=True,
        ):
            with attempt:
                async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
                    resp = await client.post(self.base_url, data=params, headers=headers)
                    resp.raise_for_status()
                    html = resp.text

        soup = BeautifulSoup(html, "lxml")
        results: list[SearchResult] = []
        for r in soup.select("div.result")[:limit]:
            a = r.select_one("a.result__a")
            snippet_el = r.select_one("a.result__snippet")
            if a is None:
                continue
            url = a.get("href") or ""
            if not isinstance(url, str):
                continue
            # DDG HTML wraps real URLs in a /l/?uddg= redirect — pull out the actual target.
            url = self._unwrap(url, query)
            results.append(
                SearchResult(
                    url=url,
                    title=a.get_text(strip=True),
                    snippet=snippet_el.get_text(strip=True) if snippet_el else "",
                    source_quality=0.5,
                )
            )
        log.info("ddg_search", query=query[:80], hits=len(results))
        return results

    @staticmethod
    def _unwrap(url: str, query: str) -> str:
        from urllib.parse import parse_qs, urlparse

        if "duckduckgo.com" in url and "uddg=" in url:
            qs = parse_qs(urlparse(url).query)
            target = qs.get("uddg", [""])[0]
            if target:
                return target
        # Last-ditch: synthesize a search-results-of URL so we still have something to attribute.
        return url or f"https://duckduckgo.com/?q={quote_plus(query)}"
