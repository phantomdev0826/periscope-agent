from __future__ import annotations

import httpx
from readability import Document
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from agent.core.logging import get_logger

log = get_logger(__name__)


async def fetch_text(url: str, *, max_chars: int = 8000) -> str | None:
    """Fetch a URL and return the readable main-content text (or None on failure).

    `local://` URLs are handled by the mock search tool, which already populates
    `fetched_text` directly; this function is only used for real HTTP sources.
    """
    if not url.startswith(("http://", "https://")):
        return None

    async for attempt in AsyncRetrying(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=6),
        retry=retry_if_exception_type((httpx.HTTPError,)),
        reraise=True,
    ):
        with attempt:
            async with httpx.AsyncClient(
                timeout=20.0,
                follow_redirects=True,
                headers={"User-Agent": "research-agent/0.1"},
            ) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                html = resp.text

    try:
        doc = Document(html)
        from bs4 import BeautifulSoup
        text = BeautifulSoup(doc.summary(), "lxml").get_text("\n", strip=True)
    except Exception as exc:  # noqa: BLE001
        log.warning("readability_failed", url=url, error=str(exc))
        return None

    text = text[:max_chars]
    log.info("fetched", url=url, chars=len(text))
    return text
