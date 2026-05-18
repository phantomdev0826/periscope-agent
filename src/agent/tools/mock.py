from __future__ import annotations

import json
import re
from pathlib import Path

from agent.core.logging import get_logger
from agent.schemas import SearchResult

log = get_logger(__name__)

_CORPUS_PATH = Path(__file__).resolve().parents[3] / "data" / "mock_corpus"

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(s: str) -> set[str]:
    return set(_TOKEN_RE.findall(s.lower()))


class MockSearchTool:
    """Deterministic search backed by a local JSON corpus.

    Each fixture is a single JSON object: {url, title, snippet, body}. The body
    contributes to scoring so MockSearchTool gives reasonable ranking, not just
    a frozen list. Useful for tests, demos, and offline development."""

    name = "mock"

    def __init__(self, corpus_dir: Path | None = None) -> None:
        self._docs: list[dict[str, str]] = []
        path = corpus_dir or _CORPUS_PATH
        if not path.exists():
            log.warning("mock_corpus_missing", path=str(path))
            return
        for p in sorted(path.glob("*.json")):
            try:
                self._docs.append(json.loads(p.read_text(encoding="utf-8")))
            except json.JSONDecodeError as exc:
                log.warning("mock_corpus_bad_json", path=str(p), error=str(exc))

    async def search(self, query: str, *, limit: int) -> list[SearchResult]:
        if not self._docs:
            return []
        q_tokens = _tokenize(query)
        scored: list[tuple[float, dict[str, str]]] = []
        for d in self._docs:
            text = " ".join([d.get("title", ""), d.get("snippet", ""), d.get("body", "")])
            d_tokens = _tokenize(text)
            if not d_tokens:
                continue
            overlap = len(q_tokens & d_tokens)
            if overlap == 0:
                continue
            score = overlap / max(len(q_tokens), 1)
            scored.append((score, d))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = [
            SearchResult(
                url=d["url"],
                title=d.get("title", ""),
                snippet=d.get("snippet", ""),
                source_quality=min(1.0, score + 0.4),
                fetched_text=d.get("body"),
            )
            for score, d in scored[:limit]
        ]
        log.info("mock_search", query=query[:80], hits=len(results))
        return results
