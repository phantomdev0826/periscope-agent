from __future__ import annotations

from pathlib import Path

from agent.tools.mock import MockSearchTool

CORPUS = Path(__file__).resolve().parents[1] / "data" / "mock_corpus"


async def test_mock_returns_results_for_known_topic() -> None:
    tool = MockSearchTool(corpus_dir=CORPUS)
    hits = await tool.search("reciprocal rank fusion BM25 hybrid retrieval", limit=5)
    assert hits, "expected at least one hit on a topic covered by the seed corpus"
    assert any("rag" in h.url.lower() or "rag" in h.title.lower() for h in hits)


async def test_mock_returns_empty_for_unrelated_query() -> None:
    tool = MockSearchTool(corpus_dir=CORPUS)
    hits = await tool.search("zzzzzz unrelatable nonsense aaaaaa", limit=5)
    assert hits == []


async def test_mock_ranks_more_overlapping_doc_first() -> None:
    tool = MockSearchTool(corpus_dir=CORPUS)
    hits = await tool.search("pgvector hnsw cosine postgres extension", limit=3)
    assert hits
    assert "pgvector" in hits[0].url.lower() or "pgvector" in hits[0].title.lower()
