from __future__ import annotations

from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import END, START, StateGraph

from agent.core.config import settings
from agent.core.logging import get_logger
from agent.graph.nodes import (
    critique,
    decompose,
    search,
    summarize,
    synthesize,
    write_report,
)
from agent.graph.routing import after_critique
from agent.graph.state import AgentState

log = get_logger(__name__)


def build_graph() -> StateGraph:
    g: StateGraph = StateGraph(AgentState)
    g.add_node("decompose", decompose)
    g.add_node("search", search)
    g.add_node("summarize", summarize)
    g.add_node("synthesize", synthesize)
    g.add_node("critique", critique)
    g.add_node("write_report", write_report)

    g.add_edge(START, "decompose")
    g.add_edge("decompose", "search")
    g.add_edge("search", "summarize")
    g.add_edge("summarize", "synthesize")
    g.add_edge("synthesize", "critique")
    g.add_conditional_edges(
        "critique",
        after_critique,
        {"refine": "search", "finalize": "write_report"},
    )
    g.add_edge("write_report", END)
    return g


@asynccontextmanager
async def compiled_graph() -> AsyncIterator[object]:
    """Yield a compiled graph with a Postgres checkpointer if reachable,
    falling back to in-memory if Postgres isn't up (so dev/tests still work)."""
    graph = build_graph()
    try:
        async with AsyncPostgresSaver.from_conn_string(_pg_conn_str()) as saver:
            await saver.setup()
            yield graph.compile(checkpointer=saver)
    except Exception as exc:  # noqa: BLE001
        log.warning("postgres_checkpointer_unavailable", error=str(exc), fallback="memory")
        yield graph.compile(checkpointer=MemorySaver())


def _pg_conn_str() -> str:
    # langgraph-checkpoint-postgres uses the libpq dsn, not the SQLAlchemy URL.
    url = settings.database_url
    return url.replace("postgresql+asyncpg://", "postgresql://").replace(
        "postgresql+psycopg://", "postgresql://"
    )
