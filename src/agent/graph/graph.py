from __future__ import annotations

from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

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


@dataclass
class GraphHandle:
    """A long-lived compiled graph + the saver it owns.

    Designed to be created once per process and reused across many requests.
    Streamlit reruns the script on every interaction; without this, every
    submission would re-open the AsyncPostgresSaver context (~100-200ms of
    connection + DDL probing per submission). Construct via
    `build_persistent_graph()` and remember to call `aclose()` at shutdown."""

    graph: Any
    _saver_cm: Any = None  # the AsyncPostgresSaver async context, if Postgres reachable

    async def aclose(self) -> None:
        if self._saver_cm is not None:
            await self._saver_cm.__aexit__(None, None, None)


async def build_persistent_graph() -> GraphHandle:
    """Build a compiled graph once. Tries Postgres; falls back to in-memory if
    the connection can't be opened so the demo still works without a DB."""
    graph_def = build_graph()
    try:
        saver_cm = AsyncPostgresSaver.from_conn_string(_pg_conn_str())
        saver = await saver_cm.__aenter__()
        await saver.setup()
        log.info("graph_compiled_with_postgres_saver")
        return GraphHandle(
            graph=graph_def.compile(checkpointer=saver), _saver_cm=saver_cm
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("postgres_checkpointer_unavailable", error=str(exc), fallback="memory")
        return GraphHandle(graph=graph_def.compile(checkpointer=MemorySaver()))
