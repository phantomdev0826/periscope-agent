from __future__ import annotations

import asyncio
import json
import uuid

import streamlit as st

from agent.core.config import settings
from agent.core.logging import configure_logging
from agent.graph.graph import GraphHandle, build_persistent_graph
from agent.graph.state import AgentState
from agent.schemas import Report

configure_logging()

st.set_page_config(page_title="Research Agent", page_icon="🔎", layout="wide")


@st.cache_resource(show_spinner="Initializing research agent…")
def _agent_runtime() -> tuple[asyncio.AbstractEventLoop, GraphHandle]:
    """Build the compiled graph + checkpointer once per process and share it
    across Streamlit reruns. The AsyncPostgresSaver pool is bound to whichever
    event loop opens it, so we keep that loop alive too and route every
    submission through `loop.run_until_complete(...)`.

    Streamlit's `cache_resource` survives reruns within a single process and
    is invalidated only on code change or explicit clear."""
    loop = asyncio.new_event_loop()
    handle = loop.run_until_complete(build_persistent_graph())
    return loop, handle


# ---- header ----
st.title("LangGraph Research Agent")
st.caption(
    "Multi-step research: decompose → search → fetch → summarize → synthesize → critique → write. "
    "Conditional re-search if confidence < threshold. State persisted to Postgres (resumable)."
)


# ---- sidebar ----
with st.sidebar:
    st.header("Config")
    st.write(f"**Model:** `{settings.anthropic_model}`")
    st.write(f"**Search:** `{settings.effective_search_provider}`")
    st.write(f"**LangSmith:** {'on' if settings.langsmith_enabled else 'off'}")
    st.write(f"**Max iterations:** {settings.max_iterations}")
    st.write(f"**Confidence threshold:** {settings.confidence_threshold}")
    if not settings.anthropic_api_key:
        st.error("ANTHROPIC_API_KEY is not set — agent will fail when it hits the LLM.")
    if settings.search_provider == "tavily" and not settings.tavily_api_key:
        st.warning("SEARCH_PROVIDER=tavily but no TAVILY_API_KEY — falling back to `mock`.")


# ---- main form ----
question = st.text_area(
    "Research question",
    placeholder="e.g. What production patterns are essential for shipping a RAG system?",
    height=80,
)

run = st.button("Run research", type="primary", disabled=not question.strip())


async def _run_agent(graph, q: str, thread_id: str):  # type: ignore[no-untyped-def]
    initial: AgentState = {"original_question": q}
    config = {"configurable": {"thread_id": thread_id}}

    async for event in graph.astream_events(initial, config=config, version="v2"):
        kind = event.get("event")
        name = event.get("name", "")
        if kind == "on_chain_end" and name in {
            "decompose",
            "search",
            "summarize",
            "synthesize",
            "critique",
            "write_report",
        }:
            yield ("node_end", name, event.get("data", {}).get("output", {}))

    final_state = await graph.aget_state(config)
    yield ("final", "", final_state.values)


if run and question.strip():
    thread_id = str(uuid.uuid4())
    st.session_state["thread_id"] = thread_id

    progress = st.empty()
    events_box = st.expander("Trace", expanded=False)
    events_log: list[str] = []
    result: dict[str, object] = {"report": None, "state": None}

    loop, handle = _agent_runtime()

    async def drive() -> None:
        async for kind, name, payload in _run_agent(handle.graph, question, thread_id):
            if kind == "node_end":
                progress.info(f"✔ {name}")
                if isinstance(payload, dict):
                    new_events = payload.get("events", [])
                    if isinstance(new_events, list):
                        events_log.extend(str(e) for e in new_events)
                        with events_box:
                            for line in events_log:
                                st.text(line)
            elif kind == "final":
                result["state"] = payload
                fr = payload.get("final_report") if isinstance(payload, dict) else None
                if isinstance(fr, Report):
                    result["report"] = fr
                elif isinstance(fr, dict):
                    result["report"] = Report.model_validate(fr)

    with st.spinner("Researching…"):
        # Reuse the cached loop so the AsyncPostgresSaver pool (bound to it)
        # stays valid across submissions. asyncio.run would create a new loop
        # each time, invalidating the pool.
        loop.run_until_complete(drive())

    final_report = result["report"] if isinstance(result["report"], Report) else None
    final_state_snapshot = result["state"] if isinstance(result["state"], dict) else None

    progress.success("Done.")

    if final_report:
        st.subheader("Executive summary")
        st.write(final_report.executive_summary)

        cols = st.columns(3)
        cols[0].metric("Confidence", f"{final_report.confidence:.2f}")
        cols[1].metric("Iterations", final_report.iterations)
        cols[2].metric("Sources", len(final_report.sources))

        st.subheader("Sub-question findings")
        for sq in final_report.sub_questions:
            with st.expander(f"{sq.question}  ·  confidence {sq.confidence:.2f}"):
                if not sq.findings:
                    st.caption("No findings.")
                for f in sq.findings:
                    st.markdown(f"**{f.claim}**")
                    for c in f.citations:
                        st.markdown(f"- [{c.title}]({c.url}) — {c.snippet}")

        st.subheader("All sources")
        for c in final_report.sources:
            st.markdown(f"- [{c.title}]({c.url})")

        with st.expander("Raw report JSON"):
            st.code(final_report.model_dump_json(indent=2), language="json")
    else:
        st.error("No final report produced. See trace.")

    if final_state_snapshot:
        with st.expander("Full final state (from checkpointer)"):
            st.code(
                json.dumps(
                    {k: str(v)[:1000] for k, v in final_state_snapshot.items()},
                    indent=2,
                ),
                language="json",
            )
