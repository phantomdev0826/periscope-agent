# LangGraph Research Agent

Multi-step research agent built on LangGraph. Decomposes a question into
sub-questions, runs web search, fetches and summarizes top sources, synthesizes
findings across sub-questions, and produces a Pydantic-validated structured
report with inline citations. The agent makes routing decisions based on
confidence, retries failed tool calls with exponential backoff, and persists
state to Postgres so interrupted runs can resume.

> **Why this exists:** most "agents" demoed online are linear chains in
> disguise — no real state management, no error recovery, no observability,
> silent production failures. This project demonstrates what a real agent
> architecture looks like.

---

## State machine

```
decompose ─→ search ─→ summarize ─→ synthesize ─→ critique
                ↑                                     │
                └──── (confidence < threshold)────────┤
                                                      │
                                                  write_report
```

Conditional edge after `critique`: if confidence is below
`CONFIDENCE_THRESHOLD` *and* iterations remain *and* the critique proposed
refinement queries, the graph loops back to `search` with those refined
queries; otherwise it finalizes.

Full state diagram in [`docs/architecture.md`](./docs/architecture.md).

## Stack

- **Agent runtime:** LangGraph 0.2 (`StateGraph`, conditional edges, async checkpointer)
- **LLM:** Claude `claude-sonnet-4-6` via `langchain-anthropic`
- **Tracing:** LangSmith — auto-enabled if `LANGCHAIN_API_KEY` + `LANGCHAIN_TRACING_V2=true`
- **Search:** Tavily (real) / DuckDuckGo HTML (no key) / local mock corpus — swappable via `SEARCH_PROVIDER`
- **Fetcher:** httpx + readability-lxml for main-content extraction
- **State persistence:** `langgraph-checkpoint-postgres` (`AsyncPostgresSaver`) — falls back to `MemorySaver` if Postgres is unreachable
- **Schemas:** Pydantic v2 (`Finding`, `SubQuestionReport`, `Report`) — invalid LLM outputs cannot escape into the report
- **Reliability:** tenacity exponential backoff on every external call, plus a circuit breaker around the search backend
- **UI:** Streamlit (live progress per node, trace expander, JSON snapshot of final state)

## Quick start

```bash
cp .env.example .env
# Set ANTHROPIC_API_KEY at minimum.
# Optional: TAVILY_API_KEY (real web search) and LANGCHAIN_API_KEY (tracing).

make up
# Streamlit: http://localhost:8501
```

Without any extra keys, the agent runs end-to-end against the bundled
mock corpus in `data/mock_corpus/`. See [`docs/sample-report.md`](./docs/sample-report.md)
for an example of what comes out.

## Search providers

| Setting              | Behavior                                                            |
| -------------------- | ------------------------------------------------------------------- |
| `SEARCH_PROVIDER=mock` (default) | Local JSON corpus; deterministic; offline-safe         |
| `SEARCH_PROVIDER=ddg`            | DuckDuckGo HTML scrape; no key required; rate-limited  |
| `SEARCH_PROVIDER=tavily`         | Tavily Search API; needs `TAVILY_API_KEY`              |

If `tavily` is selected but no key is present, the agent silently falls back
to `mock` and logs a warning. The circuit breaker around search means a dead
provider can't take down a run — after N consecutive failures, the node
switches to the mock fallback for the remainder of the run.

## What the agent produces

A Pydantic `Report`:

```python
class Report:
    question: str
    executive_summary: str          # 3-6 sentence synthesis
    sub_questions: list[SubQuestionReport]   # findings per sub-question
    sources: list[Citation]         # deduplicated source URLs
    confidence: float               # from the critique step
    iterations: int                 # 1 if no refinement loop fired
```

Every `Finding` requires at least one `Citation`. The schema enforces this,
so a hallucinated claim with no source cannot end up in the final report.

## Tracing

When `LANGCHAIN_TRACING_V2=true` and `LANGCHAIN_API_KEY` is set, every node
invocation is traced to LangSmith with inputs, outputs, latency, and token
counts. Look for the `LANGCHAIN_PROJECT` (default `research-agent`).

## Project layout

```
02-langgraph-agent/
├── src/agent/
│   ├── core/            config, logging, Claude (LangChain wrapped)
│   ├── tools/           SearchTool Protocol + tavily/ddg/mock; webpage fetcher
│   ├── graph/           state (TypedDict), nodes, routing, compiled graph
│   ├── circuit_breaker.py
│   ├── schemas.py       Pydantic input/output (Finding, Report, Critique)
│   └── app.py           Streamlit UI
├── data/mock_corpus/    JSON fixtures used by MockSearchTool
├── tests/               routing, circuit breaker, mock search, schema validation
├── docs/                architecture (mermaid) + sample report
├── docker-compose.yml   postgres + streamlit
├── Dockerfile
├── Makefile
└── .env.example
```

## Make targets

```
make up        start postgres + streamlit (http://localhost:8501)
make logs      tail container logs
make test      run pytest suite (mock search, routing, circuit breaker, schemas)
make lint      ruff + mypy strict
make format    ruff format + autofix
make psql      open psql shell
make down      stop containers
make clean     drop volumes (destructive)
```

## What this isn't

- A general-purpose tool-using agent — the toolset is intentionally narrow
  (search + fetch) because that's enough to show production patterns. Adding
  tools is just adding nodes.
- A multi-agent system — one graph, one agent. Multi-agent orchestration is
  a different problem and a different LangGraph pattern.

## License

MIT.
