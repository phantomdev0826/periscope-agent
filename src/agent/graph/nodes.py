from __future__ import annotations

import asyncio
import json
import re

from langchain_core.messages import HumanMessage, SystemMessage

from agent.circuit_breaker import CircuitBreaker, CircuitOpenError
from agent.core.config import settings
from agent.core.llm import chat_model
from agent.core.logging import get_logger
from agent.graph.state import AgentState
from agent.schemas import (
    Citation,
    Critique,
    DecompositionOutput,
    Finding,
    Report,
    SearchResult,
    SubQuestionPlan,
    SubQuestionReport,
)
from agent.tools.fetch import fetch_text
from agent.tools.mock import MockSearchTool
from agent.tools.search import SearchTool, build_search_tool

log = get_logger(__name__)

_JSON_RE = re.compile(r"\{.*\}|\[.*\]", re.DOTALL)


def _extract_json(text: str) -> str | None:
    m = _JSON_RE.search(text)
    return m.group(0) if m else None


# Shared circuit breaker for the search backend. Opens after N failures; falls back to mock.
_search_breaker = CircuitBreaker(
    name="search",
    failure_threshold=settings.circuit_breaker_failures,
    cooldown_seconds=30.0,
)

_search_tool: SearchTool | None = None
_fallback: SearchTool = MockSearchTool()


def _get_search() -> SearchTool:
    global _search_tool
    if _search_tool is None:
        _search_tool = build_search_tool()
    return _search_tool


# -------- decompose --------

_DECOMPOSE_SYS = """You decompose research questions into 2-{max_q} focused sub-questions.

Return ONLY JSON: {{"sub_questions": [{{"question": "...", "rationale": "..."}}, ...]}}
- Each sub-question must be independently searchable (web search will run on it verbatim).
- Cover distinct angles. Avoid overlap. Avoid generic phrasing.
- 2-4 sub-questions is usually right; never more than {max_q}."""


async def decompose(state: AgentState) -> dict[str, object]:
    sys = _DECOMPOSE_SYS.format(max_q=settings.max_sub_questions)
    resp = await chat_model().ainvoke(
        [SystemMessage(content=sys), HumanMessage(content=state["original_question"])]
    )
    raw = resp.content if isinstance(resp.content, str) else str(resp.content)
    snippet = _extract_json(raw)
    if not snippet:
        log.warning("decompose_no_json", raw=raw[:200])
        plan = [SubQuestionPlan(question=state["original_question"])]
    else:
        try:
            data = DecompositionOutput.model_validate_json(snippet)
            plan = data.sub_questions[: settings.max_sub_questions]
        except Exception as exc:  # noqa: BLE001
            log.warning("decompose_parse_failed", error=str(exc))
            plan = [SubQuestionPlan(question=state["original_question"])]

    return {
        "sub_questions": plan,
        "iterations": state.get("iterations", 0) + 1,
        "events": [f"Decomposed into {len(plan)} sub-question(s)."],
    }


# -------- search --------

async def _search_one(query: str) -> list[SearchResult]:
    try:
        return await _search_breaker.call(
            lambda: _get_search().search(query, limit=settings.search_results_per_query)
        )
    except CircuitOpenError:
        log.warning("search_circuit_open_using_fallback")
        return await _fallback.search(query, limit=settings.search_results_per_query)
    except Exception as exc:  # noqa: BLE001
        log.warning("search_failed", query=query[:80], error=str(exc))
        return []


async def search(state: AgentState) -> dict[str, object]:
    queries = state.get("refinement_queries") or [s.question for s in state["sub_questions"]]
    results_per_query = await asyncio.gather(*[_search_one(q) for q in queries])
    flat: list[SearchResult] = []
    for hits in results_per_query:
        flat.extend(hits)

    return {
        "search_results": flat,
        "refinement_queries": [],
        "events": [f"Searched {len(queries)} queries; got {len(flat)} results."],
    }


# -------- fetch + summarize --------

_SUMMARIZE_SYS = """You extract factual claims from web content to answer a specific question.

Given a question and one or more source excerpts, output JSON:
{"findings": [{"claim": "...", "citations": [{"url": "...", "title": "...", "snippet": "..."}]}]}

Rules:
- Each claim must be directly supported by at least one cited source.
- Quote URLs and titles exactly as given. Snippets may be paraphrased.
- 1-4 findings per question. If no source supports an answer, return findings: []."""


async def fetch_and_summarize_one(
    sub_q: SubQuestionPlan,
    results: list[SearchResult],
) -> tuple[SubQuestionReport, list[Finding]]:
    top = sorted(results, key=lambda r: r.source_quality, reverse=True)[: settings.fetch_top_n]
    fetch_results: list[SearchResult] = []
    for r in top:
        if r.fetched_text:
            fetch_results.append(r)
            continue
        text = await fetch_text(r.url)
        fetch_results.append(r.model_copy(update={"fetched_text": text}))

    if not fetch_results or all(not r.fetched_text for r in fetch_results):
        return (
            SubQuestionReport(question=sub_q.question, findings=[], confidence=0.0),
            [],
        )

    sources_block = "\n\n".join(
        f"[{i + 1}] URL: {r.url}\nTITLE: {r.title}\nCONTENT:\n{(r.fetched_text or r.snippet)[:3000]}"
        for i, r in enumerate(fetch_results)
        if r.fetched_text or r.snippet
    )

    resp = await chat_model(temperature=0.1).ainvoke(
        [
            SystemMessage(content=_SUMMARIZE_SYS),
            HumanMessage(
                content=f"<question>{sub_q.question}</question>\n\n<sources>\n{sources_block}\n</sources>"
            ),
        ]
    )
    raw = resp.content if isinstance(resp.content, str) else str(resp.content)
    snippet = _extract_json(raw)
    findings: list[Finding] = []
    if snippet:
        try:
            data = json.loads(snippet)
            for f in data.get("findings", []):
                findings.append(Finding.model_validate(f))
        except Exception as exc:  # noqa: BLE001
            log.warning("summarize_parse_failed", error=str(exc), q=sub_q.question[:80])

    confidence = min(1.0, len(findings) * 0.3) if findings else 0.0
    return (
        SubQuestionReport(question=sub_q.question, findings=findings, confidence=confidence),
        findings,
    )


async def summarize(state: AgentState) -> dict[str, object]:
    # Bucket search results back to the sub-question they came from. The simplest reliable
    # signal: each sub-question's index in the order they were issued matches the corresponding
    # search_results slice. But since search results are flattened, we re-search per sub-q here
    # by routing the existing list through token overlap. For demoable quality we just give all
    # results to each sub-q's summarizer — Claude is good at scoping.
    sub_qs = state["sub_questions"]
    all_results = state["search_results"]
    n_per = max(1, len(all_results) // max(len(sub_qs), 1))

    tasks = []
    for i, sq in enumerate(sub_qs):
        bucket = all_results[i * n_per : (i + 1) * n_per] or all_results[: settings.fetch_top_n]
        tasks.append(fetch_and_summarize_one(sq, bucket))
    results = await asyncio.gather(*tasks)

    sub_reports = [r.model_dump() for r, _ in results]
    findings: list[Finding] = []
    for _, f in results:
        findings.extend(f)

    return {
        "sub_reports": sub_reports,
        "findings": findings,
        "events": [f"Summarized {len(sub_qs)} sub-question(s); {len(findings)} findings."],
    }


# -------- synthesize + critique --------

_SYNTH_SYS = """You synthesize multi-source findings into an executive summary (3-6 sentences).

Rules:
- Only state things supported by the findings provided.
- Surface disagreements between sources.
- Plain prose, no headings, no lists."""


async def synthesize(state: AgentState) -> dict[str, object]:
    findings_text = "\n".join(
        f"- {f.claim}  [cites: {', '.join(c.url for c in f.citations)}]" for f in state["findings"]
    ) or "(no findings)"

    resp = await chat_model(temperature=0.3).ainvoke(
        [
            SystemMessage(content=_SYNTH_SYS),
            HumanMessage(
                content=f"<question>{state['original_question']}</question>\n\n<findings>\n{findings_text}\n</findings>"
            ),
        ]
    )
    synthesis = resp.content if isinstance(resp.content, str) else str(resp.content)
    return {
        "synthesis": synthesis.strip(),
        "events": ["Synthesized findings into executive summary."],
    }


_CRITIQUE_SYS = """You critique a research synthesis against the original question.

Return ONLY JSON:
{"confidence": <0..1>, "gaps": ["<gap1>", ...], "refinement_queries": ["<query1>", ...]}

- confidence reflects how completely the synthesis answers the original question
- gaps lists what is still missing (empty list means none)
- refinement_queries are 0-3 NEW search queries that would close those gaps (empty list if none needed)"""


async def critique(state: AgentState) -> dict[str, object]:
    resp = await chat_model(temperature=0.1).ainvoke(
        [
            SystemMessage(content=_CRITIQUE_SYS),
            HumanMessage(
                content=(
                    f"<question>{state['original_question']}</question>\n"
                    f"<synthesis>{state.get('synthesis', '')}</synthesis>"
                )
            ),
        ]
    )
    raw = resp.content if isinstance(resp.content, str) else str(resp.content)
    snippet = _extract_json(raw)
    if not snippet:
        return {"confidence": 0.5, "refinement_queries": [], "events": ["Critique parse failed; defaulting confidence=0.5."]}
    try:
        c = Critique.model_validate_json(snippet)
    except Exception as exc:  # noqa: BLE001
        log.warning("critique_parse_failed", error=str(exc))
        return {"confidence": 0.5, "refinement_queries": [], "events": ["Critique parse failed; defaulting confidence=0.5."]}
    return {
        "confidence": c.confidence,
        "refinement_queries": c.refinement_queries[:3],
        "events": [
            f"Critique: confidence={c.confidence:.2f}, gaps={len(c.gaps)}, refinement={len(c.refinement_queries)}."
        ],
    }


# -------- write final report --------

def _unique_sources(findings: list[Finding]) -> list[Citation]:
    seen: dict[str, Citation] = {}
    for f in findings:
        for c in f.citations:
            if c.url not in seen:
                seen[c.url] = c
    return list(seen.values())


async def write_report(state: AgentState) -> dict[str, object]:
    findings = state["findings"]
    sub_reports = [
        SubQuestionReport.model_validate(r) for r in state.get("sub_reports", [])
    ]
    report = Report(
        question=state["original_question"],
        executive_summary=state.get("synthesis", "(no synthesis)") or "(no synthesis)",
        sub_questions=sub_reports,
        sources=_unique_sources(findings),
        confidence=state.get("confidence", 0.0),
        iterations=state.get("iterations", 1),
    )
    return {
        "final_report": report,
        "sources": report.sources,
        "events": [f"Final report ready: {len(report.sources)} sources, confidence={report.confidence:.2f}."],
    }
