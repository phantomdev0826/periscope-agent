from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from agent.schemas import Citation, Finding, Report, SearchResult, SubQuestionPlan


class AgentState(TypedDict, total=False):
    """LangGraph state. Fields written by multiple nodes use Annotated[..., operator.add]
    so LangGraph merges them across parallel branches; everything else is overwrite-on-write."""

    original_question: str
    sub_questions: list[SubQuestionPlan]
    search_results: Annotated[list[SearchResult], operator.add]
    findings: Annotated[list[Finding], operator.add]
    sub_reports: list[dict[str, object]]  # populated by summarize node
    synthesis: str
    sources: list[Citation]
    confidence: float
    iterations: int
    refinement_queries: list[str]
    final_report: Report | None
    events: Annotated[list[str], operator.add]  # human-readable trace for the UI
