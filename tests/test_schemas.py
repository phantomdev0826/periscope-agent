from __future__ import annotations

import pytest
from pydantic import ValidationError

from agent.schemas import Citation, Finding, Report, SubQuestionReport


def _c() -> Citation:
    return Citation(url="https://x", title="t", snippet="s")


def test_finding_requires_at_least_one_citation() -> None:
    with pytest.raises(ValidationError):
        Finding(claim="some claim", citations=[])


def test_report_rejects_short_summary() -> None:
    with pytest.raises(ValidationError):
        Report(
            question="q",
            executive_summary="short",
            sub_questions=[],
            sources=[],
            confidence=0.5,
            iterations=1,
        )


def test_confidence_in_range() -> None:
    with pytest.raises(ValidationError):
        SubQuestionReport(
            question="q",
            findings=[Finding(claim="c", citations=[_c()])],
            confidence=1.5,
        )


def test_report_round_trip() -> None:
    r = Report(
        question="q",
        executive_summary="A long enough executive summary to pass validation.",
        sub_questions=[
            SubQuestionReport(
                question="sq",
                findings=[Finding(claim="claim text", citations=[_c()])],
                confidence=0.8,
            )
        ],
        sources=[_c()],
        confidence=0.8,
        iterations=1,
    )
    dumped = r.model_dump_json()
    re_loaded = Report.model_validate_json(dumped)
    assert re_loaded.confidence == 0.8
