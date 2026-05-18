from __future__ import annotations

from pydantic import BaseModel, Field, HttpUrl, field_validator


class SearchResult(BaseModel):
    url: str  # not HttpUrl: mock fixtures use local:// scheme
    title: str
    snippet: str
    source_quality: float = Field(default=0.5, ge=0.0, le=1.0)
    fetched_text: str | None = None


class SubQuestionPlan(BaseModel):
    question: str = Field(min_length=4, max_length=400)
    rationale: str | None = None


class DecompositionOutput(BaseModel):
    sub_questions: list[SubQuestionPlan] = Field(min_length=1, max_length=10)


class Citation(BaseModel):
    url: str
    title: str
    snippet: str

    @field_validator("url")
    @classmethod
    def _strip(cls, v: str) -> str:
        return v.strip()


class Finding(BaseModel):
    claim: str = Field(min_length=4)
    citations: list[Citation] = Field(min_length=1)


class SubQuestionReport(BaseModel):
    question: str
    findings: list[Finding]
    confidence: float = Field(ge=0.0, le=1.0)


class Report(BaseModel):
    question: str
    executive_summary: str = Field(min_length=20)
    sub_questions: list[SubQuestionReport]
    sources: list[Citation]
    confidence: float = Field(ge=0.0, le=1.0)
    iterations: int = Field(ge=1)


class Critique(BaseModel):
    confidence: float = Field(ge=0.0, le=1.0)
    gaps: list[str] = Field(default_factory=list)
    refinement_queries: list[str] = Field(default_factory=list)
