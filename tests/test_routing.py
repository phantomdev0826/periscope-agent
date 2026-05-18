from __future__ import annotations

from agent.core.config import settings
from agent.graph.routing import after_critique


def test_refine_when_low_confidence_with_iterations_left() -> None:
    state = {
        "confidence": 0.2,
        "iterations": 1,
        "refinement_queries": ["follow up query"],
    }
    assert after_critique(state) == "refine"  # type: ignore[arg-type]


def test_finalize_at_high_confidence() -> None:
    state = {
        "confidence": settings.confidence_threshold + 0.1,
        "iterations": 1,
        "refinement_queries": ["x"],
    }
    assert after_critique(state) == "finalize"  # type: ignore[arg-type]


def test_finalize_when_iterations_exhausted() -> None:
    state = {
        "confidence": 0.1,
        "iterations": settings.max_iterations,
        "refinement_queries": ["x"],
    }
    assert after_critique(state) == "finalize"  # type: ignore[arg-type]


def test_finalize_when_no_refinements_proposed() -> None:
    state = {
        "confidence": 0.1,
        "iterations": 1,
        "refinement_queries": [],
    }
    assert after_critique(state) == "finalize"  # type: ignore[arg-type]
