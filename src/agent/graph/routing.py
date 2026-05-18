from __future__ import annotations

from agent.core.config import settings
from agent.graph.state import AgentState


def after_critique(state: AgentState) -> str:
    """Conditional edge: route back to search if confidence is low AND we still have iterations."""
    confidence = state.get("confidence", 0.0)
    iterations = state.get("iterations", 1)
    refinements = state.get("refinement_queries") or []

    if (
        confidence < settings.confidence_threshold
        and iterations < settings.max_iterations
        and refinements
    ):
        return "refine"
    return "finalize"
