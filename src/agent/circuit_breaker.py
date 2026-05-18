from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TypeVar

from agent.core.logging import get_logger

log = get_logger(__name__)

T = TypeVar("T")


class CircuitOpenError(RuntimeError):
    pass


@dataclass
class CircuitBreaker:
    """Simple circuit breaker: opens after `failure_threshold` consecutive failures,
    auto-recovers after `cooldown_seconds`. Designed for wrapping external API calls
    inside async LangGraph nodes; counts only failures since the last success."""

    name: str
    failure_threshold: int = 3
    cooldown_seconds: float = 30.0
    _failures: int = 0
    _opened_at: float | None = None

    @property
    def is_open(self) -> bool:
        if self._opened_at is None:
            return False
        if time.monotonic() - self._opened_at >= self.cooldown_seconds:
            log.info("circuit_half_open", name=self.name)
            self._opened_at = None
            self._failures = 0
            return False
        return True

    async def call(self, fn: Callable[[], Awaitable[T]]) -> T:
        if self.is_open:
            raise CircuitOpenError(f"circuit '{self.name}' is open")
        try:
            result = await fn()
        except Exception:
            self._failures += 1
            if self._failures >= self.failure_threshold:
                self._opened_at = time.monotonic()
                log.warning(
                    "circuit_opened",
                    name=self.name,
                    failures=self._failures,
                    cooldown=self.cooldown_seconds,
                )
            raise
        self._failures = 0
        return result
