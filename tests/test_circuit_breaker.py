from __future__ import annotations

import asyncio

import pytest

from agent.circuit_breaker import CircuitBreaker, CircuitOpenError


async def _ok() -> str:
    return "ok"


async def _fail() -> str:
    raise RuntimeError("boom")


async def test_opens_after_threshold() -> None:
    cb = CircuitBreaker(name="t", failure_threshold=2, cooldown_seconds=0.1)
    with pytest.raises(RuntimeError):
        await cb.call(_fail)
    with pytest.raises(RuntimeError):
        await cb.call(_fail)
    with pytest.raises(CircuitOpenError):
        await cb.call(_fail)


async def test_success_resets_counter() -> None:
    cb = CircuitBreaker(name="t", failure_threshold=2, cooldown_seconds=0.1)
    with pytest.raises(RuntimeError):
        await cb.call(_fail)
    assert await cb.call(_ok) == "ok"
    # After a success the counter resets, so two more failures are needed to open.
    with pytest.raises(RuntimeError):
        await cb.call(_fail)
    assert await cb.call(_ok) == "ok"  # not yet open


async def test_recovers_after_cooldown() -> None:
    cb = CircuitBreaker(name="t", failure_threshold=1, cooldown_seconds=0.05)
    with pytest.raises(RuntimeError):
        await cb.call(_fail)
    with pytest.raises(CircuitOpenError):
        await cb.call(_ok)
    await asyncio.sleep(0.07)
    assert await cb.call(_ok) == "ok"
