"""Retry with exponential backoff and jitter.

Retryable errors:
    - network/timeout issues (httpx.TimeoutException, httpx.ConnectError)
    - HTTP 429 (rate limit)
    - HTTP 5xx (server errors)

Non-retryable:
    - HTTP 4xx other than 429 (client errors — retrying won't help)
    - any other exception (likely a bug, surface it)

Backoff: delay = base_delay * (2 ** attempt) + jitter, capped.
Jitter prevents synchronized retry storms across clients.
"""
from __future__ import annotations
import asyncio
import random
from typing import Awaitable, Callable, TypeVar

import httpx

T = TypeVar("T")


def is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, (httpx.TimeoutException, httpx.ConnectError)):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        return code == 429 or 500 <= code < 600
    return False


async def with_retry(
    fn: Callable[[], Awaitable[T]],
    max_attempts: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 4.0,
) -> T:
    last_exc: BaseException | None = None
    for attempt in range(max_attempts):
        try:
            return await fn()
        except BaseException as e:
            if not is_retryable(e):
                raise
            last_exc = e
            if attempt == max_attempts - 1:
                break
            delay = min(base_delay * (2 ** attempt), max_delay)
            delay += random.uniform(0, delay * 0.1)
            await asyncio.sleep(delay)
    assert last_exc is not None
    raise last_exc
