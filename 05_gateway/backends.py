"""Backend interface and implementations.

A "backend" is anything that can turn a prompt into a completion. Backends
implement the same three-method protocol so the gateway can treat them
interchangeably.
"""
from __future__ import annotations
import asyncio
import os
from typing import Protocol

import httpx


class Backend(Protocol):
    name: str

    async def generate(
        self, prompt: str, max_tokens: int, temperature: float
    ) -> str: ...

    async def is_healthy(self) -> bool: ...


# ---- mock ----------------------------------------------------------------

class MockBackend:
    """Instant, deterministic backend for local development and tests."""

    name = "mock"

    def __init__(self, delay_s: float = 0.05) -> None:
        self._delay_s = delay_s

    async def generate(self, prompt: str, max_tokens: int, temperature: float) -> str:
        await asyncio.sleep(self._delay_s)
        preview = prompt[:40] + ("..." if len(prompt) > 40 else "")
        return f"[mock response to: {preview}]"

    async def is_healthy(self) -> bool:
        return True


class FailingMockBackend:
    """Always fails. Used to exercise the fallback chain locally."""

    name = "failing-mock"

    def __init__(self, exc: BaseException | None = None) -> None:
        self._exc = exc or httpx.ConnectError("simulated backend down")

    async def generate(self, prompt: str, max_tokens: int, temperature: float) -> str:
        raise self._exc

    async def is_healthy(self) -> bool:
        return False


# ---- Groq (OpenAI-compatible) --------------------------------------------

class GroqBackend:
    """Groq's free-tier API. OpenAI-compatible /chat/completions shape."""

    name = "groq"

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str = "https://api.groq.com/openai/v1",
        timeout_s: float = 30.0,
    ) -> None:
        self._api_key = api_key or os.environ.get("GROQ_API_KEY", "")
        self._model = model or os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")
        self._base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(timeout=timeout_s)

    async def generate(self, prompt: str, max_tokens: int, temperature: float) -> str:
        if not self._api_key:
            raise RuntimeError("GROQ_API_KEY not set")
        resp = await self._client.post(
            f"{self._base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self._model,
                "messages": [{"role": "user", "content": prompt}],
                "max_completion_tokens": max_tokens,
                "temperature": temperature,
                "reasoning_effort": "low",
            },
        )
        resp.raise_for_status()
        data = resp.json()
        msg = data["choices"][0]["message"]
        # reasoning models may return content="" if reasoning consumed
        # the whole budget; surface reasoning as fallback text
        content = msg.get("content") or msg.get("reasoning") or ""
        return content

    async def is_healthy(self) -> bool:
        if not self._api_key:
            return False
        try:
            r = await self._client.get(
                f"{self._base_url}/models",
                headers={"Authorization": f"Bearer {self._api_key}"},
                timeout=5.0,
            )
            return r.status_code == 200
        except Exception:
            return False


# ---- vLLM (local OpenAI-compatible server) --------------------------------

class VLLMBackend:
    """Points at a local vLLM server. Same OpenAI-compatible shape as Groq."""

    name = "vllm"

    def __init__(
        self,
        base_url: str = "http://localhost:8001",
        model: str = "Qwen/Qwen2.5-0.5B-Instruct",
        timeout_s: float = 60.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._client = httpx.AsyncClient(timeout=timeout_s)

    async def generate(self, prompt: str, max_tokens: int, temperature: float) -> str:
        resp = await self._client.post(
            f"{self._base_url}/v1/chat/completions",
            json={
                "model": self._model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    async def is_healthy(self) -> bool:
        try:
            r = await self._client.get(f"{self._base_url}/health", timeout=2.0)
            return r.status_code == 200
        except Exception:
            return False