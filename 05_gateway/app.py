"""FastAPI gateway in front of an LLM serving backend chain.

Session 2: fallback chain with per-backend retry.
Session 3 will add Prometheus metrics.
"""
from __future__ import annotations
import os
import time

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from backends import Backend, GroqBackend, MockBackend
from retry import with_retry

load_dotenv()


# ---- request/response schemas -------------------------------------------

class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    max_tokens: int = Field(128, ge=1, le=4096)
    temperature: float = Field(0.7, ge=0.0, le=2.0)


class GenerateResponse(BaseModel):
    text: str
    backend: str
    latency_ms: float
    attempts: int


class BackendStatus(BaseModel):
    name: str
    healthy: bool


# ---- backend chain -------------------------------------------------------
#
# Order matters: earlier entries are preferred. The gateway tries each in
# order, retrying a few times on transient failures before moving on.
#
# For local dev, set GATEWAY_BACKENDS=mock or omit GROQ_API_KEY.

def _build_backends() -> list[Backend]:
    requested = os.environ.get("GATEWAY_BACKENDS", "").strip()
    if requested:
        names = [n.strip() for n in requested.split(",") if n.strip()]
    else:
        names = ["groq", "mock"]  # default: prefer real backend, mock as backup

    built: list[Backend] = []
    for n in names:
        if n == "groq":
            built.append(GroqBackend())
        elif n == "mock":
            built.append(MockBackend())
        else:
            raise ValueError(f"unknown backend: {n}")
    return built


app = FastAPI(title="LLM Serving Lab Gateway", version="0.1.0")
_backends: list[Backend] = _build_backends()


# ---- routes --------------------------------------------------------------

@app.get("/health")
async def health() -> dict:
    statuses = []
    for b in _backends:
        statuses.append({"name": b.name, "healthy": await b.is_healthy()})
    any_healthy = any(s["healthy"] for s in statuses)
    return {
        "status": "ok" if any_healthy else "degraded",
        "backends": statuses,
    }


@app.get("/backends", response_model=list[BackendStatus])
async def list_backends() -> list[BackendStatus]:
    return [
        BackendStatus(name=b.name, healthy=await b.is_healthy())
        for b in _backends
    ]


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest) -> GenerateResponse:
    t0 = time.perf_counter()
    errors: list[str] = []
    total_attempts = 0

    for backend in _backends:
        if not await backend.is_healthy():
            errors.append(f"{backend.name}: unhealthy")
            continue

        attempts_here = 0

        async def call(b=backend):
            nonlocal attempts_here
            attempts_here += 1
            return await b.generate(req.prompt, req.max_tokens, req.temperature)

        try:
            text = await with_retry(call, max_attempts=3)
        except Exception as e:
            errors.append(f"{backend.name}: {type(e).__name__}: {e}")
            total_attempts += attempts_here
            continue

        total_attempts += attempts_here
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return GenerateResponse(
            text=text,
            backend=backend.name,
            latency_ms=latency_ms,
            attempts=total_attempts,
        )

    raise HTTPException(status_code=503, detail={"message": "all backends failed", "errors": errors})