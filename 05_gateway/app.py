"""FastAPI gateway in front of an LLM serving backend chain.

Session 3: Prometheus metrics at /metrics.
"""
from __future__ import annotations
from contextlib import asynccontextmanager
import os
import time

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from backends import Backend, GroqBackend, MockBackend
from metrics import (
    BACKENDS_HEALTHY,
    BACKEND_ATTEMPTS_TOTAL,
    REGISTRY,
    REQUESTS_TOTAL,
    REQUEST_LATENCY_SECONDS,
)
from retry import with_retry

load_dotenv()


# ---- schemas -------------------------------------------------------------

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

def _build_backends() -> list[Backend]:
    requested = os.environ.get("GATEWAY_BACKENDS", "").strip()
    names = [n.strip() for n in requested.split(",") if n.strip()] if requested \
        else ["groq", "mock"]
    built: list[Backend] = []
    for n in names:
        if n == "groq":
            built.append(GroqBackend())
        elif n == "mock":
            built.append(MockBackend())
        else:
            raise ValueError(f"unknown backend: {n}")
    return built


_backends: list[Backend] = _build_backends()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # populate health gauges at startup so /metrics has data from the
    # first scrape, before any client has hit /health or /generate
    for b in _backends:
        h = await b.is_healthy()
        BACKENDS_HEALTHY.labels(backend=b.name).set(1 if h else 0)
    yield


app = FastAPI(
    title="LLM Serving Lab Gateway",
    version="0.1.0",
    lifespan=lifespan,
)


# ---- routes --------------------------------------------------------------

@app.get("/health")
async def health() -> dict:
    statuses = []
    for b in _backends:
        healthy = await b.is_healthy()
        BACKENDS_HEALTHY.labels(backend=b.name).set(1 if healthy else 0)
        statuses.append({"name": b.name, "healthy": healthy})
    any_healthy = any(s["healthy"] for s in statuses)
    return {"status": "ok" if any_healthy else "degraded", "backends": statuses}


@app.get("/backends", response_model=list[BackendStatus])
async def list_backends() -> list[BackendStatus]:
    out = []
    for b in _backends:
        healthy = await b.is_healthy()
        BACKENDS_HEALTHY.labels(backend=b.name).set(1 if healthy else 0)
        out.append(BackendStatus(name=b.name, healthy=healthy))
    return out


@app.get("/metrics")
async def metrics() -> Response:
    return Response(content=generate_latest(REGISTRY),
                    media_type=CONTENT_TYPE_LATEST)


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest) -> GenerateResponse:
    t0 = time.perf_counter()
    errors: list[str] = []
    total_attempts = 0

    for backend in _backends:
        healthy = await backend.is_healthy()
        BACKENDS_HEALTHY.labels(backend=backend.name).set(1 if healthy else 0)
        if not healthy:
            errors.append(f"{backend.name}: unhealthy")
            continue

        attempts_here = 0

        async def call(b=backend):
            nonlocal attempts_here
            attempts_here += 1
            try:
                result = await b.generate(req.prompt, req.max_tokens, req.temperature)
                BACKEND_ATTEMPTS_TOTAL.labels(
                    backend=b.name, outcome="success"
                ).inc()
                return result
            except Exception:
                BACKEND_ATTEMPTS_TOTAL.labels(
                    backend=b.name, outcome="failure"
                ).inc()
                raise

        try:
            text = await with_retry(call, max_attempts=3)
        except Exception as e:
            errors.append(f"{backend.name}: {type(e).__name__}: {e}")
            total_attempts += attempts_here
            continue

        total_attempts += attempts_here
        latency_s = time.perf_counter() - t0
        REQUESTS_TOTAL.labels(backend=backend.name, status="success").inc()
        REQUEST_LATENCY_SECONDS.labels(backend=backend.name).observe(latency_s)
        return GenerateResponse(
            text=text,
            backend=backend.name,
            latency_ms=latency_s * 1000.0,
            attempts=total_attempts,
        )

    REQUESTS_TOTAL.labels(backend="none", status="error").inc()
    raise HTTPException(
        status_code=503,
        detail={"message": "all backends failed", "errors": errors},
    )