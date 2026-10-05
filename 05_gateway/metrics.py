"""Prometheus metrics for the gateway.

Uses a custom CollectorRegistry (not the default global one) so that
uvicorn's --reload doesn't raise "Duplicated timeseries" on re-import.

Metrics exported:
    gateway_requests_total{backend,status}    counter, /generate outcomes
    gateway_request_latency_seconds{backend}  histogram, end-to-end latency
    gateway_backend_attempts_total{backend,outcome}  counter, per-backend attempts
    gateway_backends_healthy{backend}         gauge, 1 or 0

Multiprocess caveat: with `uvicorn --workers N`, each worker holds its own
registry. Metrics do not aggregate. Production uses prometheus_client's
multiprocess mode; out of scope here.
"""
from __future__ import annotations

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
)

REGISTRY = CollectorRegistry()

REQUESTS_TOTAL = Counter(
    "gateway_requests_total",
    "Total /generate requests, by serving backend and status.",
    ["backend", "status"],  # status in {success, error}
    registry=REGISTRY,
)

REQUEST_LATENCY_SECONDS = Histogram(
    "gateway_request_latency_seconds",
    "End-to-end /generate latency in seconds.",
    ["backend"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0),
    registry=REGISTRY,
)

BACKEND_ATTEMPTS_TOTAL = Counter(
    "gateway_backend_attempts_total",
    "Total attempts per backend, by outcome.",
    ["backend", "outcome"],  # outcome in {success, failure}
    registry=REGISTRY,
)

BACKENDS_HEALTHY = Gauge(
    "gateway_backends_healthy",
    "1 if the backend passed its health check, 0 otherwise.",
    ["backend"],
    registry=REGISTRY,
)
