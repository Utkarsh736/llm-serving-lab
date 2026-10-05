"""Concurrent load test for the gateway.

Fires N requests at /generate, reports throughput and latency percentiles.
Run against the gateway on localhost:8000. Uses asyncio + httpx.

    uv run 05_gateway/load_test.py --n 50 --concurrency 10
    uv run 05_gateway/load_test.py --n 20 --concurrency 5 --prompt "Say hi"
"""
from __future__ import annotations
import argparse
import asyncio
import csv
import statistics
import time
from pathlib import Path

import httpx

HERE = Path(__file__).parent


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * p / 100.0
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


async def one_request(client: httpx.AsyncClient, url: str, prompt: str,
                      max_tokens: int) -> dict:
    t0 = time.perf_counter()
    try:
        r = await client.post(url, json={
            "prompt": prompt, "max_tokens": max_tokens,
        })
        latency = time.perf_counter() - t0
        if r.status_code != 200:
            return {"ok": False, "latency": latency, "backend": None,
                    "attempts": None, "status": r.status_code}
        data = r.json()
        return {"ok": True, "latency": latency, "backend": data["backend"],
                "attempts": data["attempts"], "status": 200}
    except Exception as e:
        return {"ok": False, "latency": time.perf_counter() - t0,
                "backend": None, "attempts": None, "status": type(e).__name__}


async def run(n: int, concurrency: int, url: str, prompt: str,
              max_tokens: int) -> list[dict]:
    sem = asyncio.Semaphore(concurrency)

    async def worker(client):
        async with sem:
            return await one_request(client, url, prompt, max_tokens)

    async with httpx.AsyncClient(timeout=60.0) as client:
        t0 = time.perf_counter()
        results = await asyncio.gather(*[worker(client) for _ in range(n)])
        wall = time.perf_counter() - t0

    print(f"\n=== load test: n={n}, concurrency={concurrency} ===")
    print(f"wall time:  {wall:.2f} s")
    print(f"throughput: {n / wall:.2f} req/s")

    ok = [r for r in results if r["ok"]]
    latencies = [r["latency"] * 1000.0 for r in ok]
    if latencies:
        print(f"latency P50: {percentile(latencies, 50):.1f} ms")
        print(f"latency P95: {percentile(latencies, 95):.1f} ms")
        print(f"latency P99: {percentile(latencies, 99):.1f} ms")

    n_err = len(results) - len(ok)
    print(f"successes:   {len(ok)} / {n}")
    print(f"errors:      {n_err}")

    backends = {}
    for r in ok:
        backends[r["backend"]] = backends.get(r["backend"], 0) + 1
    if backends:
        print(f"backends:    {backends}")

    attempts = [r["attempts"] for r in ok if r["attempts"] is not None]
    if attempts:
        print(f"attempts:    mean={statistics.mean(attempts):.2f}  "
              f"max={max(attempts)}")

    return results


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--concurrency", type=int, default=10)
    p.add_argument("--url", default="http://localhost:8000/generate")
    p.add_argument("--prompt", default="Say hi.")
    p.add_argument("--max-tokens", type=int, default=32)
    p.add_argument("--csv", default=str(HERE / "load_test_results.csv"))
    args = p.parse_args()

    results = asyncio.run(run(
        args.n, args.concurrency, args.url, args.prompt, args.max_tokens,
    ))

    csv_path = Path(args.csv)
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["ok", "latency", "backend",
                                          "attempts", "status"])
        w.writeheader()
        w.writerows(results)
    print(f"\nwrote {csv_path}")


if __name__ == "__main__":
    main()
