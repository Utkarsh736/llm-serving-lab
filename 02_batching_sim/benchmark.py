"""Sweep policies × workload sizes, collect metrics, produce CSV + charts.

Workloads: 10, 50, 100 concurrent users.
Policies:  naive_batching, continuous_batching.
Metrics:   throughput, TTFT P50/P95, ITL P50/P95, end-to-end latency P50/P95.
"""
from __future__ import annotations
import csv
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from simulator import CostModel, clone_requests
from policies import naive_batching, continuous_batching
from workloads import generate_workload


USER_COUNTS = [10, 50, 100]
MAX_BATCH_SIZE = 8
OUT_DIR = Path(__file__).parent


def percentile(values: list[float], p: float) -> float:
    """Simple percentile (linear interpolation)."""
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * p / 100.0
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def summarize(result) -> dict:
    ttfts = [r.ttft for r in result.requests if r.ttft is not None]
    itls = [r.itl for r in result.requests if r.itl is not None]
    lats = [r.latency for r in result.requests if r.latency is not None]
    return {
        "policy": result.policy,
        "throughput_rps": result.throughput_rps,
        "total_time_ms": result.total_time_ms,
        "decode_steps": result.decode_steps,
        "ttft_p50_ms": percentile(ttfts, 50),
        "ttft_p95_ms": percentile(ttfts, 95),
        "itl_p50_ms": percentile(itls, 50),
        "itl_p95_ms": percentile(itls, 95),
        "latency_p50_ms": percentile(lats, 50),
        "latency_p95_ms": percentile(lats, 95),
    }


def main() -> None:
    cost = CostModel()
    rows: list[dict] = []

    for n_users in USER_COUNTS:
        workload = generate_workload(num_users=n_users, seed=42)

        for policy_fn in (naive_batching, continuous_batching):
            result = policy_fn(clone_requests(workload), cost, MAX_BATCH_SIZE)
            row = summarize(result)
            row["users"] = n_users
            rows.append(row)
            print(f"users={n_users:>4}  policy={row['policy']:<22}  "
                  f"throughput={row['throughput_rps']:6.2f}  "
                  f"TTFT_p95={row['ttft_p95_ms']:7.1f}  "
                  f"ITL_p95={row['itl_p95_ms']:6.2f}  "
                  f"lat_p95={row['latency_p95_ms']:8.1f}")

    csv_path = OUT_DIR / "results.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {csv_path}")

    # Chart 1: throughput vs users
    fig, ax = plt.subplots(figsize=(7, 5))
    for policy in ("naive_batching", "continuous_batching"):
        xs = [r["users"] for r in rows if r["policy"] == policy]
        ys = [r["throughput_rps"] for r in rows if r["policy"] == policy]
        ax.plot(xs, ys, "o-", label=policy)
    ax.set_xlabel("concurrent users")
    ax.set_ylabel("throughput (req/s)")
    ax.set_title("Throughput vs concurrency")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "throughput.png", dpi=120)
    print(f"wrote {OUT_DIR / 'throughput.png'}")

    # Chart 2: P95 latency vs users
    fig, ax = plt.subplots(figsize=(7, 5))
    for policy in ("naive_batching", "continuous_batching"):
        xs = [r["users"] for r in rows if r["policy"] == policy]
        ys = [r["latency_p95_ms"] for r in rows if r["policy"] == policy]
        ax.plot(xs, ys, "o-", label=policy)
    ax.set_xlabel("concurrent users")
    ax.set_ylabel("P95 end-to-end latency (ms)")
    ax.set_title("P95 latency vs concurrency")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "latency_p95.png", dpi=120)
    print(f"wrote {OUT_DIR / 'latency_p95.png'}")


if __name__ == "__main__":
    main()
