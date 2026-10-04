"""Aggregate vLLM sweep JSONs into a single CSV + charts.

Schema matches vLLM 0.30.0's `benchmark_serving.py` output:
- Aggregates: median_* (= P50), p99_*, mean_*, std_* for ttft / itl / tpot
- Arrays: `latencies` (end-to-end, seconds), `queue_times` (seconds)
- Throughput: request_throughput, output_throughput

Note: vLLM 0.30.0 does NOT emit P95. We report P50 and P99.

Input:  results/*.json
Output: results.csv, throughput.png, ttft_p99.png, latency_p99.png
"""
from __future__ import annotations
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
RESULTS = HERE / "results"
OUT_CSV = HERE / "results.csv"


def percentile(values: list[float], p: float) -> float:
    """Linear-interpolation percentile. `p` in [0, 100]."""
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * p / 100.0
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def load_runs() -> list[dict]:
    rows = []
    for path in sorted(RESULTS.glob("*.json")):
        with path.open() as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                continue

        # per-request end-to-end latencies, in seconds → ms
        latencies_s = data.get("latencies") or []
        latencies_ms = [x * 1000.0 for x in latencies_s]

        rows.append({
            "run": path.stem,
            # throughput
            "request_throughput_rps": data.get("request_throughput"),
            "output_throughput_tps":  data.get("output_throughput"),
            "total_throughput_tps":   data.get("total_token_throughput"),
            # TTFT (P50 = median, P99 = tail)
            "ttft_p50_ms":  data.get("median_ttft_ms"),
            "ttft_p99_ms":  data.get("p99_ttft_ms"),
            "ttft_mean_ms": data.get("mean_ttft_ms"),
            # ITL
            "itl_p50_ms":  data.get("median_itl_ms"),
            "itl_p99_ms":  data.get("p99_itl_ms"),
            "itl_mean_ms": data.get("mean_itl_ms"),
            # TPOT
            "tpot_p50_ms":  data.get("median_tpot_ms"),
            "tpot_p99_ms":  data.get("p99_tpot_ms"),
            # end-to-end latency (computed from raw array)
            "e2e_p50_ms": percentile(latencies_ms, 50),
            "e2e_p95_ms": percentile(latencies_ms, 95),
            "e2e_p99_ms": percentile(latencies_ms, 99),
            # metadata
            "duration_s":      data.get("duration"),
            "completed":       data.get("completed"),
            "failed":          data.get("failed"),
            "max_concurrency": data.get("max_concurrency"),
        })
    return rows


def parse_run_name(name: str) -> tuple[str, int | None]:
    """'fp16_mns8_c16' -> ('fp16_mns8', 16)."""
    parts = name.rsplit("_c", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0], int(parts[1])
    return name, None


def main() -> None:
    rows = load_runs()
    if not rows:
        print(f"no JSON files found in {RESULTS}")
        return

    for r in rows:
        cfg, conc = parse_run_name(r["run"])
        r["config"] = cfg
        r["concurrency"] = conc

    fieldnames = ["run", "config", "concurrency"] + [
        k for k in rows[0].keys() if k not in ("run", "config", "concurrency")
    ]
    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {OUT_CSV} ({len(rows)} rows)")

    configs = sorted({
    r["config"] for r in rows
    if (r.get("completed") or 0) > 0
    })
    colors = plt.cm.tab10.colors

    def plot(metric_key: str, ylabel: str, title: str, filename: str) -> None:
        fig, ax = plt.subplots(figsize=(8, 5))
        for i, cfg in enumerate(configs):
            sub = sorted(
                [r for r in rows
                 if r["config"] == cfg and r["concurrency"] is not None],
                key=lambda r: r["concurrency"],
            )
            xs = [r["concurrency"] for r in sub]
            ys = [r[metric_key] for r in sub]
            ax.plot(xs, ys, "o-", color=colors[i % len(colors)], label=cfg)
        ax.set_xlabel("max concurrency")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xscale("log", base=2)
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        out = HERE / filename
        fig.savefig(out, dpi=120)
        print(f"wrote {out}")

    plot("output_throughput_tps",
         "throughput (output tokens/sec)",
         "vLLM: output throughput vs concurrency",
         "throughput.png")

    plot("ttft_p99_ms",
         "TTFT P99 (ms)",
         "vLLM: tail TTFT vs concurrency",
         "ttft_p99.png")

    plot("e2e_p99_ms",
         "end-to-end P99 latency (ms)",
         "vLLM: tail end-to-end latency vs concurrency",
         "latency_p99.png")


if __name__ == "__main__":
    main()