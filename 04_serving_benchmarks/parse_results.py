"""Aggregate vLLM sweep JSONs into a single CSV + charts.

Input:  results/*.json    (one per (config, concurrency) run)
Output: results.csv       (one row per run)
        throughput.png    (output tokens/sec vs concurrency)
        ttft_p95.png      (tail TTFT vs concurrency)
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


def load_runs() -> list[dict]:
    rows = []
    for path in sorted(RESULTS.glob("*.json")):
        if path.name.endswith("_server.log"):
            continue
        with path.open() as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                continue

        # vLLM's benchmark_serving output fields (names vary by version).
        # We defensively read what's available.
        rows.append({
            "run": path.stem,
            "throughput_rps":     data.get("request_throughput"),
            "throughput_tps":     data.get("output_throughput"),
            "ttft_p50_ms":        data.get("median_ttft_ms"),
            "ttft_p95_ms":        data.get("p95_ttft_ms"),
            "itl_p50_ms":         data.get("median_itl_ms"),
            "itl_p95_ms":         data.get("p95_itl_ms"),
            "tpot_p50_ms":        data.get("median_tpot_ms"),
            "tpot_p95_ms":        data.get("p95_tpot_ms"),
            "e2e_p50_ms":         data.get("median_e2el_ms"),
            "e2e_p95_ms":         data.get("p95_e2el_ms"),
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

    configs = sorted({r["config"] for r in rows})
    colors = plt.cm.tab10.colors

    # Chart 1: throughput vs concurrency
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, cfg in enumerate(configs):
        sub = sorted(
            [r for r in rows if r["config"] == cfg and r["concurrency"] is not None],
            key=lambda r: r["concurrency"],
        )
        xs = [r["concurrency"] for r in sub]
        ys = [r["throughput_tps"] for r in sub]
        ax.plot(xs, ys, "o-", color=colors[i % len(colors)], label=cfg)
    ax.set_xlabel("max concurrency")
    ax.set_ylabel("throughput (output tokens/sec)")
    ax.set_title("vLLM: output throughput vs concurrency")
    ax.set_xscale("log", base=2)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(HERE / "throughput.png", dpi=120)
    print(f"wrote {HERE / 'throughput.png'}")

    # Chart 2: P95 TTFT vs concurrency
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, cfg in enumerate(configs):
        sub = sorted(
            [r for r in rows if r["config"] == cfg and r["concurrency"] is not None],
            key=lambda r: r["concurrency"],
        )
        xs = [r["concurrency"] for r in sub]
        ys = [r["ttft_p95_ms"] for r in sub]
        ax.plot(xs, ys, "o-", color=colors[i % len(colors)], label=cfg)
    ax.set_xlabel("max concurrency")
    ax.set_ylabel("TTFT P95 (ms)")
    ax.set_title("vLLM: tail TTFT vs concurrency")
    ax.set_xscale("log", base=2)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(HERE / "ttft_p95.png", dpi=120)
    print(f"wrote {HERE / 'ttft_p95.png'}")


if __name__ == "__main__":
    main()
