"""Sweep block_size and report the tradeoff.

Small blocks: less internal fragmentation, bigger page tables.
Large blocks: more internal fragmentation, smaller page tables.

Goal: find the sweet spot. Real vLLM defaults to block_size=16.
"""
from __future__ import annotations
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fragmentation import make_workload, paged_usage


BLOCK_SIZES = [4, 8, 16, 32, 64, 128]
POOL_SLOTS = 65_536          # fixed pool size in token slots (4096 blocks × 16)
OUT_DIR = Path(__file__).parent

# bytes per KV token slot: 2 (K and V) × 6 layers × 6 heads × 64 head_dim × 4 bytes (fp32)
BYTES_PER_SLOT = 2 * 6 * 6 * 64 * 4


def main() -> None:
    workload = make_workload(n=500, seed=42)
    rows = []

    for bs in BLOCK_SIZES:
        num_blocks = POOL_SLOTS // bs
        peak_b, peak_s, waste = paged_usage(workload, num_blocks, bs)

        avg_blocks_per_seq = peak_b / max(peak_s, 1)
        page_table_bytes = peak_s * avg_blocks_per_seq * 4

        alloc_slots = peak_b * bs
        frag_pct = (waste / alloc_slots * 100) if alloc_slots else 0.0

        # total memory = wasted slots (KV bytes) + page-table metadata
        waste_bytes = waste * BYTES_PER_SLOT
        total_bytes = waste_bytes + page_table_bytes

        rows.append({
            "block_size": bs,
            "peak_seqs": peak_s,
            "peak_blocks": peak_b,
            "waste_slots": waste,
            "fragmentation_pct": frag_pct,
            "page_table_bytes": page_table_bytes,
            "total_overhead_bytes": total_bytes,
        })
        print(f"bs={bs:>3}  seqs={peak_s:>4}  blocks={peak_b:>5}  "
              f"frag={frag_pct:5.2f}%  ptable={page_table_bytes/1024:6.1f} KB  "
              f"waste={waste_bytes/1024:7.1f} KB  "
              f"total={total_bytes/1024:7.1f} KB")

    csv_path = OUT_DIR / "results.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {csv_path}")

    # chart: three curves — fragmentation %, page table KB, total overhead KB
    fig, ax1 = plt.subplots(figsize=(8, 5))
    xs = [r["block_size"] for r in rows]

    ax1.plot(xs, [r["fragmentation_pct"] for r in rows], "o-",
             color="tab:red", label="internal fragmentation %")
    ax1.set_xlabel("block size (tokens)")
    ax1.set_ylabel("internal fragmentation (%)", color="tab:red")
    ax1.tick_params(axis="y", labelcolor="tab:red")
    ax1.set_xscale("log", base=2)
    ax1.grid(True, which="both", alpha=0.3)
    ax1.axvline(16, color="gray", linestyle="--", alpha=0.6,
            label="vLLM default (16)")

    ax2 = ax1.twinx()
    ax2.plot(xs, [r["page_table_bytes"] / 1024 for r in rows], "s-",
             color="tab:blue", label="page table (KB)")
    ax2.plot(xs, [r["total_overhead_bytes"] / 1024 for r in rows], "^-",
             color="tab:green", label="total overhead (KB)")
    ax2.set_ylabel("bytes (KB)", color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:blue")

    # mark the minimum of total overhead
    min_row = min(rows, key=lambda r: r["total_overhead_bytes"])
    print(f"\nmemory-minimizing block size: {min_row['block_size']}")
    print("note: this model captures waste + page-table bytes only. "
          "Real engines pick bs=16 because smaller blocks hurt "
          "kernel memory throughput — not captured here.")
    ax2.axvline(min_row["block_size"], color="tab:green", linestyle=":",
                alpha=0.5, label=f"min @ bs={min_row['block_size']}")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    fig.suptitle("Block size tradeoff: fragmentation vs total memory overhead")
    fig.tight_layout()
    png = OUT_DIR / "tradeoff.png"
    fig.savefig(png, dpi=120)
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
