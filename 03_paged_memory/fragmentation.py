"""Compare paged allocation against two baselines:

1. Naive contiguous pre-allocation: reserve max_seq_len per sequence.
   Wastes (max_seq_len - actual_len) slots per sequence.
2. Expected-length reservation: reserve mean output length per sequence.
   Wastes less on short requests but fails (OOMs) when a request exceeds
   its reservation — the same failure mode as real pre-allocation.

Paged allocation: allocate ceil(tokens / block_size) blocks; waste is
bounded by (block_size - 1) per sequence.
"""
from __future__ import annotations
from dataclasses import dataclass

from block_manager import BlockManager


MAX_SEQ_LEN = 2048
BLOCK_SIZE = 16


@dataclass
class Sequence:
    id: int
    prompt_len: int
    output_len: int


def make_workload(n: int, seed: int = 0) -> list[Sequence]:
    import random, math
    rng = random.Random(seed)
    seqs = []
    for i in range(n):
        p = max(1, int(rng.lognormvariate(math.log(60), 0.7)))
        o = max(1, int(rng.lognormvariate(math.log(100), 0.8)))
        seqs.append(Sequence(id=i, prompt_len=p, output_len=o))
    return seqs


# ---- three allocation strategies ---------------------------------------

def paged_usage(workload, num_blocks, block_size):
    """Admit as many sequences concurrently as the pool allows.

    Each sequence holds prompt_len + output_len tokens for its whole life
    (worst case — no reuse of freed blocks within a sequence). Blocks are
    not freed until the pool is full, so peak concurrency is what we measure.
    """
    mgr = BlockManager(num_blocks=num_blocks, block_size=block_size)
    admitted = 0
    for s in workload:
        mgr.register(s.id)
        try:
            mgr.append_tokens(s.id, s.prompt_len + s.output_len)
            admitted += 1
        except MemoryError:
            mgr.free(s.id)
            break

    alloc_slots = mgr.num_allocated() * block_size
    used_slots = sum(
        s.prompt_len + s.output_len for s in workload[:admitted]
    )
    waste = alloc_slots - used_slots
    return mgr.num_allocated(), admitted, waste


def naive_usage(workload, num_blocks, block_size, max_seq_len=2048):
    """Reserve max_seq_len per sequence. Nothing else fits after that."""
    pool_slots = num_blocks * block_size
    slots_per_seq = max_seq_len
    admitted = min(pool_slots // slots_per_seq, len(workload))

    alloc_slots = admitted * slots_per_seq
    used_slots = sum(
        s.prompt_len + s.output_len for s in workload[:admitted]
    )
    waste = alloc_slots - used_slots
    blocks = admitted * (max_seq_len // block_size)
    return blocks, admitted, waste


def fixed_slot_usage(workload, num_blocks, block_size, expected_len=200):
    """Reserve expected_len per sequence. Sequences longer than expected OOM."""
    pool_slots = num_blocks * block_size
    admitted = 0
    oom = 0
    used_slots = 0
    for s in workload:
        actual = s.prompt_len + s.output_len
        if actual > expected_len:
            oom += 1
            continue
        if used_slots + expected_len > pool_slots:
            break
        used_slots += expected_len
        admitted += 1

    alloc_slots = admitted * expected_len
    actual_used = sum(
        s.prompt_len + s.output_len
        for s in workload[:admitted + oom]
        if s.prompt_len + s.output_len <= expected_len
    )
    waste = alloc_slots - actual_used
    blocks = admitted * (expected_len // block_size + 1)
    return blocks, admitted, waste, oom

# ---- report ------------------------------------------------------------

def main() -> None:
    workload = make_workload(n=500, seed=42)
    pool_blocks = 1024 * 4          # 4096 blocks
    pool_slots = pool_blocks * BLOCK_SIZE
    print(f"workload: {len(workload)} sequences, "
          f"prompt~logN(60,0.7), output~logN(100,0.8)")
    print(f"pool: {pool_blocks} blocks × {BLOCK_SIZE} = {pool_slots:,} slots")
    print(f"max_seq_len reservation: {MAX_SEQ_LEN}\n")

    blocks, admitted, waste = paged_usage(workload, pool_blocks, BLOCK_SIZE)
    print(f"paged (block_size={BLOCK_SIZE}):")
    print(f"  admitted concurrent sequences : {admitted}")
    print(f"  blocks allocated              : {blocks} / {pool_blocks}")
    print(f"  waste                         : {waste} slots "
          f"({waste/pool_slots*100:.2f}% of pool)\n")

    blocks, admitted, waste = naive_usage(workload, pool_blocks, BLOCK_SIZE)
    print(f"naive pre-allocation (reserve {MAX_SEQ_LEN} slots/seq):")
    print(f"  admitted concurrent sequences : {admitted}")
    print(f"  blocks allocated              : {blocks} / {pool_blocks}")
    print(f"  reserved-but-unused           : {waste} slots "
          f"({waste/pool_slots*100:.2f}% of pool)\n")

    blocks, admitted, waste, oom = fixed_slot_usage(
        workload, pool_blocks, BLOCK_SIZE, expected_len=200)
    print(f"fixed 200-slot reservation:")
    print(f"  admitted concurrent sequences : {admitted}")
    print(f"  blocks allocated              : {blocks} / {pool_blocks}")
    print(f"  requests rejected (over-reservation) : {oom}\n")


if __name__ == "__main__":
    main()
