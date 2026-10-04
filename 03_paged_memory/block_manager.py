"""Paged memory model for the KV cache.

We model the KV cache as a fixed pool of physical blocks, each holding
BLOCK_SIZE token slots of K and V for one sequence. A sequence's cache
is scattered across non-contiguous blocks and indexed by a per-sequence
page table (logical block index -> physical block ID).

This mirrors vLLM's PagedAttention: contiguous-looking sequences,
non-contiguous physical storage. The point is to eliminate the internal
fragmentation of naive max_seq_len pre-allocation.

All quantities are in *token slots*, not bytes. Multiply by
2 * n_layer * n_head * head_dim * dtype_bytes at the end for real memory.
"""
from __future__ import annotations


class BlockManager:
    """Fixed-size block allocator with per-sequence page tables."""

    def __init__(self, num_blocks: int, block_size: int):
        if num_blocks <= 0 or block_size <= 0:
            raise ValueError("num_blocks and block_size must be positive")
        self.num_blocks = num_blocks
        self.block_size = block_size
        self._free: list[int] = list(range(num_blocks))   # LIFO free list
        self._pages: dict[int, list[int]] = {}            # seq_id -> [phys block]
        self._tokens: dict[int, int] = {}                 # seq_id -> tokens stored

    # ---- lifecycle -------------------------------------------------------

    def register(self, seq_id: int) -> None:
        """A new sequence enters. It starts with zero blocks."""
        if seq_id in self._pages:
            raise ValueError(f"seq {seq_id} already registered")
        self._pages[seq_id] = []
        self._tokens[seq_id] = 0

    def append_tokens(self, seq_id: int, n: int) -> None:
        """Sequence writes n more tokens. Allocate blocks as needed."""
        if seq_id not in self._pages:
            raise KeyError(f"seq {seq_id} not registered")
        if n < 0:
            raise ValueError("n must be >= 0")

        self._tokens[seq_id] += n
        needed = (self._tokens[seq_id] + self.block_size - 1) // self.block_size
        held = len(self._pages[seq_id])
        while held < needed:
            if not self._free:
                raise MemoryError(f"out of blocks (seq {seq_id} needs more)")
            self._pages[seq_id].append(self._free.pop())
            held += 1

    def free(self, seq_id: int) -> None:
        """Sequence finishes. Its blocks return to the pool."""
        blocks = self._pages.pop(seq_id, None)
        if blocks is None:
            raise KeyError(f"seq {seq_id} not registered")
        self._tokens.pop(seq_id)
        self._free.extend(blocks)

    # ---- introspection ---------------------------------------------------

    def num_free(self) -> int:
        return len(self._free)

    def num_allocated(self) -> int:
        return self.num_blocks - len(self._free)

    def page_table(self, seq_id: int) -> list[int]:
        return list(self._pages[seq_id])

    def tokens_stored(self, seq_id: int) -> int:
        return self._tokens[seq_id]

    def waste_slots(self, seq_id: int) -> int:
        """Unused token slots in this sequence's allocated blocks."""
        return len(self._pages[seq_id]) * self.block_size - self._tokens[seq_id]

    def total_waste_slots(self) -> int:
        return sum(self.waste_slots(s) for s in self._pages)


def _demo() -> None:
    mgr = BlockManager(num_blocks=16, block_size=16)
    print(f"pool: {mgr.num_blocks} blocks × {mgr.block_size} slots "
          f"= {mgr.num_blocks * mgr.block_size} token slots\n")

    # three sequences arrive with different prompt lengths
    plans = [(0, "prompt 30"), (1, "prompt 10"), (2, "prompt 5")]
    for seq_id, label in plans:
        mgr.register(seq_id)
        prompt_len = int(label.split()[-1])
        mgr.append_tokens(seq_id, prompt_len)
        print(f"seq {seq_id} ({label}): page table = "
              f"{mgr.page_table(seq_id)}  waste = {mgr.waste_slots(seq_id)} slots")

    print(f"\nafter prefills: free={mgr.num_free()}  "
          f"allocated={mgr.num_allocated()}  "
          f"total waste={mgr.total_waste_slots()} slots\n")

    # simulate decode: sequence 0 keeps growing, sequence 2 finishes
    for step in range(20):
        mgr.append_tokens(0, 1)
    mgr.free(2)
    print(f"seq 0 grew to {mgr.tokens_stored(0)} tokens; "
          f"page table = {mgr.page_table(0)}")
    print(f"seq 2 finished; free now = {mgr.num_free()}\n")

    # try to over-allocate
    mgr.register(3)
    try:
        mgr.append_tokens(3, 16 * 20)   # way more than the pool can hold
    except MemoryError as e:
        print(f"expected failure: {e}")


if __name__ == "__main__":
    _demo()
