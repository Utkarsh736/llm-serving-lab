"""Batching policies.

Both operate on the same Request/CostModel interface and return a SimResult.
The only difference is *when* the batch is refilled from the queue.
"""
from __future__ import annotations
from simulator import Request, CostModel, SimResult


def naive_batching(requests: list[Request], cost: CostModel, max_batch_size: int) -> SimResult:
    """Static batching: fill batch, run to completion, refill.

    Wastes slots: a request that finishes early leaves its slot idle until
    the slowest request in the batch finishes.
    """
    pending = sorted(requests, key=lambda r: r.arrival_time)
    clock = 0.0
    steps = 0
    next_idx = 0

    while next_idx < len(pending):
        # wait for the next request to arrive
        if clock < pending[next_idx].arrival_time:
            clock = pending[next_idx].arrival_time

        # form the batch
        batch: list[Request] = []
        while next_idx < len(pending) and len(batch) < max_batch_size:
            batch.append(pending[next_idx])
            next_idx += 1

        # prefill each request in the batch (sequential prefills)
        for r in batch:
            clock += cost.prefill(r.prompt_len)
            r.start_time = clock

        # decode until the slowest request in the batch is done
        max_output = max(r.output_len for r in batch)
        for _ in range(max_output):
            clock += cost.decode_step(len(batch))
            steps += 1
            for r in batch:
                if r.tokens_generated < r.output_len:
                    r.tokens_generated += 1
                    if r.tokens_generated == r.output_len:
                        r.end_time = clock
        # Note: a finished request's slot stays idle until the batch ends.

    return SimResult(
        policy="naive_batching",
        requests=pending,
        total_time_ms=clock,
        decode_steps=steps,
    )


def continuous_batching(requests: list[Request], cost: CostModel, max_batch_size: int) -> SimResult:
    """Iteration-level scheduling: refill as slots free.

    A request that finishes frees its slot immediately; a new request can
    be prefilled and start decoding on the very next iteration.
    """
    pending = sorted(requests, key=lambda r: r.arrival_time)
    clock = 0.0
    steps = 0
    next_idx = 0
    running: list[Request] = []

    while next_idx < len(pending) or running:
        # if nothing is running, jump to the next arrival
        if not running and next_idx < len(pending) and clock < pending[next_idx].arrival_time:
            clock = pending[next_idx].arrival_time

        # fill the batch while there is room and requests have arrived
        while (len(running) < max_batch_size
               and next_idx < len(pending)
               and pending[next_idx].arrival_time <= clock):
            r = pending[next_idx]
            clock += cost.prefill(r.prompt_len)
            r.start_time = clock
            running.append(r)
            next_idx += 1

        if not running:
            continue

        # one decode step for the whole running batch
        clock += cost.decode_step(len(running))
        steps += 1
        for r in running:
            r.tokens_generated += 1
            if r.tokens_generated == r.output_len:
                r.end_time = clock

        # free finished slots immediately
        running = [r for r in running if r.end_time is None]

    return SimResult(
        policy="continuous_batching",
        requests=pending,
        total_time_ms=clock,
        decode_steps=steps,
    )
