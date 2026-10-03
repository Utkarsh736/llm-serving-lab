"""Tiny workload — 5 requests, side-by-side comparison of the two policies.

This is the session-1 sanity check. Once the output makes sense, benchmark.py
scales up to 10 / 50 / 100 concurrent users.
"""
from __future__ import annotations
from simulator import CostModel, Request, clone_requests
from policies import naive_batching, continuous_batching


def print_result(result) -> None:
    print(f"\n=== {result.policy} ===")
    print(f"total time    : {result.total_time_ms:.1f} ms")
    print(f"decode steps  : {result.decode_steps}")
    print(f"throughput    : {result.throughput_rps:.2f} req/s")
    header = f"{'id':>3} {'arrive':>8} {'start':>8} {'end':>8} {'TTFT':>8} {'lat':>8} {'ITL':>8}"
    print(header)
    for r in result.requests:
        itl = r.itl if r.itl is not None else 0.0
        print(f"{r.id:>3} {r.arrival_time:>8.1f} {r.start_time:>8.1f} {r.end_time:>8.1f} "
              f"{r.ttft:>8.1f} {r.latency:>8.1f} {itl:>8.2f}")


def main() -> None:
    cost = CostModel()
    base_requests = [
        Request(id=0, prompt_len=10, output_len=5,  arrival_time=0.0),
        Request(id=1, prompt_len=10, output_len=10, arrival_time=0.0),
        Request(id=2, prompt_len=10, output_len=15, arrival_time=0.0),
        Request(id=3, prompt_len=10, output_len=20, arrival_time=0.0),
        Request(id=4, prompt_len=10, output_len=25, arrival_time=0.0),
    ]
    max_batch_size = 3

    print(f"workload: 5 requests, output_len=[5,10,15,20,25], "
          f"prompt_len=10, max_batch_size={max_batch_size}")
    print(f"cost:     prefill = {cost.prefill_base_ms} + "
          f"{cost.prefill_per_token_ms}*L ms, "
          f"decode = {cost.decode_base_ms} + "
          f"{cost.decode_per_request_ms}*B ms")

    naive = naive_batching(clone_requests(base_requests), cost, max_batch_size)
    cont = continuous_batching(clone_requests(base_requests), cost, max_batch_size)

    print_result(naive)
    print_result(cont)

    print("\n=== comparison ===")
    print(f"total time : naive {naive.total_time_ms:8.1f} ms   "
          f"continuous {cont.total_time_ms:8.1f} ms   "
          f"({naive.total_time_ms / cont.total_time_ms:.2f}x)")
    print(f"throughput : naive {naive.throughput_rps:8.2f} req/s  "
          f"continuous {cont.throughput_rps:8.2f} req/s")


if __name__ == "__main__":
    main()
