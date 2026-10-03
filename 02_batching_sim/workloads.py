"""Workload generator: N concurrent users with realistic arrival patterns.

We generate requests with:
- Poisson arrivals (a standard model for independent users)
- Prompt lengths sampled from a log-normal (short median, long tail)
- Output lengths sampled from a log-normal (same reasoning)

Why log-normal and not uniform? In practice, LLM requests have a heavy
tail: most are short, a few are very long. Uniform would understate the
head-of-line blocking that naive batching suffers.
"""
from __future__ import annotations
import random
from simulator import Request


def generate_workload(
    num_users: int,
    arrival_rate_rps: float = 5.0,
    prompt_mean: int = 50,
    prompt_sigma: float = 0.7,
    output_mean: int = 80,
    output_sigma: float = 0.8,
    seed: int = 0,
) -> list[Request]:
    """Generate num_users requests with Poisson arrivals and log-normal lengths.

    arrival_rate_rps: average requests per second across the whole run.
    prompt_mean, output_mean: median lengths; sigma controls the tail width.
    """
    rng = random.Random(seed)
    requests: list[Request] = []
    clock = 0.0

    for i in range(num_users):
        # Poisson inter-arrival: exponential with mean 1/rate
        clock += rng.expovariate(arrival_rate_rps)

        prompt_len = max(1, int(rng.lognormvariate(
            __import__("math").log(prompt_mean), prompt_sigma)))
        output_len = max(1, int(rng.lognormvariate(
            __import__("math").log(output_mean), output_sigma)))

        requests.append(Request(
            id=i,
            prompt_len=prompt_len,
            output_len=output_len,
            arrival_time=clock * 1000.0,  # seconds → ms
        ))

    return requests
