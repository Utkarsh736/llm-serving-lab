"""Token-level scheduler simulator for LLM serving.

No real model, no GPU. We model the cost of each scheduling decision with
a simple latency function and simulate how different policies behave under
the same workload.

Cost model (toy but qualitatively correct):
    prefill_cost(prompt_len) = prefill_base_ms + prefill_per_token_ms * prompt_len
    decode_step_cost(batch)  = decode_base_ms  + decode_per_request_ms * batch_size

The decode step cost captures the key insight from Milestone 1: there is a
constant per-step floor (MLP, LayerNorm, sampling) that does not depend on
batch size. Batching amortizes that floor across requests.
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class Request:
    id: int
    prompt_len: int
    output_len: int
    arrival_time: float = 0.0
    # filled in by the simulation:
    start_time: float | None = None      # first token emitted (after prefill)
    end_time: float | None = None        # last token emitted
    tokens_generated: int = 0

    @property
    def ttft(self) -> float | None:
        """Time to first token, relative to arrival."""
        if self.start_time is None:
            return None
        return self.start_time - self.arrival_time

    @property
    def latency(self) -> float | None:
        """End-to-end latency, relative to arrival."""
        if self.end_time is None:
            return None
        return self.end_time - self.arrival_time

    @property
    def itl(self) -> float | None:
        """Average inter-token latency (ms per output token after the first)."""
        if self.end_time is None or self.start_time is None or self.output_len < 2:
            return None
        return (self.end_time - self.start_time) / (self.output_len - 1)


@dataclass
class CostModel:
    prefill_base_ms: float = 5.0
    prefill_per_token_ms: float = 0.05
    decode_base_ms: float = 4.0
    decode_per_request_ms: float = 1.0

    def prefill(self, prompt_len: int) -> float:
        return self.prefill_base_ms + self.prefill_per_token_ms * prompt_len

    def decode_step(self, batch_size: int) -> float:
        return self.decode_base_ms + self.decode_per_request_ms * batch_size


@dataclass
class SimResult:
    policy: str
    requests: list[Request]
    total_time_ms: float
    decode_steps: int

    @property
    def num_requests(self) -> int:
        return len(self.requests)

    @property
    def throughput_rps(self) -> float:
        if self.total_time_ms == 0:
            return 0.0
        return self.num_requests / (self.total_time_ms / 1000.0)


def clone_requests(requests: list[Request]) -> list[Request]:
    """Fresh copies so both policies see the same starting state."""
    return [
        Request(
            id=r.id,
            prompt_len=r.prompt_len,
            output_len=r.output_len,
            arrival_time=r.arrival_time,
        )
        for r in requests
    ]
