# Colab T4 Setup for vLLM Sweep

Working environment: Colab T4, vLLM 0.30.0, torch 2.13.0+cu130.

## 1. Runtime
Runtime → Change runtime type → T4 GPU → Save.
Confirm: `!nvidia-smi`

## 2. Install

    !pip install -q "vllm>=0.6.3" "openai>=1.52.2" nest_asyncio

## 3. Pre-download models (optional but recommended)

    !hf download Qwen/Qwen2.5-0.5B-Instruct
    !hf download Qwen/Qwen2.5-0.5B-Instruct-AWQ

Note: `huggingface-cli` is deprecated in favor of `hf`. The old name
still runs but prints a deprecation warning.

## 4. vLLM 0.30.0 specifics

- Use `vllm serve`, NOT `vllm server` (the deprecation message in 0.30.0
  incorrectly says `vllm server`).
- `--disable-log-requests` does not exist; the flag is `--enable-log-requests`
  and logging is off by default.
- Benchmark CLI: `vllm bench serve`.
- First server boot takes 60–180 s on a cold T4. Set wait timeout to 300 s.

## 5. Persist results
Download /content/results/*.json immediately after the sweep ends.
Sessions die without warning. Do not leave the tab idle.