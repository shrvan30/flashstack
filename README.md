<h1 align="center">FlashStack</h1>

<p align="center">
  <b>A small LLM inference stack built from scratch — CUDA kernel to agent benchmark — to measure where serving time actually goes.</b>
</p>

<p align="center">
  <a href="https://github.com/shrvan30/flashstack/actions/workflows/ci.yml"><img src="https://github.com/shrvan30/flashstack/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://github.com/shrvan30/flashstack/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License: MIT"></a>
  <img src="https://img.shields.io/badge/python-3.10%2B-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/CUDA-13.0-76B900.svg" alt="CUDA">
  <img src="https://img.shields.io/badge/PyTorch-2.11%2Bcu130-EE4C2C.svg" alt="PyTorch">
</p>

> **What did I build?** A from-scratch inference stack — a hand-written CUDA
> FlashAttention kernel, a PyTorch engine with a KV cache, an OpenAI-compatible
> streaming server, and a 20-task agent benchmark — measured against **vLLM** and a
> **hosted API** on one machine. Headline finding: my server is 7x slower than vLLM
> at decode, and the profiler proves the kernel is **not** the reason — the GPU is
> only ~6% busy, waiting on Python.

---

## Table of Contents

1. [What is FlashStack?](#1-what-is-flashstack)
2. [What can it actually do, today?](#2-what-can-it-actually-do-today)
3. [Motivation](#3-motivation)
4. [Architecture](#4-architecture)
5. [End-to-end flow](#5-end-to-end-flow)
6. [CUDA kernel](#6-cuda-kernel)
7. [Attention integration](#7-attention-integration)
8. [Inference engine](#8-inference-engine)
9. [API / streaming server](#9-api--streaming-server)
10. [Benchmark methodology](#10-benchmark-methodology)
11. [Results](#11-results)
12. [Engineering decisions](#12-engineering-decisions)
13. [Challenges and debugging](#13-challenges-and-debugging)
14. [Limitations](#14-limitations)
15. [Reproduce](#15-reproduce)
16. [Tests / CI](#16-tests--ci)
17. [Future work](#17-future-work)
- [Documentation](#documentation) · [License](#license)

---

## 1. What is FlashStack?

Ask ChatGPT a question and thousands of GPU operations fire before the first word
appears. FlashStack is that pipeline rebuilt small enough to see through: only the
attention math is replaced — with
[my own CUDA kernel](https://github.com/shrvan30/flash-attention-cuda) — and
everything else is kept deliberately simple PyTorch, so when the benchmark says
where the time goes, the answer is attributable.

## 2. What can it actually do, today?

No buzzwords — concrete capabilities, and the honest cannot-yet list.

**You can:**

- Serve **two real models** — GPT-2 (124M) and **Qwen2.5-0.5B-Instruct** — on a
  single NVIDIA GPU, with attention running on the hand-written kernel.
- Talk to it with **any OpenAI client**: change `base_url`, chat, stream tokens.
  No SDK changes, no custom protocol.
- Handle up to **4 requests at once** (static batching, 25 ms window), proven by a
  test that reads batch formation from `/metrics`.
- Run a complete **20-task tool-using agent benchmark** against this server, vLLM,
  or any hosted OpenAI-compatible API — and regenerate every published number with
  the commands in [Reproduce](#15-reproduce).

**It cannot (yet):** serve models whose head size is not 64 (so nothing bigger
than the 0.5B class), handle many concurrent users (no continuous batching), or
match vLLM's throughput — it is **7x slower at decode**, and the most valuable
artifact here is the profiler evidence of exactly why:
[`docs/profiles/decode_dispatch.md`](docs/profiles/decode_dispatch.md).

## 3. Motivation

Modern inference is fast. **Why?** Faster GPUs, better kernels, better batching,
better software? I wanted to measure it instead of guessing — so I built the whole
ladder myself and put the same workload on every rung:

my CUDA kernel → a tiny engine → an OpenAI-compatible server → an agent benchmark
→ compared against **vLLM** and a **hosted API** with identical tasks.

## 4. Architecture

![Architecture](docs/architecture.svg)

*Deep dive: [`docs/architecture.md`](docs/architecture.md)*

Two repositories, one dependency direction:
[flash-attention-cuda](https://github.com/shrvan30/flash-attention-cuda) is the
pure kernel library (pip-installable PyTorch extension); **flashstack** is the
application that consumes it.

```
flashstack/
├── engine/    model runners (GPT-2, Qwen2.5), KV cache, sampling, patching
├── server/    FastAPI app, scheduler (static batching), OpenAI schemas
├── agent/     ReAct loop, 3 tools, frozen 20-task suite + fictional corpus
├── bench/     runner, provenance-labelled metrics, scoring, report generator
└── docs/      writeup, architecture diagram, dispatch analysis
```

## 5. End-to-end flow

One decode token, end to end:

```
HTTP request (OpenAI JSON)
  → server/app.py            parse, stream setup
  → server/scheduler.py      queue → batch (≤4, 25 ms window)
  → engine/models/*.py       ModelRunner.decode_step()
  → engine/kv_cache.py       append this token's K,V row
  → flashattn_cuda.decode()  split-K attention over the cache   ← the custom kernel
  → sampling → tokenizer → SSE chunk to the client
```

Prefill is the same path with the whole prompt at once, calling
`flashattn_cuda.prefill()`.

## 6. CUDA kernel

Lives in its own repo; the short version: fp16 inputs / fp32 accumulation,
shared-memory tiling with online softmax, causal tile skipping, and a **separate
split-K decode kernel** whose chunk size is chosen at launch to keep ~2 blocks per
SM in flight (~3x over a fixed chunk at batch 1). Verified against a full-fp32
PyTorch reference to 2.44e-4 across 99 tests. Kernel-level tables:
[benchmarks](https://github.com/shrvan30/flash-attention-cuda/blob/main/docs/benchmarks.md) ·
bridge doc in this repo: [`docs/kernel.md`](docs/kernel.md).

## 7. Attention integration

`engine/patching.py` swaps the kernel into HuggingFace GPT-2: split the fused QKV
projection, reshape to `(B, 12, N, 64)`, call `prefill(causal=True, scale=1/8)`,
merge heads, keep the original output projection. A module patch, not a fork.
Proof: greedy generation matches the stock model token-for-token
(`examples/generate.py` prints `identical output : True`). Qwen2.5-0.5B adds RoPE
(applied host-side with true position offsets) and GQA (2 KV heads repeated to 14
host-side — the memory cost is documented as the motivation for kernel-side GQA).

## 8. Inference engine

- **ModelRunner** contract: `prefill(input_ids)` and `decode_step(token_ids)`,
  parity-tested against stock `model.generate()` under an fp16-ulp tie rule.
- **KV cache**: preallocated per-layer fp16 `(B_max=8, H_kv, 2048, 64)`, slot
  lifecycle per request, per-sequence lengths; all indexing logic CPU-tested
  against an independent loop-based reference.
- **Sampling**: greedy, temperature, top-p — pure-torch, CPU-testable.
- Everything except attention stays plain PyTorch **on purpose**: the kernel is
  the variable under study; the rest is the control.

*Deep dive: [`docs/engine.md`](docs/engine.md)*

## 9. API / streaming server

- `POST /v1/chat/completions` — streamed (SSE, OpenAI chunk framing, final
  `[DONE]`) and non-streamed; `GET /v1/models`; `GET /metrics`.
- **Static batching**: requests queue, group up to 4 within 25 ms, batched prefill
  then lockstep decode with finished sequences dropping out.
- **Honest TTFT**: the handler awaits the first generated token before sending
  headers, so `x-ttft-ms` is a real first-token time.
- Discovered by building: decode tokens/s **cannot** be a response header on a
  stream (headers precede the body), so it rides in the final chunk's metrics.

*Deep dive: [`docs/server.md`](docs/server.md)*

## 10. Benchmark methodology

The part I would defend hardest in a review (full document:
[`docs/benchmark.md`](docs/benchmark.md)):

- **Frozen suite**: 20 fixed tasks (8 single-tool, 8 two-tool, 4 multi-step) over
  three deterministic tools and a 10-document **fictional** corpus, so answers are
  offline-verifiable and can't come from model priors. Frozen byte-identical,
  verified by hash.
- **Solvability gate first**: before freezing, the suite had to score ≥80% on a
  strong hosted model. It scored **20/20** — so later low scores measure the small
  model, not broken tasks ([`bench/results/gate/`](bench/results/gate/)).
- **Metric provenance**: every figure carries a source label (client-stream /
  server header / usage) from call → task → JSON → report; the report refuses
  unlabelled figures and prints each column's source. Published TTFT and decode
  tok/s are **client-stream on every backend** — one column, one meaning.
- **Accounting rules**: parse retries are billed (real prefill+decode); HTTP-429
  throttle waits are not (no tokens consumed) and are counted separately.
- **One machine**: every published number comes from environment E1 (RTX 3090,
  driver 580.126.09, CUDA 13.0, torch 2.11+cu130), recorded in the report header.

## 11. Results

| Backend | Model | Success | TTFT p50 | Decode | Cost/task |
|---|---|---|---|---|---|
| **FlashStack** | Qwen2.5-0.5B-Instruct | 3/20 | 101 ms | 39.0 tok/s | $0.00092 |
| vLLM 0.26 | Qwen2.5-0.5B-Instruct | 2/20 | 24 ms | 273.8 tok/s | $0.00007 |
| Hosted anchor | Llama-3.3-70B (deliberately larger) | 20/20 | 234 ms | 523.4 tok/s | $0.00137 |

Read it correctly:

- **Success measures the model, not the stack.** The hosted 20/20 proves the tasks
  are solvable; both locals scoring ~equal (3 vs 2 on n=20 is noise) is the
  serving-correctness result — my engine preserved the model's capability as
  faithfully as vLLM did. Per-tier: both locals went **0/4 on multi-step** — a
  0.5B model cannot sustain schema-perfect JSON across 4+ chained calls.
- **The interesting discovery** is in the throughput column. I assumed a better
  attention kernel meant a faster server. The trace says otherwise: ~1,388 GPU ops
  per token, 2.8 µs each, separated by ~79 µs of host gap — **GPU ~6% busy, ~94%
  idle**. Remove only those gaps and the identical GPU work runs at ~651 tok/s:
  16.7x my speed and **2.4x past vLLM**. Host dispatch alone explains the whole
  7x gap; the attention kernel's share is ≤ ~1.5%.
- Cost is the same currency with different accounting: locals bill wall-clock GPU
  time; hosted bills list-price tokens.

Full report with gap attribution and per-figure provenance:
[`bench/results/report.md`](bench/results/report.md) ·
Dispatch evidence: [`docs/profiles/decode_dispatch.md`](docs/profiles/decode_dispatch.md)

## 12. Engineering decisions

- **The agent is a measurement workload, not a product** — minimal ReAct, frozen
  prompt, deterministic tools; its job is generating realistic multi-call traffic.
- **AST-whitelisted calculator, never `eval`** — untrusted model output reaches
  the evaluator.
- **A published column carries the one meaning measurable on all backends**
  (client-side observation); richer server figures become labelled cross-checks.
- **Backoff is the floor, `Retry-After` honored only when larger** — under load
  the provider's ~200 ms hints just get refused again; finite budget (8), then the
  task errors honestly.
- **No provider failover in a benchmark** — failover is an availability pattern;
  a benchmark holds its instrument constant.
- **Diagnose, don't optimise, the dispatch bound** — CUDA graphs were out of
  scope by plan; the measured attribution is the deliverable.

## 13. Challenges and debugging

- **The 2/20 that wasn't.** An early run scored 2/20 and looked like model
  failure. Root cause: my retrieval snippet cap (450 chars) truncated a 743-char
  catalogue — the asked-for prices were physically unreachable. Fixed, pinned by
  tests, re-run honestly. Rule learned: when an eval collapses, first ask whether
  the right answer was even *reachable*.
- **The near-miss the provenance system caught.** The old code preferred
  server-reported figures when present — which would have published my engine at
  66.5 ms in the same column as vLLM's client-measured 24 ms. Self-flattering by a
  third, invisibly. That's why every figure now carries its source.
- **The profiler that lied by 1.75x.** Tracing inflates exactly the host gaps
  being measured (113.5 vs 64.8 ms/step), so GPU-busy % is computed against the
  untraced step time.
- **The 429 wall.** A free-tier limit (12k tokens/min) failed 12 tasks before any
  model call; the fix distinguishes throttle waits from real work in the books.

## 14. Limitations

- Head size must be 64 → nothing larger than the 0.5B class today.
- Static batching only; no continuous batching, paged KV, or CUDA graphs — which
  is precisely why decode trails vLLM by 7x, per the attribution above.
- Success rates reflect a 0.5B model's ceiling on strict-JSON multi-step tool use
  (0/4 multi-step on both locals).
- n=20 tasks: success differences of 1–2 are noise and are treated as such.
- Hardware counters were unavailable on the measurement host; the kernel repo
  tracks counter-validated profiles as follow-up work.

## 15. Reproduce

```bash
# 1. Install flashstack + the kernel
git clone https://github.com/shrvan30/flashstack && cd flashstack
pip install -e .
git clone https://github.com/shrvan30/flash-attention-cuda
TORCH_CUDA_ARCH_LIST=8.6 pip install -e ./flash-attention-cuda --no-build-isolation
# (8.6 = RTX 3090/A10 · 8.9 = RTX 4090 · 9.0 = H100)

# 2. Serve
python -m uvicorn server.app:app --port 8000

# 3. Chat with any OpenAI client
python examples/generate.py   # or point openai's base_url at http://localhost:8000/v1

# 4. Benchmark this server
python -m bench.run --backend flashstack --base-url http://localhost:8000/v1 \
  --model Qwen/Qwen2.5-0.5B-Instruct --stream

# 5. Benchmark vLLM on the same GPU (separate venv recommended)
vllm serve Qwen/Qwen2.5-0.5B-Instruct --port 8001
python -m bench.run --backend vllm --base-url http://localhost:8001/v1 \
  --model Qwen/Qwen2.5-0.5B-Instruct --stream

# 6. Optional hosted anchor (any OpenAI-compatible endpoint)
export HOSTED_BASE_URL=... HOSTED_API_KEY=... HOSTED_MODEL=...
python -m bench.run --backend hosted --base-url "$HOSTED_BASE_URL" \
  --model "$HOSTED_MODEL" --stream \
  --input-price-per-mtok 0.59 --output-price-per-mtok 0.79

# 7. Report
python -m bench.report        # -> bench/results/report.md + comparison.svg
```

The task suite (`agent/tasks.json`, `agent/corpus/`, system prompt) is **frozen**;
editing it invalidates every published comparison.

## 16. Tests / CI

| Workflow | Purpose |
|---|---|
| [`ci.yml`](.github/workflows/ci.yml) | ruff lint + ~172 CPU tests: KV-cache indexing vs a reference impl, sampling, SSE framing, agent JSON parsing, metric provenance |

GPU-marked tests (kernel parity, server end-to-end) run on real hardware before
every tag — there is no GPU runner in CI, and this README won't pretend otherwise.

## 17. Future work

- CUDA graphs / compiled host loop — attacks the measured 94%-idle gap directly
- Continuous batching, then paged KV — the concurrency story
- Head-dim 128 + kernel-side GQA in the kernel repo — unlocks larger models,
  which is what raises the success column
- Counter-validated kernel profiles (tracked in the kernel repo)

---

## Documentation

| Resource | Description |
|---|---|
| [`docs/writeup.md`](docs/writeup.md) | The full story, layer by layer, and the surprise at the end |
| [`docs/architecture.md`](docs/architecture.md) | System architecture: the two repos, four layers, and design principles |
| [`docs/engine.md`](docs/engine.md) | Inference engine internals: ModelRunner, KV cache, RoPE/GQA, the tie gate |
| [`docs/server.md`](docs/server.md) | The OpenAI-compatible server: SSE, static batching, honest TTFT |
| [`docs/kernel.md`](docs/kernel.md) | Bridge to the CUDA kernel: what FlashStack calls and why it's shaped that way |
| [`docs/benchmark.md`](docs/benchmark.md) | Benchmark methodology: frozen suite, solvability gate, provenance, accounting |
| [`docs/faq.md`](docs/faq.md) | The questions people actually ask, answered honestly |
| [`bench/results/report.md`](bench/results/report.md) | Complete benchmark report: gap attribution, provenance, accounting rules |
| [`docs/profiles/decode_dispatch.md`](docs/profiles/decode_dispatch.md) | Nsight Systems dispatch analysis (the ~6%-busy finding) |
| [`docs/architecture.svg`](docs/architecture.svg) | Architecture diagram |
| [Kernel repo](https://github.com/shrvan30/flash-attention-cuda) | The CUDA kernel: design, benchmarks, roofline analysis |

## License

MIT — see [LICENSE](LICENSE). Thanks to the CUDA, PyTorch, HuggingFace, and vLLM
communities, and the FlashAttention authors.
