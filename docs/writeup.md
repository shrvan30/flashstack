# From a CUDA kernel to an agent

This project was built bottom-up in four stages: a FlashAttention kernel written from scratch,
an engine that runs real models on it, an OpenAI-compatible server, and an agent benchmark that
measures the whole stack against vLLM and a hosted API. The goal was a specific question — how
much is a good attention kernel worth to end-to-end serving? — and the honest answer turned out
to be much smaller than the effort that went into the kernel. That result is the most valuable
thing the project produced.

## Layer 1: the kernel

The kernel is batched multi-head causal FlashAttention for `head_dim = 64`, fp16 in and out with
fp32 accumulation, in two variants. **Prefill** runs one block per `(batch, head, query tile)`
with an online softmax keeping running max and sum in registers, and skips key tiles entirely
above the diagonal rather than computing and masking them. **Decode** is a different problem: a
single query row against a long cache offers almost no parallelism along the query dimension, so
it splits the key dimension, computes partial `(m, l, acc)` triples in parallel, and merges them
with a log-sum-exp reduction.

The decode split size turned out to matter more than anything else in the kernel. With the chunk
fixed at 512 keys, one sequence at context 1024 launches 24 blocks on an 82-SM card — the GPU is
three-quarters empty. Choosing the largest power-of-two chunk that still reaches two blocks per
SM is worth **2.9–3.0x**. The chunk is purely a scheduling decision: the log-sum-exp merge makes
the result numerically identical whatever the split, and the test suite asserts exactly that
across every chunk in the range.

Against `flash-attn` 2.8.3 the prefill kernel is **2.9–3.7x slower** at B=8 causal, and 62x
faster than the v1 kernels it replaces. It sustains 10.4–11.7 TFLOP/s, about 29–32% of the
RTX 3090's 36.2 TFLOP/s fp32 FMA peak. The gap is structural, not a tuning deficit: every
multiply happens on the CUDA cores with operands arriving through shared memory at roughly 1.75
bytes per FMA against an SM that sustains about 1.0, which caps the design near 57% of fp32 peak
before any overhead. flash-attn wins by running the matmuls on tensor cores, taking operands from
registers via `ldmatrix`/`mma`. Closing that gap means writing a different kernel, not polishing
this one.

That explanation is a **model**, and the documentation says so. Confirming which unit actually
saturates needs hardware performance counters, and the machine everything here was measured on
denies access to them. Rather than estimate, the counter-derived figures are absent and tracked
as open work. Nothing in this project claims a measurement it did not make.

## Layer 2 and 3: engine and server

The engine runs GPT-2 and Qwen2.5-0.5B-Instruct with attention on the kernel and **every other
operation in stock eager PyTorch** — RoPE, GQA head expansion, layernorm, the SwiGLU MLP. The KV
cache is preallocated fp16, and the server is FastAPI speaking `/v1/chat/completions` with SSE
streaming and static batching that groups up to four requests in a 25 ms window.

The correctness bar was behavioural, not approximate: greedy generation through the patched model
must reproduce the stock model's token sequence exactly, with any divergence permitted only after
a position where the stock model's top-two logits are within 1e-2 — a genuine fp16 tie rather
than a bug.

## Layer 4: the measurement, and what it found

The benchmark runs a ReAct agent over 20 frozen tasks against a fictional-company corpus, so every
answer is verifiable offline and none can be recalled from model priors. The same suite, the same
agent, temperature 0; only `base_url` and the model change.

Before publishing anything, one harness bug had to be fixed. The code preferred each backend's
own reported metrics where available. flashstack reports its own TTFT; vLLM and the hosted
endpoint do not. So the TTFT column would have held a server timestamp for one row and a client
timestamp for the other two — three backends, two definitions, one column, and nothing visible
saying so. The fix was to fix the *published* figure at the client-side measurement everywhere,
because it is the only definition all three can supply, and to carry each server's own figure
alongside as a labelled cross-check. Every number now travels with a provenance label from the
call that produced it all the way into the report, and the report refuses to present a mixed
column as comparable. Had the old rule stood, flashstack would have published 66.5 ms against
vLLM's 24 ms, flattering itself by a third, invisibly.

Then the result. vLLM decodes at 273.8 tok/s against flashstack's 39.0 — a **7.0x gap on
identical weights**. Decomposing it:

- **Host dispatch overhead is sufficient on its own.** The GPU is 6% busy and 94% idle during
  decode, issuing ~1,388 operations per token, each ~2.8 µs, separated by ~79 µs of host-side
  gap. A dispatch-free ceiling is ~651 tok/s — 2.4x *beyond* vLLM's measured rate.
- **Continuous batching contributes exactly zero.** The agent is strictly sequential, so there is
  never a second request to admit. It is a genuine vLLM advantage under concurrent load and it
  explains none of this measurement. Citing it would mean borrowing an explanation from a
  workload that was not run.
- **Attention kernel quality is worth at most 1.5%.** 0.38 ms of a 25.6 ms token. An infinitely
  fast attention kernel moves 39.0 tok/s to 39.6.

The benchmark measures how fast Python can issue launches. The hand-written kernel — the thing
the whole project is built around — controls about one and a half percent of the number it is
usually assumed to control.

## What I would do next, and what I learned

The changes that would actually move the number all attack the launch count: CUDA graphs to
replay a captured launch sequence, operator fusion, paged KV to raise achievable batch size.
None is about attention.

The broader lesson is about measurement discipline. Three separate times, the careful thing and
the convenient thing diverged: publishing the server's flattering TTFT, listing continuous
batching among the causes because it is true in general, and estimating counter-derived figures
rather than leaving them absent. Each would have produced a more impressive-looking document and
a less true one. The reason to build the harness carefully is that it makes the inconvenient
answer as easy to publish as the convenient one — and here, the inconvenient answer was the
interesting one.
