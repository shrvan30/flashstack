# Agent benchmark: three backends, one task suite

Every backend runs the identical 20-task suite through the identical agent
at temperature 0. The only thing that changes between runs is `base_url`
and the model name.

| | |
| :-- | :-- |
| GPU | NVIDIA GeForce RTX 3090 |
| Driver | 580.126.09 |
| CUDA (torch) | 13.0 |
| PyTorch | 2.11.0+cu130 |
| flashstack commit | `224f69c` |
| flash-attention-cuda commit | `91c091e` |
| suite | 20 tasks over the Halden Systems (fictional) corpus |

## Results

| backend | model | success | calls/task | retries | TTFT p50 (ms) | decode (tok/s) | task p50 (s) | task p95 (s) | cost/task |
| :-- | :-- | --: | --: | --: | --: | --: | --: | --: | --: |
| flashstack | `Qwen/Qwen2.5-0.5B-Instruct` | 15% (3/20) | 6.8 | 20 | 101 | 39.0 | 9.4 | 13.6 | 0.00092 |
| vLLM | `Qwen/Qwen2.5-0.5B-Instruct` | 10% (2/20) | 3.7 | 20 | 24 | 273.8 | 0.5 | 1.9 | 0.00007 |
| hosted anchor | `llama-3.3-70b-versatile` | 100% (20/20) | 3.0 | 0 | 234 | 523.4 | 1.3 | 3.3 | 0.00137 |

TTFT and decode throughput above are client-side stream measurements on every row; see [Metric provenance](#metric-provenance) for the per-backend labels and the server-reported cross-check.

Cost is not one number measured two ways. Local backends are billed by the
hour, so their cost is wall-clock GPU time at the rented rate; a hosted
backend is billed per token, so its cost comes from the usage its API
reports. The two columns are the same currency and different accounting.

- **flashstack**: wall-clock GPU time at 0.35/hour
- **vLLM**: wall-clock GPU time at 0.35/hour
- **hosted anchor**: token pricing (0.59/Mtok in, 0.79/Mtok out)

## Per-tier breakdown

| backend | single-tool | two-tool | multi-step (4+ calls) |
| :-- | --: | --: | --: |
| flashstack | 1/8 | 2/8 | 0/4 |
| vLLM | 2/8 | 0/8 | 0/4 |
| hosted anchor | 8/8 | 8/8 | 4/4 |

## Metric provenance

Published TTFT and decode throughput are the client-side stream measurement on every backend. A server-reported figure is recorded as a cross-check and never substituted into a published column.

| backend | streamed | TTFT p50 source | decode tok/s source | server cross-check |
| :-- | :-- | :-- | :-- | :-- |
| flashstack | yes | client-stream | client-stream | 66 ms from final-chunk (135 calls) |
| vLLM | yes | client-stream | client-stream | none reported |
| hosted anchor | yes | client-stream | client-stream | none reported |

Every published latency and throughput figure above is a client-side
stream measurement, taken the same way on all three backends.

The cross-check column is flashstack's own `x-ttft-ms`, which the other two
backends do not report. It is shown because the difference between it and the
published client figure is a real quantity — the transport and framing cost
the client pays on top of the server's own timing — and hidden nowhere: it is
never the number in the results table.

For flashstack that difference is **34.9 ms** (101.4 ms observed by the client against 66.5 ms reported by the server, 1.53x). That gap is the measured cost of SSE framing and serialization plus loopback transport: the server stops its clock when the first token is generated, while the client starts seeing it only after the chunk has been JSON-encoded, wrapped in an SSE event and pushed through the socket. Published TTFT is the client figure because that is what a caller actually waits for, and because it is the only definition the other two backends can also supply.

## How retries and throttling are counted

Two different things can make a backend issue more calls than the task
needs, and they are accounted differently because they cost differently.

- **Parse retries** are billed work. The model returned something that was
  not a valid action object, the agent sent one corrective message, and the
  backend generated a second time. Those calls appear in `calls/task`, their
  tokens appear in the token counts, and their latency stays inside the task
  latency figures. A backend that formats badly should look more expensive,
  because it is.
- **Throttle waits** are not work. A hosted endpoint refused the request
  before serving it, so nothing was computed and nothing was billed. The
  wait is counted and reported separately, and is subtracted from task
  latency and from the run's wall clock. Leaving it in would charge a local
  backend's GPU-hour rate for time spent sleeping, and would make a
  rate-limited hosted anchor look computationally slow when it was merely
  queued.

## Charts

![decode throughput](comparison.svg)


## Where the gap comes from

vLLM decodes at 273.8 tok/s against flashstack's 39.0, a **7.0x** gap on identical weights. Three causes are usually offered for a gap like this. Only one of them explains this one, and the numbers say which.

### 1. Host dispatch overhead — sufficient on its own

An Nsight Systems trace of decode steps ([docs/profiles/decode_dispatch.md](../../docs/profiles/decode_dispatch.md)) measures the GPU **6% busy and 94% idle** during decoding. The engine issues about 1,388 GPU operations per token, each running ~2.8 us, separated by ~79 us of host-side gap: Python attribute lookups, tensor allocation, argument checking and the launch call itself.

If dispatch were free and the same GPU work were issued back to back, flashstack would decode at 651 tok/s (16.7x its measured rate). That ceiling is **2.4x beyond vLLM's measured 273.8 tok/s**.

So dispatch overhead alone more than accounts for the whole 7.0x gap. Nothing else needs to be invoked to explain it — which does not mean nothing else is true, only that nothing else is *required*.

### 2. Continuous batching — contributes nothing to this measurement

**0x, here.** Continuous batching raises throughput by admitting new requests between decode steps instead of waiting for a batch to drain. This benchmark never gives it the chance: the agent is strictly sequential, with one request in flight at a time, so there is never a second request to admit. vLLM's scheduler is running the same single stream flashstack's is.

This is a real architectural advantage of vLLM and it is the right answer for a served deployment under concurrent load. It explains none of the gap measured *here*, and quoting it as the cause would be borrowing an explanation from a workload this report did not run.

### 3. Attention kernel quality — bounded at about 1.5%

The decode attention kernels cost 15.9 us per layer per token at S=1024 (split + merge, from the kernel repository's nsys traces). Across Qwen2.5-0.5B's 24 layers that is **0.38 ms of attention per token**, against a measured 25.6 ms per token overall — about **1.5%** of the decode wall clock.

Making attention *infinitely fast* would therefore take flashstack from 39.0 to about 39.6 tok/s. The hand-written kernel this whole project is built around is worth at most 1.5% of the number it is most often assumed to control.

### What that adds up to

| cause | contribution to this gap |
| :-- | :-- |
| Host dispatch overhead | sufficient alone — a 16.7x ceiling against a 7.0x gap |
| Continuous batching | 0x — the workload is sequential |
| Attention kernel quality | <= 1.5% of decode wall clock |

The honest reading is that this comparison measures **how fast Python can issue launches**, not how good the attention kernel is. That is the finding, and it is the opposite of the one the project set out expecting. The kernel work is real, measured, and correct; the serving gap it sits inside is dominated by everything around it.

The changes that would actually move this number are CUDA graphs (replay a captured launch sequence instead of re-issuing it), operator fusion (fewer launches for the same work), and paged KV (more concurrent sequences per byte, which raises achievable batch size). All three attack the launch count. None of them is about attention, and all are out of scope for this stage by instruction.

For completeness, the Phase-1 profiles show the prefill kernel reaching
roughly a third of this card's fp32 FMA peak and the decode kernel up to 27%
of DRAM peak. Respectable in isolation, and — as the decomposition above
shows — almost irrelevant to the end-to-end figures here.

## What the success column does and does not say

Success rate here is a property of **Qwen2.5-0.5B-Instruct**, not of the
serving stack. Two backends running the same weights at temperature 0 should
agree closely; where they do not, the difference is sampling and numerics,
not capability. The column is included because agent success rate drives
cost — a failed task still burns every call it made — and because a large
divergence between two backends on identical weights would itself be a bug
worth finding.
