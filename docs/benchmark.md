# Benchmark methodology

The question the benchmark answers: **for the same agent workload, what do
latency, throughput, and cost look like across three ways of serving an LLM** —
this stack, vLLM, and a hosted API? The design choices below exist so that when
the table says something inconvenient, it can be trusted anyway.

Results live in [`bench/results/report.md`](https://github.com/shrvan30/flashstack/blob/main/bench/results/report.md); this
page is why the numbers mean what they say.

## The workload: 20 frozen tasks

- **Composition:** 8 single-tool, 8 two-tool, 4 multi-step (4+ chained calls) —
  three tiers that isolate lookup, composition, and sustained tool chains.
- **Tools:** an AST-whitelisted calculator (never `eval` — untrusted model output
  reaches the evaluator), BM25 search over a bundled corpus, and a deterministic
  mock API.
- **A fictional corpus, on purpose:** 10 documents about a made-up company, so
  every answer is offline-verifiable and *cannot come from model priors*. A real
  company's prices might sit in training data — and then the benchmark would
  measure memory, not tool use.
- **Frozen byte-identical:** `agent/tasks.json`, `agent/corpus/`, and the system
  prompt are hash-verified frozen. Editing any of them invalidates every
  published comparison.

## The solvability gate (run before anything was compared)

A suite a strong model cannot pass measures itself, not the stacks under it. So
before freezing, the suite had to score >= 80% on a strong hosted model. It
scored **20/20** (evidence in [`../bench/results/gate/`](../bench/results/gate/),
kept deliberately separate from the published anchor run). That control is what
makes a low local score attributable to the small model rather than to broken
tasks.

## Metric definitions — one column, one meaning

- **TTFT** and **decode tok/s** are published as **client-stream measurements on
  every backend** — the one meaning available on all three, and what a consumer
  of each backend actually experiences. Server-reported figures are kept as
  labelled cross-checks (the delta on this server is itself a finding: ~34.9 ms
  of SSE/serialization overhead).
- **Provenance is enforced, not hoped for.** Every figure carries a source label
  (client-stream / server header / usage) from call -> task -> results JSON ->
  report; the report generator *refuses* unlabelled figures and prints each
  column's source. This system exists because its absence nearly published this
  engine's server-side 66.5 ms in the same column as vLLM's client-side 24 ms —
  self-flattering by a third, invisibly.
- **Success** measures the **model**, not the serving stack — two backends on
  identical weights at temperature 0 should agree, and they do (3 vs 2 on n=20 is
  noise). It stays in the table because success drives cost: a failed task still
  burns every call it made.

## Accounting rules

- **Parse retries are billed.** A malformed-JSON retry is a real prefill and
  decode; it counts in calls, latency, and cost. Retry economics are a
  first-class property of small-model agents (both locals logged 20 retries; the
  70B logged zero).
- **Throttle waits are not billed.** An HTTP 429 consumed no tokens and says
  nothing about serving speed; waits are counted separately and excluded from
  latency and wall clock. Retry policy: exponential backoff is the floor, the
  server's `Retry-After` hint is honoured only when larger (under load the
  provider's ~200 ms hints just get refused again), finite budget of 8, then the
  task errors honestly. Local backends never emit 429, so the path is provably
  inert there — the workload stays identical across backends.
- **Cost is one currency, two accountings.** Local backends: wall-clock GPU time
  x the rental rate. Hosted: list-price tokens from API-reported usage — list
  price even on a free tier, because quota is a promotion, not a price, and a
  $0.00 column would claim nothing.

## The anchor is a different model, and says so

The hosted backend runs a much larger model **deliberately**: it is a
latency/cost anchor and the solvability control, not an apples-to-apples model
comparison, and it is labelled as such everywhere it appears.

## One machine

Every published number comes from one recorded environment (RTX 3090, driver
580.126.09, CUDA 13.0, torch 2.11+cu130), stated in the report header. Numbers
without their environment are not reproducible; several conclusions in this
project moved when the toolchain did, which is why this rule exists.

## Statistical honesty

n = 20 tasks. Success differences of 1–2 are noise and are treated as noise; the
throughput and latency columns rest on hundreds of calls and carry p50/p95. No
sentence in the report claims a win the sample size cannot support.

## Running it

See [Reproduce in the README](../README.md#15-reproduce) for the exact commands
for all three backends, and `python -m bench.report` to regenerate
`report.md` + `comparison.svg` from the result JSONs.
