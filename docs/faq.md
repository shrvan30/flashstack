# FAQ — the questions people actually ask

**Why is the success rate only 3/20? Is the stack broken?**
No — that column measures the *model*, not the stack. The suite was validated at
20/20 on a 70B model first (the solvability control), so low local scores are
attributable to Qwen2.5-0.5B's capacity for strict-JSON multi-step tool use — see
the per-tier table: both local backends went 0/4 on tasks needing 4+ chained
calls. The serving-correctness result is that my engine and vLLM score the same
(3 vs 2 on n=20 is noise) on identical weights. The systems story lives in the
TTFT, throughput, and cost columns.

**You're 7x slower than vLLM. Why publish that?**
Because the *explanation* is the product. The trace shows ~1,388 GPU operations
per token, 2.8 us each, separated by ~79 us of host gap — the GPU is ~6% busy.
Remove only those gaps and the identical GPU work runs at ~651 tok/s: 16.7x this
stack's speed and 2.4x past vLLM's measured rate. Host dispatch alone accounts
for the entire gap; the attention kernel's share is <= ~1.5%. A pretty number was
one shortcut away; an explained number is worth more.

**So the custom kernel didn't matter? Why build it?**
It mattered for exactly what it was built for: learning the algorithm to the
metal, and *earning the right to exonerate it with evidence*. The end-to-end
finding — that the layer above dominates — is only credible because the kernel
layer is measured, tested, and attributable. That is the project's thesis: the
layers only make sense together.

**Why not just add CUDA graphs and fix the dispatch problem?**
Scope discipline. Graphs, fusion, and continuous batching were declared out of
scope before measurement began; the deliverable is the measured diagnosis, not a
partial reimplementation of vLLM. They are the ranked first items in future work,
with the expected magnitude already quantified by the ceiling arithmetic above.

**Why only head-dim-64 models? Can this serve Llama-3 or a 7B?**
Not yet. The kernel fixes head dimension at 64, which covers GPT-2 and
Qwen2.5-0.5B exactly. Larger models use head dimension 128 — supporting it means
re-deriving the kernel's shared-memory and register budgets, and it is the named
next step in the kernel repo precisely because it is what raises the success
column.

**Why is the hosted backend a 70B when the locals run a 0.5B? That's unfair.**
It is not a model comparison and is labelled as such: the hosted row is a
latency/cost *anchor* and the suite's solvability control. Its 20/20 is what
makes the 0.5B scores interpretable at all.

**Why a fictional company corpus?**
So no model can answer from memory. A real company's prices might be in training
data — then the benchmark measures recall, not tool use. Fictional documents make
every answer offline-verifiable and force the retrieval path.

**Why does the published TTFT differ from the server's own header?**
Published TTFT and decode tok/s are client-stream measurements on *every*
backend — one column, one meaning, the thing a client actually experiences. The
server's own figures are kept as labelled cross-checks; the gap between them
(~34.9 ms, 1.53x here) is the measured SSE/serialization overhead.

**Why no provider failover when the hosted API rate-limits?**
Failover is an availability pattern; a benchmark needs a measurement pattern —
hold the instrument constant. Tasks landing on different models and prices would
make the hosted column a meaningless mixture. Instead: wait out the 429 within
the same provider (backoff as the floor, `Retry-After` honoured only when
larger), count the wait separately, never bill it as serving time.

**Why static batching with a tiny 25 ms window?**
It is the honest minimum: enough machinery to prove batched prefill and lockstep
decode work (asserted from `/metrics` in a test), without pretending to be
continuous batching. Under the sequential benchmark it costs nothing; the
concurrency story is explicitly future work.

**Is any of this actually tested?**
~172 CPU tests (KV-cache indexing against an independent reference, sampling,
SSE framing, agent JSON parsing, metric provenance) run in CI on every commit;
GPU-marked tests (kernel parity under an fp16-ulp tie gate, server end-to-end)
run on real hardware before every tag. The kernel repo carries its own 99-test
suite.

**What would you build next, in order?**
CUDA graphs or a compiled host loop (attacks the measured 94%-idle gap), then
continuous batching and paged KV (the concurrency story), then head-dim 128 and
kernel-side GQA in the kernel repo (unlocks larger models — which is what raises
the success column), plus counter-validated kernel profiles (tracked there as
v2.0.1). Every item attacks a bottleneck that has already been measured, not one
being guessed at.
