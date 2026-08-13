# Inference engine internals

The engine's job: run a HuggingFace model where the attention math goes through
`flashattn_cuda`, one prompt pass (prefill) and then one token at a time (decode),
against a preallocated KV cache. Everything here is testable, and most of it is
testable **on a CPU**.

## The ModelRunner contract (`engine/models/base.py`)

```
prefill(input_ids)   -> logits for the last position; the KV cache is now filled
decode_step(tokens)  -> logits for the next position; the cache grew by one row
```

Two methods on purpose: prefill and decode are different computations with
different kernels underneath (see [kernel.md](kernel.md)), and a runner that hides
that difference also hides where the time goes.

Correctness bar: each runner's greedy generation must match stock
`model.generate(do_sample=False)` token for token, with any divergence required to
pass the shared tie gate (below).

## GPT-2 (`engine/models/gpt2.py` + `engine/patching.py`)

GPT-2 computes Q, K, V in one fused `c_attn` matrix. The patch:

1. split the fused projection into Q, K, V
2. reshape each to `(batch, 12 heads, N, 64)`
3. call `flashattn_cuda.prefill(q, k, v, causal=True, scale=1/8)`  — 1/8 = 1/sqrt(64)
4. merge heads back and hand the result to GPT-2's original output projection

It is a **module patch, not a fork**: under transformers 5.x the pluggable
attention registry is the API that churns, so the patch grabs the stabler seam.
Proof it works: `examples/generate.py` prints stock vs patched generations with
`identical output : True`.

## Qwen2.5-0.5B-Instruct (`engine/models/qwen25.py`)

Adds three things GPT-2 does not have, and the placement of each is deliberate:

- **RoPE** — rotary position embedding, applied to Q and K **host-side, before the
  kernel**, from precomputed cos/sin tables. During decode the new token must be
  rotated with its true absolute position `t`, not 0; the decode-vs-prefill
  consistency test would expose an offset bug immediately.
- **GQA** — 14 query heads share 2 KV heads. Handled correctness-first by
  repeating the KV heads 2 -> 14 host-side before the kernel. The memory cost of
  that repeat is real and documented; removing it (kernel-side GQA) is named
  future work in the kernel repo.
- **Chat template** — via `tokenizer.apply_chat_template`, so the server's OpenAI
  `messages` array becomes exactly the prompt format the model was trained on.

The kernel itself is unchanged between the two models because both use head
dimension 64 — which is why they were chosen.

## KV cache (`engine/kv_cache.py`)

- Preallocated per layer as fp16 `(B_max=8, H_kv, S_max=2048, 64)`; no allocation
  happens on the decode path.
- Slot lifecycle per request (allocate on admit, free on finish), per-sequence
  lengths, `append(layer, slot, k, v)` writes one row per decode step.
- Prefill writes the whole prompt's K,V in one shot; decode passes the cache
  tensors plus `seq_lens` straight to `flashattn_cuda.decode` — no copies, which
  is why contiguity is validated at the kernel boundary.
- Every piece of indexing logic has CPU tests against an independent loop-based
  reference implementation, so cache bugs are caught without a GPU.

## Sampling (`engine/sampling.py`)

Greedy, temperature, top-p — pure-torch functions, CPU-testable, EOS stop,
max-tokens cap. The benchmark runs everything at temperature 0 for repeatability.

## The tie gate (why "identical output" is defined carefully)

At fp16, two logits can be **one representable step apart** — and one fp16 ulp
near magnitude 16 is 0.015625, so an absolute threshold like 1e-2 is not strict,
it is *unsatisfiable*. The shared gate in `tests/parity_utils.py` therefore
defines a tie as: top-2 gap <= 2 fp16 ulps, **plus an fp32 arbiter** — recompute
the position in fp32; if it prefers our token, pass; if it prefers the baseline's,
our error must be within 1.5x of the baseline's; if it prefers neither, fail.

Both parity suites import this one definition. The reason it exists as a
mechanism rather than a count: the same kernel scored 3/5 exact greedy matches on
one torch build and 5/5 on another — at a true tie, argmax depends on floating
point reduction order, so exact-sequence match measures the *environment*, and
the invariant gates carry the correctness claim.

## What the engine deliberately is not

No continuous batching, no paged cache, no CUDA graphs, no quantisation. Those
are serving-framework features; this engine exists to make one comparison clean.
The measured price of their absence is the repo's headline finding
([profiles/decode_dispatch.md](profiles/decode_dispatch.md)).
