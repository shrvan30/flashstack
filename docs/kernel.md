# The FlashAttention kernel

The kernel lives in its own repository —
**[flash-attention-cuda](https://github.com/shrvan30/flash-attention-cuda)** —
because it is a component with its own tests, benchmarks, CI, and version
history. This page is the bridge: what FlashStack uses, why it is shaped that
way, and where the deep material lives.

## What FlashStack calls

Two functions, from a pip-installable PyTorch extension:

```python
import flashattn_cuda

# prompt phase: q, k, v are (batch, heads, N, 64), fp16, CUDA
out = flashattn_cuda.prefill(q, k, v, causal=True, scale=1/8)

# generation phase: q is (batch, heads, 1, 64); caches are (batch, heads, max_len, 64)
out = flashattn_cuda.decode(q, k_cache, v_cache, seq_lens, scale=1/8)
```

`engine/patching.py` routes GPT-2's attention through `prefill`;
`engine/kv_cache.py` hands its tensors straight to `decode` each step. The
contract — fp16, contiguous, head dimension exactly 64 — is validated in C++ at
the boundary, because a silently-wrong kernel is the worst failure mode a
numerics project can have.

## The two-kernel design, in one paragraph each

**Prefill** processes the whole prompt: shared-memory tiling (the N x N score
matrix never touches HBM), online softmax (running max and sum per row, fp32),
causal tiles skipped by loop bounds, fp16 inputs with fp32 accumulation. It
matches a full-fp32 PyTorch reference to **2.44e-4**.

**Decode** is a different problem — one query row against thousands of cached
rows has almost no natural parallelism — so it splits the cache into chunks
(**split-K**), computes a partial `(m, l, acc)` per chunk, and merges exactly.
The chunk size is chosen **at launch** to keep ~2 blocks per SM in flight; a
fixed chunk had left 24 blocks on an 82-SM GPU (a starved machine, not a slow
kernel), and the adaptive rule is worth ~3x at batch 1. A test proves every legal
chunk size produces identical output.

## Honest performance summary

On the recorded machine (RTX 3090, driver 580.126.09, CUDA 13.0):

- Prefill: **~3–4x slower than official flash-attn** — no tensor cores; ~31–34%
  of the fp32 FMA peak with a modeled ceiling near 57% from shared-memory
  bandwidth. Tensor cores are the named door past it.
- Decode: **1.3–1.8x faster** than a per-step eager SDPA loop up to context
  1024, **roughly equal at 2048** — and that baseline is labelled a low bar; the
  serious decode baselines (flash-attn's `flash_attn_with_kvcache`,
  FlashDecoding) are deferred and named.

And the finding this whole repo exists for: in end-to-end serving, the kernel is
**not** where the time goes — decode is host-dispatch-bound at ~6% GPU busy
([profiles/decode_dispatch.md](profiles/decode_dispatch.md)). A world-class
kernel under a slow dispatcher is a fast chef with a slow waiter.

## Where the deep material lives (kernel repo)

| Document | Contents |
|---|---|
| [README](https://github.com/shrvan30/flash-attention-cuda#readme) | Design, capabilities, engineering decisions, ELI5 walkthrough |
| [docs/benchmarks.md](https://github.com/shrvan30/flash-attention-cuda/blob/main/docs/benchmarks.md) | Full tables vs SDPA, flash-attn, and the v1 kernels |
| [docs/profiles/analysis.md](https://github.com/shrvan30/flash-attention-cuda/blob/main/docs/profiles/analysis.md) | Roofline + occupancy analysis, measured vs modeled, explicitly labelled |
| [docs/profiles/summary.md](https://github.com/shrvan30/flash-attention-cuda/blob/main/docs/profiles/summary.md) | The pending counter-validation plan (v2.0.1) |
