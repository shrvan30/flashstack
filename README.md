# flashstack

[![CI](https://github.com/shrvan30/flashstack/actions/workflows/ci.yml/badge.svg)](https://github.com/shrvan30/flashstack/actions/workflows/ci.yml)

An inference stack built bottom-up on hand-written CUDA: a FlashAttention kernel,
an engine that runs real models on it, an OpenAI-compatible server, and an agent
benchmark that measures the whole thing under realistic multi-call traffic.

Attention runs on [flash-attention-cuda](https://github.com/shrvan30/flash-attention-cuda)
— a from-scratch batched multi-head causal FlashAttention implementation with
separate prefill and decode kernels. Every other operation stays plain PyTorch;
the kernel is the point, not a re-implementation of the whole framework.

## Status

Early. The scaffold and the kernel integration land first; the engine, server and
benchmark follow.

| layer | state |
| :-- | :-- |
| `engine/` — model runners, KV cache, sampling | GPT-2 attention runs on the kernel |
| `server/` — OpenAI-compatible FastAPI server | not started |
| `agent/` — ReAct agent used as a measurement workload | not started |
| `bench/` — three-backend benchmark harness | not started |

## The kernel runs a real model

`engine.patching.patch_gpt2` replaces every GPT-2 block's attention with the
`flashattn_cuda` prefill kernel — splitting the fused QKV projection, reshaping to
the kernel's `(B, H, N, 64)` contract, and merging the heads back through the
original output projection. Weights are untouched; only the operation changes.

```console
$ python examples/generate.py --prompt "The capital of France is"
model            : gpt2
device           : NVIDIA GeForce RTX 3090
prompt           : 'The capital of France is'
new tokens       : 48 (greedy, use_cache=False)

--- stock attention -------------------------------------------------
 the capital of the French Republic, and the capital of the French Republic is the capital of the French Republic.

The French Republic is the capital of the French Republic.

The French Republic is the capital of the French Republic.

--- flashattn_cuda prefill ------------------------------------------
 the capital of the French Republic, and the capital of the French Republic is the capital of the French Republic.

The French Republic is the capital of the French Republic.

The French Republic is the capital of the French Republic.

identical output : True
stock            :    805.8 ms     59.6 tok/s
flashattn_cuda   :    793.6 ms     60.5 tok/s
```

The interesting result is `identical output : True`. Greedy generation of 64
tokens reproduces the stock model's token sequence exactly on all five parity
prompts, and logits differ by at most 0.19 — the patched model is never more than
1.03× further from an fp32 reference than stock fp16 attention is.

Those throughput numbers are **not** a kernel benchmark. This stage is
prefill-only, so both models run with `use_cache=False` and recompute the whole
sequence every step. The comparison is fair — identical work on both sides — but
far slower than cached decoding. The KV cache and the decode kernel arrive with
the engine; kernel-level numbers live in the
[kernel repo's benchmarks](https://github.com/shrvan30/flash-attention-cuda/blob/main/docs/benchmarks.md).

## Requirements

- NVIDIA GPU with compute capability 8.6 (developed on an RTX 3090)
- CUDA toolkit with `nvcc` on `PATH`
- Python 3.10+
- PyTorch matching the installed CUDA runtime

## Install

```bash
pip install -e .
```

The attention kernels are a separate, source-built CUDA extension and are **not**
a hard dependency of this package — that keeps the CPU test suite installable on a
machine without a toolchain. Install them explicitly:

```bash
pip install "flashattn_cuda @ git+https://github.com/shrvan30/flash-attention-cuda"
```

For development against a local checkout of the kernels, use an editable install
from the sibling directory instead, so kernel changes are picked up without a
reinstall:

```bash
git clone https://github.com/shrvan30/flash-attention-cuda
TORCH_CUDA_ARCH_LIST=8.6 pip install -e ./flash-attention-cuda --no-build-isolation
```

`--no-build-isolation` matters: the extension links against the torch already in
the environment, and an isolated build would compile it against a second, ABI-
incompatible copy.

Verify:

```bash
python -c "import flashattn_cuda; print(flashattn_cuda.__version__)"
```

## Tests

```bash
pip install -e ".[dev]"
pytest              # GPU tests skip automatically without a device or the extension
pytest -m gpu       # GPU tests only
ruff check .
```

## License

MIT — see [LICENSE](LICENSE).
