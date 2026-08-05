# flashstack

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
| `engine/` — model runners, KV cache, sampling | in progress |
| `server/` — OpenAI-compatible FastAPI server | not started |
| `agent/` — ReAct agent used as a measurement workload | not started |
| `bench/` — three-backend benchmark harness | not started |

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
