# flashstack

[![CI](https://github.com/shrvan30/flashstack/actions/workflows/ci.yml/badge.svg)](https://github.com/shrvan30/flashstack/actions/workflows/ci.yml)

An inference stack built bottom-up on hand-written CUDA — a FlashAttention kernel, an engine
that runs real models on it, an OpenAI-compatible streaming server, and an agent benchmark that
measures the whole thing against vLLM and a hosted API on one frozen task suite. It was built to
find out how much a good attention kernel is worth to end-to-end serving. The answer, measured
rather than assumed, is **about 1.5%** — and the benchmark that says so is the most useful thing
here.

Attention runs on [flash-attention-cuda](https://github.com/shrvan30/flash-attention-cuda), a
from-scratch batched multi-head causal FlashAttention implementation with separate prefill and
decode kernels. Every other operation stays plain PyTorch; the kernel is the point, not a
re-implementation of the framework.

![architecture](docs/architecture.svg)

## The headline

20 fixed tasks, one ReAct agent, temperature 0, streaming on every backend. The only thing that
changes between rows is `base_url` and the model. RTX 3090, driver 580.126.09, CUDA 13.0,
PyTorch 2.11.0+cu130.

| backend | model | success | calls/task | TTFT p50 | decode | task p50 | cost/task |
| :-- | :-- | --: | --: | --: | --: | --: | --: |
| flashstack | Qwen2.5-0.5B-Instruct | 3/20 | 6.8 | 101 ms | 39.0 tok/s | 9.4 s | $0.00092 |
| vLLM 0.26 | Qwen2.5-0.5B-Instruct | 2/20 | 3.7 | 24 ms | 273.8 tok/s | 0.5 s | $0.00007 |
| hosted anchor | llama-3.3-70b-versatile | 20/20 | 3.0 | 234 ms | 523.4 tok/s | 1.3 s | $0.00137 |

Full tables, per-tier breakdown and metric provenance: [bench/results/report.md](bench/results/report.md).

**Read the success column carefully.** It is a property of the *model*, not the serving stack.
flashstack and vLLM run identical weights and agree within one task — which is exactly what
licenses reading the latency and throughput columns as properties of the stack. The hosted row
is a much larger model and is a latency and cost anchor only, not an apples-to-apples comparison.

TTFT and decode throughput are **client-side stream measurements on every row**. flashstack can
report its own server-side TTFT and the other two cannot, so publishing it would have put two
different definitions in one column; it is kept as a labelled cross-check instead. The gap
between them — 101.4 ms client against 66.5 ms server — is the measured cost of SSE framing and
transport, and it is reported rather than absorbed.

## Where the 7x decode gap comes from

Three causes get offered for a gap like this. Only one survives the numbers.

| cause | contribution |
| :-- | :-- |
| **Host dispatch overhead** | sufficient alone — a 16.7x ceiling against a 7.0x gap |
| **Continuous batching** | **0x** — this workload is strictly sequential |
| **Attention kernel quality** | **≤ 1.5%** of decode wall clock |

An Nsight Systems trace ([docs/profiles/decode_dispatch.md](docs/profiles/decode_dispatch.md))
measures the GPU **6% busy and 94% idle** while decoding: ~1,388 GPU operations per token, each
running ~2.8 µs, separated by ~79 µs of host-side gap — Python attribute lookups, tensor
allocation, argument checking, the launch call itself. If dispatch were free, flashstack would
decode at ~651 tok/s, which is 2.4x *beyond* vLLM's measured rate. Dispatch alone more than
accounts for the entire gap.

Continuous batching contributes nothing **here** because the agent never has two requests in
flight, so there is never a second request to admit. It is a real vLLM advantage under
concurrent load, and citing it for this measurement would borrow an explanation from a workload
that was not run.

Attention costs 0.38 ms of a 25.6 ms token. An *infinitely fast* attention kernel would take
flashstack from 39.0 to 39.6 tok/s.

That is the finding, and it is the opposite of the project's starting assumption. The kernel
work is real, correct and measured — and the serving gap it sits inside is dominated by
everything around it. The fixes that would move the number are CUDA graphs, operator fusion and
paged KV: all attack the launch count, none is about attention.

## Quickstart

**Server**, three commands:

```bash
pip install -e .
TORCH_CUDA_ARCH_LIST=8.6 pip install -e ../flash-attention-cuda --no-build-isolation
python -m uvicorn server.app:app --port 8000
```

Then point any OpenAI client at it — it speaks `/v1/chat/completions` with SSE streaming:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
for chunk in client.chat.completions.create(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    messages=[{"role": "user", "content": "Explain KV caching in one sentence."}],
    stream=True,
):
    print(chunk.choices[0].delta.content or "", end="")
```

**Benchmark**, one command:

```bash
python -m bench.run --backend flashstack --base-url http://localhost:8000/v1 \
    --model Qwen/Qwen2.5-0.5B-Instruct --stream
```

Then `python -m bench.report` to combine every `bench/results/<backend>.json` into the table,
the chart and the gap attribution. Always pass `--stream`: without it the harness can only
report whole-call wall clock, labels it as such, and the report refuses to present it as
comparable.

## What is here

| layer | what it does |
| :-- | :-- |
| `engine/` | model runners (GPT-2, Qwen2.5-0.5B), preallocated fp16 KV cache, sampling |
| `server/` | FastAPI OpenAI-compatible server, SSE streaming, static batching (≤4 requests / 25 ms window) |
| `agent/` | ReAct agent with calculator, BM25 doc search and mock API tools; 20 frozen tasks |
| `bench/` | three-backend harness, metric provenance, report and chart generation |

The engine deliberately keeps every non-attention operation in eager PyTorch. RoPE, GQA head
expansion, layernorm and the MLP are all stock ops. That is what makes the dispatch measurement
above meaningful — and it is also why the number is what it is.

## Requirements

- NVIDIA GPU with compute capability 8.6 (developed on an RTX 3090)
- CUDA toolkit with `nvcc` on `PATH`, matching the CUDA your PyTorch was built against
- Python 3.10+

## Install

```bash
pip install -e .
```

The attention kernels are a separate, source-built CUDA extension and are **not** a hard
dependency — that keeps the CPU test suite installable on a machine without a toolchain:

```bash
pip install "flashattn_cuda @ git+https://github.com/shrvan30/flash-attention-cuda"
```

For development against a local checkout, use an editable install from the sibling directory:

```bash
git clone https://github.com/shrvan30/flash-attention-cuda
TORCH_CUDA_ARCH_LIST=8.6 pip install -e ./flash-attention-cuda --no-build-isolation
```

`--no-build-isolation` matters: the extension links against the torch already in the
environment, and an isolated build would compile it against a second, ABI-incompatible copy.

## Tests

```bash
pip install -e ".[dev]"
pytest              # GPU tests skip automatically without a device or the extension
pytest -m gpu       # GPU tests only
ruff check .
```

## Further reading

- [docs/writeup.md](docs/writeup.md) — the whole story, from a CUDA kernel to an agent
- [bench/results/report.md](bench/results/report.md) — full benchmark tables and provenance
- [docs/profiles/decode_dispatch.md](docs/profiles/decode_dispatch.md) — the dispatch measurement
- [kernel benchmarks](https://github.com/shrvan30/flash-attention-cuda/blob/main/docs/benchmarks.md) — prefill and decode kernel numbers

## License

MIT — see [LICENSE](LICENSE).
