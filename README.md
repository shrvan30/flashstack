<p align="center">
  <img src="docs/images/banner.png" width="900" alt="FlashStack">
</p>

<h1 align="center">⚡ FlashStack</h1>

<p align="center">
  <b>A small AI inference engine built from scratch, powered by a custom CUDA FlashAttention kernel.</b>
</p>

<p align="center">
  Learn how ChatGPT generates answers, how GPUs make AI fast, and where the time actually goes.
</p>

<p align="center">
  <a href="https://github.com/shrvan30/flashstack/actions/workflows/ci.yml"><img src="https://github.com/shrvan30/flashstack/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://github.com/shrvan30/flashstack/actions/workflows/gpu-tests.yml"><img src="https://github.com/shrvan30/flashstack/actions/workflows/gpu-tests.yml/badge.svg" alt="GPU Tests"></a>
  <a href="https://github.com/shrvan30/flashstack/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License: MIT"></a>
  <img src="https://img.shields.io/badge/python-3.10%2B-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/CUDA-12.x-success.svg" alt="CUDA">
  <img src="https://img.shields.io/badge/PyTorch-2.x-orange.svg" alt="PyTorch">
</p>

---

## 📚 Documentation

| Resource | Description |
|---|---|
| [`docs/writeup.md`](https://github.com/shrvan30/flashstack/blob/main/docs/writeup.md) | Complete explanation, from CUDA kernel to inference engine |
| [`docs/architecture.md`](https://github.com/shrvan30/flashstack/blob/main/docs/architecture.md) | System architecture |
| [`docs/engine.md`](https://github.com/shrvan30/flashstack/blob/main/docs/engine.md) | Inference engine internals |
| [`docs/server.md`](https://github.com/shrvan30/flashstack/blob/main/docs/server.md) | OpenAI-compatible API server |
| [`docs/kernel.md`](https://github.com/shrvan30/flashstack/blob/main/docs/kernel.md) | FlashAttention kernel implementation |
| [`docs/benchmark.md`](https://github.com/shrvan30/flashstack/blob/main/docs/benchmark.md) | Benchmark methodology |
| [`docs/profiles/decode_dispatch.md`](https://github.com/shrvan30/flashstack/blob/main/docs/profiles/decode_dispatch.md) | Nsight Systems profiling and dispatch analysis |
| [`docs/faq.md`](https://github.com/shrvan30/flashstack/blob/main/docs/faq.md) | Frequently asked questions |

## 🔗 Important Links

| Link | URL |
|---|---|
| Repository | https://github.com/shrvan30/flashstack |
| FlashAttention CUDA kernel | https://github.com/shrvan30/flash-attention-cuda |
| Benchmark report | https://github.com/shrvan30/flashstack/blob/main/bench/results/report.md |
| Performance charts | https://github.com/shrvan30/flashstack/tree/main/bench/results |
| System write-up | https://github.com/shrvan30/flashstack/blob/main/docs/writeup.md |
| Architecture | https://github.com/shrvan30/flashstack/blob/main/docs/architecture.md |
| Dispatch analysis | https://github.com/shrvan30/flashstack/blob/main/docs/profiles/decode_dispatch.md |
| Kernel benchmarks | https://github.com/shrvan30/flash-attention-cuda/blob/main/docs/benchmarks.md |
| License | https://github.com/shrvan30/flashstack/blob/main/LICENSE |

---

## 📖 Table of Contents

- [What is FlashStack?](#-what-is-flashstack)
- [Why did I build it?](#-why-did-i-build-this)
- [How does ChatGPT answer a question?](#-how-does-chatgpt-answer-a-question)
- [What is FlashAttention?](#-what-is-flashattention)
- [How FlashStack works](#️-how-flashstack-works)
- [Project structure](#-project-structure)
- [Features](#-features)
- [Benchmark results](#-benchmark-results)
- [Installation](#-installation)
- [Running the server](#-running-the-server)
- [Running the benchmark](#-running-benchmarks)
- [Continuous integration](#-continuous-integration-ci)
- [Technologies used](#-technologies-used)
- [Future improvements](#-future-improvements)
- [Related research](#-related-research)
- [License](#-license)

---

## 🤖 What is FlashStack?

Imagine asking ChatGPT:

> **"What is the capital of India?"**

The AI doesn't magically know the answer. It performs **thousands of mathematical operations** on the GPU.

FlashStack is a project that shows **how those operations happen.**

Instead of using the default attention operation inside PyTorch, FlashStack uses a **CUDA FlashAttention kernel written completely from scratch.** Everything else is kept deliberately simple using PyTorch, so you can see exactly where the time goes.

---

## 🎯 Why did I build this?

Modern AI models are very fast. But **why?**

Is it because of:

- Faster GPUs?
- Better CUDA kernels?
- Better software?
- Better batching?

I wanted to measure it instead of guessing.

So I built:

✅ My own FlashAttention CUDA kernel
↓
✅ A tiny inference engine
↓
✅ An OpenAI-compatible server
↓
✅ An AI agent benchmark

and compared everything against **vLLM** and a **hosted API** using exactly the same tasks.

---

## 🧠 How does ChatGPT answer a question?

Suppose you ask:

```
Explain gravity.
```

The model performs these steps:

```
You
 │
 ▼
Tokenization
 │
 ▼
Embedding
 │
 ▼
Transformer Layers
 │
 ├── Attention
 ├── MLP
 ├── LayerNorm
 └── Residual Connections
 │
 ▼
Output Tokens
 │
 ▼
"Gravity is..."
```

FlashStack changes only one part:

```
Attention
```

Instead of using PyTorch's implementation, it uses:

```
My CUDA FlashAttention Kernel
```

Everything else stays unchanged.

---

## 🚀 What is FlashAttention?

Imagine reading a book. Suppose the current word is:

```
dog
```

To understand it, you may need to remember:

```
The  small  brown  dog  ran  fast
```

The AI also has to remember previous words. This process is called **attention**.

The problem is that re-reading all previous words becomes slow.

FlashAttention is a smarter algorithm that:

- reads fewer values from GPU memory
- keeps working data in shared memory
- avoids materializing huge intermediate matrices
- computes softmax in a single streaming pass

which makes attention much faster and much lighter on memory.

---

## 🖥️ How FlashStack Works

```
User Question
 │
 ▼
FastAPI Server
 │
 ▼
Inference Engine
 │
 ▼
Transformer Model
 │
 ▼
FlashAttention CUDA Kernel
 │
 ▼
GPU
 │
 ▼
Generated Text
```

---

## 📂 Project Structure

```
flashstack/
│
├── engine/          Model loading, KV cache, sampling, transformer execution
├── server/          OpenAI-compatible API, streaming, FastAPI app
├── agent/           Calculator tool, BM25 search, mock API tool, agent tasks
├── bench/           Benchmark runner, reports, charts
└── docs/            Architecture, kernel, benchmarks, profiling, FAQ
```

Documentation layout:

```
docs/
│
├── architecture.md          Complete architecture
├── writeup.md               Entire project explanation
├── benchmark.md             Benchmark methodology
├── engine.md                Inference engine internals
├── server.md                OpenAI-compatible API
├── kernel.md                CUDA FlashAttention
├── faq.md                   Frequently asked questions
└── profiles/
    └── decode_dispatch.md   Nsight Systems dispatch analysis
```

Benchmark layout:

```
bench/
│
├── run.py
├── report.py
├── tasks/
└── results/
    ├── flashstack.json
    ├── vllm.json
    ├── hosted.json
    ├── report.md
    └── plots/
```

---

## ✨ Features

### ✅ FlashAttention CUDA kernel

Written completely from scratch. Supports:

- FP16
- Online (streaming) softmax
- Shared-memory tiling
- Causal masking
- Batched multi-head attention
- Separate prefill and decode kernels

### ✅ OpenAI-compatible API

Works like the OpenAI Chat API:

```
POST /v1/chat/completions
```

Supports streaming, multi-turn messages, temperature, and max tokens.

### ✅ AI agent

Includes a calculator tool, a BM25 search tool, and a mock API tool, used to build realistic agent benchmark tasks.

### ✅ Benchmark system

Measures time to first token, decode speed, total time, and success rate.

---

## 📊 Benchmark Results

| Backend | Decode speed |
|---|---|
| FlashStack | 39 tokens/sec |
| vLLM | 273 tokens/sec |
| Hosted API | 523 tokens/sec |

Full report: [`bench/results/report.md`](https://github.com/shrvan30/flashstack/blob/main/bench/results/report.md)

### The interesting discovery

I originally believed:

> Better FlashAttention = huge speedup

The benchmark showed something different.

Most of the time was **not spent inside FlashAttention.** Nsight Systems profiling showed the GPU was only about **6% busy** during decode — it spent the rest of the time waiting on the CPU to launch thousands of tiny kernels. That host dispatch overhead alone accounts for the **7.0× decode gap** against vLLM.

So the biggest remaining wins are:

- CUDA Graphs
- Operator fusion
- Better scheduling and batching

rather than further micro-optimizing the attention kernel.

Details: [`docs/profiles/decode_dispatch.md`](https://github.com/shrvan30/flashstack/blob/main/docs/profiles/decode_dispatch.md)

---

## ⚙️ Installation

Clone the project:

```bash
git clone https://github.com/shrvan30/flashstack
cd flashstack
```

Install:

```bash
pip install -e .
```

Install the FlashAttention CUDA kernel:

```bash
git clone https://github.com/shrvan30/flash-attention-cuda

TORCH_CUDA_ARCH_LIST=8.6 \
pip install -e ./flash-attention-cuda --no-build-isolation
```

> Set `TORCH_CUDA_ARCH_LIST` to match your GPU (8.6 = RTX 3090 / A10, 8.9 = RTX 4090, 9.0 = H100).

---

## ▶ Running the Server

```bash
python -m uvicorn server.app:app --port 8000
```

### Example client

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed",
)

for chunk in client.chat.completions.create(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    messages=[{"role": "user", "content": "Explain AI"}],
    stream=True,
):
    print(chunk.choices[0].delta.content or "", end="")
```

---

## 🧪 Running Benchmarks

```bash
python -m bench.run \
  --backend flashstack \
  --base-url http://localhost:8000/v1 \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --stream
```

Generate the report:

```bash
python -m bench.report
```

---

## 🔬 Continuous Integration (CI)

| Workflow | Purpose |
|---|---|
| [`ci.yml`](https://github.com/shrvan30/flashstack/blob/main/.github/workflows/ci.yml) | Unit tests, import checks, packaging |
| [`gpu-tests.yml`](https://github.com/shrvan30/flashstack/blob/main/.github/workflows/gpu-tests.yml) | Kernel correctness tests on GPU runners |
| [`lint.yml`](https://github.com/shrvan30/flashstack/blob/main/.github/workflows/lint.yml) | Formatting and static checks |
| [`docs.yml`](https://github.com/shrvan30/flashstack/blob/main/.github/workflows/docs.yml) | Documentation build |

---

## 🛠 Technologies Used

```
FlashStack
│
├── CUDA / C++
├── PyTorch
├── Transformers (HuggingFace)
├── FastAPI + Uvicorn
├── OpenAI SDK
├── NumPy
├── BM25
├── FlashAttention CUDA (custom kernel)
└── Nsight Systems
```

---

## 📈 Learning Roadmap

```
README
   ▼
Architecture
   ▼
Engine
   ▼
CUDA FlashAttention
   ▼
Benchmark
   ▼
Nsight Profiling
   ▼
Future Work
```

After working through this project you will understand how transformers generate text, what attention does, why GPUs matter, how CUDA kernels and shared memory work, how streaming inference APIs are built, and how to benchmark and profile an inference stack.

---

## 🚀 Future Improvements

- Continuous batching
- CUDA Graphs
- Operator fusion
- Paged KV cache
- Tensor parallelism
- Multi-GPU support

---

## 📖 Related Research

- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)
- [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691)
- [NVIDIA CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [NVIDIA Nsight Systems Documentation](https://docs.nvidia.com/nsight-systems/)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [HuggingFace Transformers Documentation](https://huggingface.co/docs/transformers)
- [vLLM Documentation](https://docs.vllm.ai/)
- [OpenAI Chat Completions API Reference](https://platform.openai.com/docs/api-reference/chat)

---

## ❤️ Acknowledgements

Thanks to the NVIDIA CUDA, PyTorch, HuggingFace, and vLLM communities, and to the authors of the FlashAttention papers.

---

## 📜 License

MIT License — see [LICENSE](https://github.com/shrvan30/flashstack/blob/main/LICENSE).

---

<p align="center">
  ⭐ If this project helped you, star it on GitHub. Happy learning 🚀
</p>
