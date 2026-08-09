# 📊 Benchmark: Three Engines, One Test

> What happened when I raced my own AI engine against two professional ones —
> and why the result surprised me.

---

## 🎯 The setup

I gave **20 tasks** to an AI agent (a program that uses tools like a
calculator and a search function to answer questions), then ran those exact
same 20 tasks through three different engines.

```
Same 20 tasks
Same AI model
Same settings
      │
      ├──▶ flashstack     (my engine, built from scratch)
      ├──▶ vLLM           (a professional open-source engine)
      └──▶ hosted API     (a big model in a data centre)
```

The **only** thing that changed between runs was which engine answered. That
is what makes it a fair race.

| | |
| :-- | :-- |
| Graphics card | NVIDIA RTX 3090 |
| Driver / CUDA / PyTorch | 580.126.09 / 13.0 / 2.11.0+cu130 |
| flashstack commit | `224f69c` |
| kernel commit | `91c091e` |
| Tasks | 20 questions about a made-up company, Halden Systems |

---

## 📏 Reading the numbers

Think of ordering food at a restaurant:

| Term | Meaning | Restaurant version |
| :-- | :-- | :-- |
| **TTFT** | Time To First Token — delay before the *first word* | When the first dish arrives |
| **Decode speed** | Words per second after that | How fast the rest comes out |
| **Task time** | Time to finish one whole task | The full meal |
| **Calls per task** | How many times the AI was asked | Trips the waiter made |

Also: **p50** is the typical case (half the runs were faster). **p95** is the
bad day — 95 runs in 100 beat it. p95 matters because averages hide
disasters: nine tasks at 1 second and one at 60 averages out fine, but a real
user would notice.

---

## 🏁 Results

| Engine | Correct | First word | Speed | Typical task | Bad-day task |
| :-- | --: | --: | --: | --: | --: |
| **flashstack** (mine) | 3 / 20 | 101 ms | 39 words/sec | 9.4 s | 13.6 s |
| **vLLM** | 2 / 20 | 24 ms | 274 words/sec | 0.5 s | 1.9 s |
| **Hosted API** | 20 / 20 | 234 ms | 523 words/sec | 1.3 s | 3.3 s |

Two stories here.

**Speed:** my engine is about **7× slower** than vLLM — on the identical model
and the identical graphics card. That gap is the mystery this report solves.

**Correctness:** the hosted API got everything right, the other two almost
nothing. But my engine and vLLM scored roughly the same (3 vs 2), which is the
clue — see the last section.

### Cost per task

| flashstack | vLLM | Hosted API |
| --: | --: | --: |
| $0.00092 | $0.00007 | $0.00137 |

Same currency, **two different counting methods**. My engine and vLLM run on a
card I rent by the hour ($0.35/hr), so cost = time, like a taxi meter. The
hosted API charges per word ($0.59/$0.79 per million in/out), so cost = word
count, like a fixed price per kilometre.

### Which tasks were hard?

| Engine | Easy (1 tool) | Medium (2 tools) | Hard (4+ tools) |
| :-- | --: | --: | --: |
| flashstack | 1 / 8 | 2 / 8 | 0 / 4 |
| vLLM | 2 / 8 | 0 / 8 | 0 / 4 |
| Hosted API | 8 / 8 | 8 / 8 | 4 / 4 |

Both small-model engines scored **0 / 4** on hard tasks. The tiny model could
sometimes manage one step; chaining four without losing track, never. That is
a limit of the model's brain, not the engine running it.

---

## ⏱️ Measuring honestly

There are two ways to time "how long until the first word":

```
 Server's stopwatch          Client's stopwatch
 ─────────────────           ──────────────────
 Stops: word created         Stops: word arrives on my screen
      = 66.5 ms                   = 101.4 ms
```

The **34.9 ms** difference is real time spent packing the word into JSON,
wrapping it in a streaming message and pushing it through the socket.

The smaller number would flatter my engine. I published the bigger one,
because it is what a user actually waits for, and because it is the only
measurement all three engines can supply — vLLM and the hosted API do not
report internal timings. **Every published number is measured the same way on
all three engines.**

Same principle for extra calls:

- **Parse retries count as work.** The AI replied with something garbled, the
  agent said "try again," the GPU genuinely ran twice. An engine that formats
  badly *should* look more expensive. It is.
- **Throttle waits do not.** The hosted service refused *before* computing
  anything. Counting it would charge my rented GPU's hourly rate for sleeping.

---

## 🔍 The big question: where did the 7× go?

Mine: 39 words/sec. vLLM: 274. Same model, same card. Three suspects — only
one is guilty.

### 🚨 Suspect 1: the GPU is waiting around — **GUILTY**

NVIDIA Nsight Systems recorded what the card actually did while generating:

```
GPU busy:  ██ 6%
GPU idle:  ██████████████████████████████████ 94%
```

For every word, the engine fires ~**1,388 tiny jobs** at the GPU. Each runs
~**2.8 microseconds** — but there is a **79 microsecond gap** between them
while Python works out what to send next.

```
Chef (GPU):    cooks a dish in 3 seconds       ⚡
Waiter (CPU):  79 seconds to bring the next order  🐌
```

The chef is not the problem. With no waiting, my engine would hit **651
words/sec** — **16.7×** its real speed and **2.4× past vLLM**. This one cause
more than covers the whole 7× gap by itself.

### 🤔 Suspect 2: batching — **NOT GUILTY (here)**

vLLM can slot new questions in while answering an existing one, like a bus
picking up passengers anywhere instead of waiting at the depot. Great feature
— but my agent asks one question at a time and waits, so there was never a
second passenger. It is a real advantage of vLLM for a busy website, and it
explains **none** of the gap measured here. Blaming it would mean borrowing an
explanation from a test I never ran.

### 😬 Suspect 3: my attention kernel — **BARELY INVOLVED**

The uncomfortable one. This whole project is built around a FlashAttention
CUDA kernel I wrote by hand, so how much is its fault?

```
15.9 µs per layer per word  ×  24 layers  =  0.38 ms per word
0.38 ms  ÷  25.6 ms total per word        =  about 1.5%
```

Making attention *infinitely fast* would take me from 39 words/sec to **39.6**.

### 📋 Verdict

| Suspect | Contribution |
| :-- | :-- |
| GPU waiting on Python | Enough alone (16.7× ceiling vs a 7× gap) |
| Missing batching | 0× — the workload is sequential |
| Attention kernel quality | 1.5% at most |

**This benchmark measures how fast Python can hand out instructions, not how
good my attention kernel is** — the exact opposite of what I expected. The
kernel is real, correct and measured; it just sits inside a system where
everything around it costs more. (In isolation it is fine: prefill reaches
about a third of the card's compute peak, decode 27% of memory bandwidth.
Respectable, and nearly irrelevant to the final score.)

### What would actually help

| Fix | What it does | Chef analogy |
| :-- | :-- | :-- |
| **CUDA Graphs** | Record the sequence once, replay it | Hand over the full menu upfront |
| **Operator Fusion** | Merge many small jobs into fewer big ones | Combine 10 orders into one |
| **Paged KV Cache** | Store memory efficiently, fit more work | A better-organised fridge |

All three attack the launch count. None is about attention. That was the
lesson.

---

## ⚠️ What the success column does *not* say

It is tempting to read 3-vs-2 as "my engine is smarter than vLLM." It is not.

**Success rate belongs to the model, not the engine.** Both ran the same small
model (Qwen2.5-0.5B, ~500 million parameters); the hosted API ran Llama 3.3 at
70 billion — roughly 140× larger. The engine is the delivery truck, the model
is the cargo. Two trucks carrying the same cargo deliver the same thing, so
3-vs-2 is just noise in the maths.

The column still earns its place: failed tasks burn every call they made, and
a *large* divergence on identical weights would signal a bug. 3 and 2 means
everything is behaving.

---

## 📌 Summary

1. My engine is 7× slower than vLLM on the same model and card.
2. The GPU was idle 94% of the time, waiting on Python.
3. That single fact more than explains the entire gap.
4. My hand-written attention kernel accounts for just 1.5% of the time.
5. **Measuring beats guessing — even when the measurement is not the answer
   you wanted.**

---

<p align="center">
  <a href="../../docs/writeup.md">Full write-up</a> ·
  <a href="../../docs/profiles/decode_dispatch.md">Dispatch profiling</a> ·
  <a href="../../docs/kernel.md">Kernel design</a> ·
  <a href="../../docs/benchmark.md">Benchmark methodology</a>
</p>
