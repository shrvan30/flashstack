# 🧱 From a CUDA Kernel to an Agent

> How this project was built, one layer at a time — and the surprise at the end.

I built this in four stages, bottom to top. Each layer sits on the one below.

```
Layer 4:  Agent Benchmark   ← measures everything
Layer 3:  Server            ← talks to the outside world
Layer 2:  Engine            ← runs the AI model
Layer 1:  CUDA Kernel       ← does the maths on the GPU
```

I was chasing one question: **how much is a really good attention kernel worth
to a real AI system?**

The honest answer turned out to be *much smaller* than the effort I spent
building one. That disappointing result is the most valuable thing here.

---

## 🔬 Layer 1: The Kernel

A **kernel** is a small program that runs on the graphics card. Mine does
*attention* — the step where the AI looks back at earlier words to understand
the current one.

Mine handles words 64 numbers wide, stores them in a compact 16-bit format to
save memory, but adds them up in a more precise 32-bit format so small errors
do not pile up.

> **Key idea:** store cheap, calculate carefully. Cheap storage saves memory
> bandwidth; careful arithmetic protects accuracy. You want both.

### Two jobs, two kernels

AI generates text in two very different phases, so I wrote two kernels.

**Prefill — reading your question.** All the words arrive at once, so there is
plenty to do in parallel. Each group of GPU threads takes one chunk of
questions and streams the earlier words past it, keeping a running maximum and
running total in the fastest memory available. It never builds the giant
score table that the textbook version does.

It also skips work it does not need. A word can only look *backwards*, never
forwards. So instead of computing the future half and then throwing it away
with a mask, the kernel never computes it.

> **Teaching point:** the fastest work is work you never do. Ask students to
> spot which half of a triangle diagram is wasted before you tell them.

**Decode — writing the answer, one word at a time.** Now there is exactly
*one* new word looking back at possibly thousands of old ones. One word means
almost no parallel work — a GPU with thousands of workers has one thing to do.

The fix: split the *old* words into chunks and let many worker groups each
handle a chunk at the same time. Each returns a partial result. Then a final
step stitches the partials together into the correct answer.

```
        one new word
             │
   ┌────┬────┼────┬────┐
   ▼    ▼    ▼    ▼    ▼
 chunk chunk chunk chunk chunk   ← all at once
   └────┴────┼────┴────┘
             ▼
      stitch together
```

### The setting that mattered most

How big should each chunk be? This turned out to be the single biggest lever
in the whole kernel.

My card has **82 processors**. With chunks fixed at 512 old words and a
context of 1024, one sequence produces only **24 chunks of work**:

```
82 processors, 24 jobs
██████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  three-quarters idle
```

Using smaller chunks makes more jobs — enough to keep every processor at least
twice as busy. That change alone made decoding **~3× faster**.

Here is the beautiful part: **the answer does not change.** The stitching
maths is built so that any chunk size gives a numerically identical result.
The test suite checks exactly this across every size in the range. So chunk
size is purely about *scheduling*, never about correctness — a free 3×.

### Comparing against the professionals

Against `flash-attn` 2.8.3, the industry-standard library:

| Comparison | Result |
| :-- | :-- |
| vs. my own first version | **62× faster** |
| vs. flash-attn (prefill) | **2.9–3.7× slower** |
| Raw throughput | 10.4–11.7 TFLOP/s |
| Share of the card's peak | 29–32% of 36.2 TFLOP/s |

So: a huge win over where I started, and still well behind the professionals.
**Why?**

Not because my code is sloppy. Because of a **structural** limit.

Every multiplication in my kernel runs on the card's general-purpose CUDA
cores, and the numbers arrive through shared memory. That path needs about
**1.75 bytes delivered per multiply**, but the processor can only sustain
about **1.0**. The delivery pipe is the bottleneck, not the maths.

```
Needs:     1.75 bytes per calculation
Supplies:  1.00 bytes per calculation
           ────────────────────────────
Ceiling:   ~57% of peak — before any other overhead
```

flash-attn avoids this by using **tensor cores** — specialised hardware that
does matrix maths directly, pulling numbers from registers instead of shared
memory. Different pipe, different ceiling.

**The lesson: no amount of polishing my kernel reaches their speed. It needs
a different design.** Knowing *which* kind of problem you have — a tuning
problem or a design problem — saves months.

### A note on honesty

That 1.75-vs-1.0 explanation is a **model**, not a measurement. Proving which
part of the chip truly runs out of capacity needs special hardware counters,
and the rented machine I worked on blocks access to them.

I could have estimated those numbers. They would have looked authoritative.
Instead they are **absent**, and listed as open work.

> **Teaching point:** "I do not know yet" is a legitimate entry in a technical
> report. Nothing in this project claims a measurement it did not make.

---

## ⚙️ Layer 2 and 3: Engine and Server

The **engine** runs real models — GPT-2 and Qwen2.5-0.5B-Instruct. Attention
runs on my kernel; everything else uses ordinary PyTorch, deliberately
unoptimised. That is what makes the final measurement meaningful: only one
part is special, so only one part can be credited.

The **server** is a FastAPI app that speaks the same language as the OpenAI
API, streams words out as they are generated, and can group up to four
requests arriving within 25 milliseconds.

### How do you know a rewritten kernel is correct?

This is the part worth teaching, because "it looks about right" is how bugs
survive.

The rule: **with randomness switched off, my version must produce the exact
same words as the original, in the exact same order.**

There is one narrow exception. Sometimes the model's top two choices are
almost perfectly tied — within 0.01. At a genuine tie, the tiniest rounding
difference can tip the choice, and that is normal, not a bug. So divergence is
allowed *only after* such a tie.

```
Same words, same order          ✅ pass
Different word, after a tie     ✅ pass  (genuine coin-flip)
Different word, no tie          ❌ bug
```

> **Teaching point:** a good test says exactly what "correct" means *before*
> you run it — including which failures are acceptable and why.

---

## 📐 Layer 4: The Measurement

The benchmark runs an agent through **20 fixed tasks** about a **made-up
company**. Fictional on purpose: the model cannot have memorised the answers,
so it must actually use its tools. Every answer is checkable offline.

Same tasks, same agent, randomness off. Only the engine changes.

### The bug I found before publishing

My harness had a rule: *use each engine's own reported timing where
available*. Sounds reasonable. It was not.

My engine reports its own internal timing. vLLM and the hosted API do not. So
one column would have mixed two different definitions with nothing saying so:

```
flashstack   66.5 ms   ← server's internal stopwatch
vLLM         24.0 ms   ← time until the word arrived
hosted      234.0 ms   ← time until the word arrived
```

My engine would have looked **a third faster than it is**, and no reader could
have seen why.

The fix: publish the *arrival* time for everyone, because it is the only
definition all three can supply. Each server's own figure is carried alongside
as a clearly labelled cross-check. Every number now travels with a label
saying where it came from, and the report refuses to print a mixed column.

> **Teaching point:** the bug was not in the maths. It was in the definition.
> Those are the ones that survive review.

### The result

vLLM decodes at **273.8 words/sec**. Mine manages **39.0**. A **7× gap** on
identical weights and identical hardware. Where does it go?

**Cause 1 — the GPU is waiting. Guilty, and sufficient alone.**
The graphics card is **6% busy and 94% idle** while generating. Each word
needs ~1,388 tiny jobs of ~2.8 microseconds each, separated by ~79
microseconds of Python deciding what to send next. Remove the waiting and the
ceiling is ~651 words/sec — **2.4× beyond vLLM**. This one cause more than
covers the whole gap.

**Cause 2 — missing batching. Exactly zero.**
vLLM can serve several requests at once. My agent asks one question at a time,
so there was never a second request to serve. It is a real vLLM advantage
under real load, and it explains *none* of this measurement. Citing it would
mean borrowing an explanation from a test I did not run.

**Cause 3 — my kernel. At most 1.5%.**
0.38 ms of attention inside a 25.6 ms word. An infinitely fast attention
kernel takes 39.0 words/sec to **39.6**.

> **The finding:** this benchmark measures how fast Python can issue
> instructions — not how good my kernel is. The thing the entire project is
> built around controls about **one and a half percent** of the number it is
> usually assumed to control.

---

## 🔭 What next, and what I learned

The fixes that would actually move the number all attack the same thing —
**the number of instructions sent** — and none of them is about attention:
CUDA graphs (record a sequence once, replay it), operator fusion (fewer,
bigger jobs), paged KV cache (fit more work in memory).

The bigger lesson is about discipline. **Three separate times, the careful
thing and the convenient thing pointed in opposite directions:**

| The convenient thing | The careful thing |
| :-- | :-- |
| Publish my flattering internal timing | Publish the timing all three engines can supply |
| List batching as a cause — it is true in general | Leave it out — it did nothing *here* |
| Estimate the missing hardware figures | Leave them absent, marked as open work |

Each convenient choice would have produced a more impressive document and a
less true one.

The reason to build a measurement harness carefully is precisely this: **it
makes the inconvenient answer as easy to publish as the convenient one.**

And here, the inconvenient answer was the interesting one.
