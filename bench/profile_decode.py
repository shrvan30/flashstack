"""Run a fixed number of decode steps under a profiler capture range.

This exists to measure *why* decode is slow, not to make it faster. The engine
runs every layer as eager PyTorch ops, so a decode step issues on the order of
two hundred kernel launches, each with Python dispatch overhead in front of it.
The question this answers is how much of the wall clock the GPU is actually
busy — which is the number the backend comparison needs in order to attribute
its gap honestly.

Only the timed steps are captured: `torch.cuda.profiler.start()` opens the
capture range after warm-up, so model loading, the prompt prefill and CUDA
context setup stay out of the trace.

    nsys profile -t cuda --capture-range=cudaProfilerApi --capture-range-end=stop \
        -o docs/profiles/decode_steps python bench/profile_decode.py
"""

from __future__ import annotations

import argparse
import json
import time

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--prompt", default="Explain what a GPU kernel is.")
    parser.add_argument("--summary", default=None, help="write wall-clock JSON here")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("this needs a CUDA device")

    if "gpt2" in args.model.lower():
        from engine.models.gpt2 import GPT2Runner as Runner
    else:
        from engine.models.qwen25 import Qwen25Runner as Runner

    runner = Runner(args.model)

    if hasattr(runner, "apply_chat_template"):
        prompt_ids = runner.apply_chat_template([{"role": "user", "content": args.prompt}])
    else:
        prompt_ids = (
            runner.tokenizer(args.prompt, return_tensors="pt").input_ids[0].to(runner.device)
        )

    slot = runner.allocate()
    try:
        logits = runner.prefill(prompt_ids, slot)
        token = int(logits.argmax(-1))

        # Warm-up steps are outside the capture range: the first few steps pay
        # allocator growth and autotuning that the steady state does not.
        for _ in range(args.warmup):
            logits = runner.decode_step(
                torch.tensor([token], device=runner.device), [slot]
            )[0]
            token = int(logits.argmax(-1))

        torch.cuda.synchronize()
        torch.cuda.profiler.start()
        started = time.perf_counter()

        for _ in range(args.steps):
            logits = runner.decode_step(
                torch.tensor([token], device=runner.device), [slot]
            )[0]
            token = int(logits.argmax(-1))

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        torch.cuda.profiler.stop()
    finally:
        runner.free(slot)

    summary = {
        "model": args.model,
        "layers": runner.num_layers,
        "steps": args.steps,
        "warmup": args.warmup,
        "context_after": int(prompt_ids.shape[0]) + args.warmup + args.steps,
        "wall_s": elapsed,
        "ms_per_step": elapsed / args.steps * 1e3,
        "tokens_per_s": args.steps / elapsed,
    }
    print(json.dumps(summary, indent=2))
    if args.summary:
        with open(args.summary, "w") as handle:
            json.dump(summary, handle, indent=2)


if __name__ == "__main__":
    main()
