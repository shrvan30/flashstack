"""Side-by-side generation: stock GPT-2 attention versus the flashattn_cuda kernel.

This is the "my kernel runs a real model" proof. It generates greedily from both
models and prints the two continuations next to each other, so the interesting
result is that they are the same text.

The throughput figures are *not* a kernel benchmark. Both models run with
`use_cache=False`, because the Phase 2 patch is prefill-only: every step
recomputes the whole sequence. That makes the comparison fair — the two models do
identical work — but it makes both numbers far slower than cached decoding, and
the tokens/s therefore measures full-sequence prefill throughput, not decode.

    python examples/generate.py --prompt "The capital of France is"
"""

from __future__ import annotations

import argparse
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from engine.patching import patch_gpt2


def greedy_generate(model, input_ids: torch.Tensor, max_new_tokens: int):
    """Greedy decode without a KV cache; returns (new tokens, seconds elapsed)."""
    sequence = input_ids.clone()

    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model(sequence, use_cache=False).logits[:, -1, :]
        sequence = torch.cat([sequence, logits.argmax(-1, keepdim=True)], dim=1)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    return sequence[0, input_ids.shape[1] :], elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gpt2", help="HuggingFace model id")
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument(
        "--warmup",
        type=int,
        default=2,
        help="untimed forward passes before measuring, to exclude one-off setup",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("this demo needs a CUDA device")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    input_ids = tokenizer(args.prompt, return_tensors="pt").input_ids.cuda()

    def load():
        return (
            AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16)
            .cuda()
            .eval()
        )

    stock = load()
    patched = patch_gpt2(load())

    for model in (stock, patched):
        for _ in range(args.warmup):
            with torch.no_grad():
                model(input_ids, use_cache=False)

    stock_tokens, stock_seconds = greedy_generate(stock, input_ids, args.max_new_tokens)
    patched_tokens, patched_seconds = greedy_generate(
        patched, input_ids, args.max_new_tokens
    )

    stock_text = tokenizer.decode(stock_tokens)
    patched_text = tokenizer.decode(patched_tokens)
    identical = torch.equal(stock_tokens, patched_tokens)

    print(f"model            : {args.model}")
    print(f"device           : {torch.cuda.get_device_name(0)}")
    print(f"prompt           : {args.prompt!r}")
    print(f"new tokens       : {args.max_new_tokens} (greedy, use_cache=False)")
    print()
    print("--- stock attention -------------------------------------------------")
    print(stock_text)
    print()
    print("--- flashattn_cuda prefill ------------------------------------------")
    print(patched_text)
    print()
    print(f"identical output : {identical}")
    print(
        f"stock            : {stock_seconds * 1e3:8.1f} ms  "
        f"{args.max_new_tokens / stock_seconds:7.1f} tok/s"
    )
    print(
        f"flashattn_cuda   : {patched_seconds * 1e3:8.1f} ms  "
        f"{args.max_new_tokens / patched_seconds:7.1f} tok/s"
    )
    print()
    print(
        "Throughput here is full-sequence prefill per step, not cached decoding; "
        "the KV cache and the decode kernel arrive in Phase 3."
    )


if __name__ == "__main__":
    main()
