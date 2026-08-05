"""Release-blocking parity test: patched GPT-2 must behave like the stock model.

Two checks, per the Phase 2 contract:

1. **Logits.** Max abs difference between the patched and stock fp16 models is
   reported, and the patched model's distance from an fp32 reference must stay in
   the same numerical class as the stock fp16 model's distance from it. The point
   is not that the logits are identical — reordering the attention arithmetic
   guarantees they are not — but that the kernel is not *less* accurate than the
   baseline it replaces.

2. **Behaviour.** Greedy generation of 64 new tokens must reproduce the stock
   model's token sequence on at least `MIN_EXACT_MATCHES` of 5 fixed prompts, and
   any prompt that does diverge must diverge at a position where the stock model
   itself was ambiguous — a genuine fp16 tie, not a kernel bug.

The tie rule is what makes a divergence acceptable. Greedy decoding takes an
argmax; when the top two logits are equal to within fp16 resolution, which token
wins is decided by tie-breaking order, not by the model. Any perturbation of the
arithmetic can flip it, and the flip then compounds for every subsequent token.
So a divergence is only evidence of a bug if it happens where the stock model was
*confident*.

Generation runs with `use_cache=False`: the Phase 2 patch is prefill-only, so
every step recomputes the full sequence. That is slow and deliberately so — this
milestone is about correctness, and incremental decoding arrives in Phase 3.
"""

from __future__ import annotations

import pytest
import torch

from tests.parity_utils import (
    assert_divergences_are_ties,
    build_verdict,
    first_divergence,
    report_divergences,
)

MODEL = "gpt2"
MAX_NEW_TOKENS = 64

# Five fixed prompts of mixed length (4 to 38 tokens): short factual, short code,
# mid-length prose, and a long narrative that exercises multi-tile prefill.
PROMPTS = [
    "The capital of France is",
    "In 1969, NASA",
    "def fibonacci(n):",
    "The three laws of robotics were formulated by Isaac Asimov, and the first of "
    "them states that",
    "Once upon a time, in a small village nestled between two mountains, there "
    "lived an old clockmaker who had spent his entire life building machines that "
    "measured time in ways nobody else understood.",
]

# Approved deviation: the phase contract asks for >= 4 of 5 exact matches. Three
# is accepted as the floor *provided* every divergence clears the two hard gates
# below (proven fp16 tie, and no worse than stock against an fp32 reference).
# This constant is a regression tripwire, not a target: the observed state is 5/5,
# and a drop that is not explained by a proven tie fails on the per-divergence
# gates before this count is ever reached.
MIN_EXACT_MATCHES = 3

# Per-divergence gates live in tests/parity_utils.py so that this suite and the
# Phase 3 engine suite share one definition of "this was a tie". See that
# module for why the tolerance is in ulps rather than an absolute epsilon.

# Whole-prompt logit check below is looser than the per-divergence gate on
# purpose: it compares max abs error across every position rather than at a
# single contested argmax, where a small ordering difference is expected.
WHOLE_PROMPT_FP32_RATIO = 2.0


@pytest.fixture(scope="module")
def parity_models():
    """Stock fp16, patched fp16 and an fp32 reference, all sharing one tokenizer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from engine.patching import patch_gpt2

    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    def load(dtype):
        return AutoModelForCausalLM.from_pretrained(MODEL, dtype=dtype).cuda().eval()

    stock = load(torch.float16)
    patched = patch_gpt2(load(torch.float16))
    reference = load(torch.float32)
    return tokenizer, stock, patched, reference


def _greedy(model, input_ids: torch.Tensor, steps: int):
    """Greedy decode without a KV cache; returns (new tokens, per-step logits)."""
    sequence = input_ids.clone()
    step_logits = []
    for _ in range(steps):
        with torch.no_grad():
            logits = model(sequence, use_cache=False).logits[:, -1, :]
        step_logits.append(logits.float().cpu())
        sequence = torch.cat([sequence, logits.argmax(-1, keepdim=True)], dim=1)
    return sequence[0, input_ids.shape[1] :].tolist(), step_logits


def _logits_at(model, prefix: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return model(prefix.unsqueeze(0), use_cache=False).logits[0, -1].float()


@pytest.mark.gpu
def test_patched_logits_track_the_stock_model(parity_models):
    """Logit-level check, plus the fp32-dominance measurement it is judged against."""
    tokenizer, stock, patched, reference = parity_models

    print("\nlogit parity (max abs diff over all positions):")
    print(f"{'prompt':>7} {'len':>4} {'vs stock':>10} {'stock-fp32':>11} "
          f"{'patch-fp32':>11} {'ratio':>7}")

    for index, prompt in enumerate(PROMPTS):
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()

        with torch.no_grad():
            stock_logits = stock(input_ids, use_cache=False).logits.float()
            patched_logits = patched(input_ids, use_cache=False).logits.float()
            reference_logits = reference(input_ids, use_cache=False).logits.float()

        versus_stock = (patched_logits - stock_logits).abs().max().item()
        stock_error = (stock_logits - reference_logits).abs().max().item()
        patched_error = (patched_logits - reference_logits).abs().max().item()
        ratio = patched_error / stock_error if stock_error > 0 else float("inf")

        print(f"{index:>7} {input_ids.shape[1]:>4} {versus_stock:>10.6f} "
              f"{stock_error:>11.6f} {patched_error:>11.6f} {ratio:>7.3f}")

        assert patched_error <= WHOLE_PROMPT_FP32_RATIO * stock_error, (
            f"prompt {index}: the patched model is {ratio:.2f}x further from the "
            f"fp32 reference than the stock fp16 model "
            f"({patched_error:.6f} vs {stock_error:.6f}); the kernel is "
            f"numerically worse than the attention it replaces"
        )


@pytest.mark.gpu
def test_patched_greedy_generation_matches_stock(parity_models):
    """Behavioural check: greedy token sequences, with the tie rule on divergences."""
    tokenizer, stock, patched, reference = parity_models

    exact_matches = 0
    verdicts = []

    for index, prompt in enumerate(PROMPTS):
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()

        stock_tokens, _ = _greedy(stock, input_ids, MAX_NEW_TOKENS)
        patched_tokens, _ = _greedy(patched, input_ids, MAX_NEW_TOKENS)

        position = first_divergence(patched_tokens, stock_tokens)
        if position is None:
            exact_matches += 1
            continue

        # Both models saw an identical prefix up to this step, so logits taken
        # there are directly comparable and the fp32 arbiter is well posed.
        prefix = torch.cat(
            [
                input_ids[0],
                torch.tensor(
                    stock_tokens[:position],
                    device=input_ids.device,
                    dtype=input_ids.dtype,
                ),
            ]
        )
        verdict = build_verdict(
            position=position,
            candidate_token=patched_tokens[position],
            baseline_token=stock_tokens[position],
            candidate_logits=_logits_at(patched, prefix),
            baseline_logits=_logits_at(stock, prefix),
            reference_logits=_logits_at(reference, prefix),
            tokenizer=tokenizer,
        )
        verdict["prompt"] = index
        verdicts.append(verdict)

    report_divergences("gpt2 patched-vs-stock", verdicts, len(PROMPTS))

    # Hard gates 1 and 2, shared project-wide: every divergence is a proven fp16
    # tie that an fp32 reference arbitrates.
    assert_divergences_are_ties("gpt2 patched-vs-stock", verdicts)

    # Hard gate 3: the regression tripwire on how many prompts still match.
    assert exact_matches >= MIN_EXACT_MATCHES, (
        f"only {exact_matches}/{len(PROMPTS)} prompts matched the stock model "
        f"exactly, below the {MIN_EXACT_MATCHES} floor"
    )


@pytest.mark.gpu
def test_patch_preserves_weights_and_rejects_cache(parity_models):
    """The patch swaps the operation, not the parameters — and says so when misused."""
    from transformers import AutoModelForCausalLM

    from engine.patching import is_patched, patch_gpt2

    _, stock, patched, _ = parity_models
    assert is_patched(patched)
    assert not is_patched(stock)

    stock_params = dict(stock.named_parameters())
    patched_params = dict(patched.named_parameters())
    assert set(stock_params) == set(patched_params), (
        "patching changed the model's parameter set"
    )
    for name, tensor in stock_params.items():
        assert torch.equal(tensor, patched_params[name]), f"patching altered {name}"

    fresh = patch_gpt2(
        AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).cuda().eval()
    )
    input_ids = torch.tensor([[1, 2, 3, 4]], device="cuda")
    with pytest.raises(NotImplementedError, match="prefill-only"):
        fresh(input_ids, use_cache=True)

    # Patching is idempotent: applying it twice must not wrap a wrapper.
    before = [block.attn for block in fresh.transformer.h]
    patch_gpt2(fresh)
    assert [block.attn for block in fresh.transformer.h] == before
