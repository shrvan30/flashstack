"""T3.1 acceptance: both runners must reproduce stock transformers' greedy output.

This is the test that proves the engine — cache, position handling, prefill *and*
decode kernels — is correct end to end. Unlike the Phase 2 parity test, which ran
prefill-only with no cache, every token after the prompt here comes from
`flashattn_cuda.decode` reading the engine's own KV cache.

Divergences are judged by mechanism, not by count. The two gates:

* **Tie gate.** The stock model's top-2 logit gap at the divergence must be at
  most `MAX_TIE_ULPS` fp16 ulps — i.e. the two tokens were within rounding of
  each other in the precision the model actually ran at.
* **Arbiter gate.** An fp32 reference must agree with one of the two candidates.
  If it prefers the runner's token, the runner was *right* and stock fp16 was
  wrong. If it prefers stock's token, the runner must not be materially further
  from fp32 truth than stock is.

See the module docstring of `parity_utils` for why the gap is measured in ulps
rather than against an absolute epsilon.
"""

from __future__ import annotations

import pytest
import torch

from tests.parity_utils import (
    assert_divergences_are_ties,
    build_verdict,
    first_divergence,
    fp16_ulp,
    report_divergences,
)

MAX_NEW_TOKENS = 64

GPT2_PROMPTS = [
    "The capital of France is",
    "In 1969, NASA",
    "def fibonacci(n):",
]

QWEN_PROMPTS = [
    "What is the capital of France?",
    "Write a haiku about rain.",
    "Explain gravity in one sentence.",
]


def _stock_greedy(model, input_ids: torch.Tensor, max_new_tokens: int) -> list[int]:
    generated = model.generate(
        input_ids.unsqueeze(0), max_new_tokens=max_new_tokens, do_sample=False
    )
    return generated[0, input_ids.shape[0] :].tolist()


def _judge(runner, stock, reference, tokenizer, prompt_ids, runner_tokens, stock_tokens):
    """Verdict for one divergence, or None if the sequences are identical."""
    position = first_divergence(runner_tokens, stock_tokens)
    if position is None:
        return None

    shared_prefix = torch.cat(
        [prompt_ids, torch.tensor(stock_tokens[:position], device=prompt_ids.device)]
    )
    with torch.no_grad():
        stock_logits = stock(shared_prefix.unsqueeze(0)).logits[0, -1].float()
        reference_logits = reference(shared_prefix.unsqueeze(0)).logits[0, -1].float()

    slot = runner.allocate()
    try:
        runner_logits = runner.prefill(shared_prefix, slot).float()
    finally:
        runner.free(slot)

    return build_verdict(
        position=position,
        candidate_token=runner_tokens[position],
        baseline_token=stock_tokens[position],
        candidate_logits=runner_logits,
        baseline_logits=stock_logits,
        reference_logits=reference_logits,
        tokenizer=tokenizer,
    )


def _check(label: str, verdicts: list[dict], total: int) -> None:
    report_divergences(label, verdicts, total)
    assert_divergences_are_ties(label, verdicts)


@pytest.fixture(scope="module")
def gpt2_parity():
    from transformers import AutoModelForCausalLM

    from engine.models.gpt2 import DEFAULT_MODEL, GPT2Runner

    runner = GPT2Runner()
    stock = (
        AutoModelForCausalLM.from_pretrained(DEFAULT_MODEL, dtype=torch.float16)
        .cuda()
        .eval()
    )
    reference = (
        AutoModelForCausalLM.from_pretrained(DEFAULT_MODEL, dtype=torch.float32)
        .cuda()
        .eval()
    )
    return runner, stock, reference


@pytest.fixture(scope="module")
def qwen_parity():
    from transformers import AutoModelForCausalLM

    from engine.models.qwen25 import DEFAULT_MODEL, Qwen25Runner

    runner = Qwen25Runner()
    stock = (
        AutoModelForCausalLM.from_pretrained(DEFAULT_MODEL, dtype=torch.float16)
        .cuda()
        .eval()
    )
    reference = (
        AutoModelForCausalLM.from_pretrained(DEFAULT_MODEL, dtype=torch.float32)
        .cuda()
        .eval()
    )
    return runner, stock, reference


@pytest.mark.gpu
def test_gpt2_runner_matches_stock_generate(gpt2_parity):
    runner, stock, reference = gpt2_parity
    tokenizer = runner.tokenizer

    verdicts = []
    for prompt in GPT2_PROMPTS:
        prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids[0].cuda()
        runner_tokens = runner.generate_greedy(prompt_ids, MAX_NEW_TOKENS, stop_at_eos=True)
        stock_tokens = _stock_greedy(stock, prompt_ids, MAX_NEW_TOKENS)
        verdict = _judge(
            runner, stock, reference, tokenizer, prompt_ids, runner_tokens, stock_tokens
        )
        if verdict is not None:
            verdicts.append(verdict)

    _check("gpt2 runner-vs-stock", verdicts, len(GPT2_PROMPTS))


@pytest.mark.gpu
def test_qwen_runner_matches_stock_generate(qwen_parity):
    runner, stock, reference = qwen_parity
    tokenizer = runner.tokenizer

    verdicts = []
    for prompt in QWEN_PROMPTS:
        prompt_ids = runner.apply_chat_template([{"role": "user", "content": prompt}])
        runner_tokens = runner.generate_greedy(prompt_ids, MAX_NEW_TOKENS, stop_at_eos=True)
        stock_tokens = _stock_greedy(stock, prompt_ids, MAX_NEW_TOKENS)
        verdict = _judge(
            runner, stock, reference, tokenizer, prompt_ids, runner_tokens, stock_tokens
        )
        if verdict is not None:
            verdicts.append(verdict)

    _check("qwen2.5 runner-vs-stock", verdicts, len(QWEN_PROMPTS))


@pytest.mark.gpu
def test_decode_path_matches_prefill_of_the_same_sequence(gpt2_parity):
    """The engine's own consistency check: decoding to length N must equal prefilling N.

    This isolates the cache and the decode kernel from the model. If position
    handling or the pending-length bookkeeping were wrong, the two paths would
    disagree even though both are 'the model'.
    """
    runner, _, _ = gpt2_parity
    tokenizer = runner.tokenizer
    prompt_ids = tokenizer("The quick brown fox", return_tensors="pt").input_ids[0].cuda()

    slot = runner.allocate()
    try:
        logits = runner.prefill(prompt_ids, slot)
        grown = prompt_ids.clone()
        for _ in range(8):
            token = int(logits.argmax(-1))
            grown = torch.cat([grown, torch.tensor([token], device=grown.device)])
            logits = runner.decode_step(
                torch.tensor([token], device=grown.device), [slot]
            )[0]
        decoded_logits = logits.float()
    finally:
        runner.free(slot)

    slot = runner.allocate()
    try:
        prefilled_logits = runner.prefill(grown, slot).float()
    finally:
        runner.free(slot)

    difference = (decoded_logits - prefilled_logits).abs().max().item()

    # The bound has to scale with the logits' magnitude for the same reason the
    # tie gate does: GPT-2's largest logits sit near 128, where a single fp16 ulp
    # is already 0.125. An absolute epsilon below that would be asking for a
    # precision the output dtype cannot represent.
    scale = max(
        decoded_logits.abs().max().item(), prefilled_logits.abs().max().item()
    )
    tolerance = 4 * fp16_ulp(scale)
    print(
        f"\ndecode-vs-prefill after 8 steps: max abs logit diff {difference:.6f}, "
        f"largest |logit| {scale:.1f}, 1 ulp there {fp16_ulp(scale):.6f}, "
        f"tolerance {tolerance:.6f}"
    )
    assert difference <= tolerance, (
        f"decoding to length {grown.shape[0]} disagrees with prefilling the same "
        f"sequence by {difference:.6f} (> {tolerance:.6f} = 4 ulp at magnitude "
        f"{scale:.1f}); the cache or the position offsets are wrong"
    )
    assert torch.equal(decoded_logits.argmax(-1), prefilled_logits.argmax(-1)), (
        "decode and prefill disagree on the next token, not merely on rounding"
    )
