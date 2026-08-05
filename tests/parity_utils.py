"""The project's single definition of when a generation divergence is acceptable.

Both parity suites — the Phase 2 prefill-only model patch and the Phase 3 engine
runners — import their gates from here, so there is one tie definition rather than
one per test file.

## Why the tolerance is in ulps and not an absolute epsilon

fp16 carries a 10-bit significand, so a value is stored as `s x 2^e` with `s`
having 11 significant bits. The spacing between adjacent representable numbers —
one **ulp**, unit in the last place — is therefore not constant; it scales with
the exponent:

    ulp(x) = 2^(floor(log2 |x|) - 10)

which doubles every time `|x|` crosses a power of two:

    |x| ~ 1     ->  ulp = 0.000977
    |x| ~ 8     ->  ulp = 0.007812
    |x| ~ 16    ->  ulp = 0.015625
    |x| ~ 32    ->  ulp = 0.031250
    |x| ~ 128   ->  ulp = 0.125

An absolute threshold like `1e-2` therefore means completely different things at
different logit magnitudes. It is roughly ten ulps for GPT-2's small logits and
*less than one ulp* for anything above 16 — where it becomes unsatisfiable, since
two distinct fp16 numbers cannot be closer together than one ulp. Qwen2.5's logits
sit above 16, and a real observed divergence there had a top-2 gap of exactly
0.015625: one ulp, the tightest possible non-identity, and still a failure under
the absolute rule. Measuring the gap in ulps asks the question that was always
meant: *were these two tokens adjacent in the precision the model actually ran
at?*

## Why an fp32 arbiter, and what it does and does not prove

A tie means the two implementations disagree about something the model could not
resolve. That alone does not say either is correct, so a third opinion is needed —
the same model in fp32, which has ~21 more significand bits and is effectively
exact at fp16's resolution.

The arbiter's verdict is read as follows:

* **Prefers the candidate.** The candidate's ordering matched higher-precision
  truth and the baseline's did not. This is the strongest outcome available and
  passes outright.
* **Prefers the baseline.** The candidate broke the tie the other way. That is
  admissible only if the candidate is not *systematically* further from truth, so
  its distance to the fp32 logits must stay within `FP32_ERROR_RATIO` of the
  baseline's.
* **Prefers neither.** Both fp16 paths disagree with fp32, so the disagreement is
  not a tie-break between two defensible answers and the story does not hold.
  Fails.

What this does **not** prove: that the candidate is more accurate in general. A
single tie resolved in its favour is one sample of a coin flip. The claim being
gated is only the negative one — that the divergence is not evidence of a defect.
"""

from __future__ import annotations

import torch

# A divergence must sit within this many fp16 ulps of an exact tie. Two rather
# than one, because the losing logit can itself be a rounding away from the
# winner, so an adjacent pair can read as two steps apart after both are rounded.
MAX_TIE_ULPS = 2.0

# When the fp32 arbiter sides with the baseline, the candidate may be this many
# times further from it before counting as a real accuracy regression.
FP32_ERROR_RATIO = 1.5


def fp16_ulp(value: float) -> float:
    """Size of one fp16 ulp at `value`'s magnitude."""
    half = torch.tensor(abs(value), dtype=torch.float16)
    nxt = torch.nextafter(half, torch.tensor(float("inf"), dtype=torch.float16))
    return float(nxt.float() - half.float())


def top2_gap(logits: torch.Tensor) -> tuple[float, float]:
    """`(gap, gap_in_ulps)` between the top two logits, measured at fp16.

    Measured in fp16 on purpose: the question is whether the model *as executed*
    could separate the two tokens, and it executed in fp16. Two logits that differ
    in an fp32 view can be the identical fp16 value, in which case the argmax is
    settled by index order and any reordering of the arithmetic can flip it.
    """
    top2 = torch.topk(logits.detach().half().cpu(), 2).values.float()
    gap = float(top2[0] - top2[1])
    ulp = fp16_ulp(float(top2[0]))
    return gap, (gap / ulp if ulp > 0 else float("inf"))


def first_divergence(left: list[int], right: list[int]) -> int | None:
    """Index of the first differing token, or None if one is a prefix of the other."""
    for index in range(min(len(left), len(right))):
        if left[index] != right[index]:
            return index
    return None


def build_verdict(
    position: int,
    candidate_token: int,
    baseline_token: int,
    candidate_logits: torch.Tensor,
    baseline_logits: torch.Tensor,
    reference_logits: torch.Tensor,
    tokenizer,
) -> dict:
    """Assemble everything needed to judge one divergence.

    All three logit vectors must be taken at the *same shared prefix* — the point
    at which both implementations had produced identical tokens — or the
    comparison is not well posed.
    """
    gap, gap_ulps = top2_gap(baseline_logits)
    reference_token = int(reference_logits.argmax())

    if reference_token == candidate_token:
        prefers = "candidate"
    elif reference_token == baseline_token:
        prefers = "baseline"
    else:
        prefers = "neither"

    return {
        "position": position,
        "gap": gap,
        "gap_ulps": gap_ulps,
        "candidate_token": candidate_token,
        "baseline_token": baseline_token,
        "reference_token": reference_token,
        "reference_prefers": prefers,
        "candidate_error": (candidate_logits - reference_logits).abs().max().item(),
        "baseline_error": (baseline_logits - reference_logits).abs().max().item(),
        "decoded": {
            "candidate": tokenizer.decode([candidate_token]),
            "baseline": tokenizer.decode([baseline_token]),
            "reference": tokenizer.decode([reference_token]),
        },
    }


def report_divergences(label: str, verdicts: list[dict], total: int) -> None:
    """Print the per-divergence detail that makes a change in behaviour visible."""
    matched = total - len(verdicts)
    print(f"\n{label}: {matched}/{total} prompts matched the baseline exactly")
    for v in verdicts:
        print(
            f"  diverged at new-token index {v['position']}: "
            f"candidate={v['decoded']['candidate']!r} "
            f"baseline={v['decoded']['baseline']!r} "
            f"fp32={v['decoded']['reference']!r} (prefers {v['reference_prefers']}); "
            f"baseline top-2 gap {v['gap']:.6f} = {v['gap_ulps']:.2f} ulp; "
            f"fp32 error candidate {v['candidate_error']:.6f} "
            f"vs baseline {v['baseline_error']:.6f}"
        )
    if not verdicts:
        print("  no divergences")


def assert_divergences_are_ties(label: str, verdicts: list[dict]) -> None:
    """The project-wide gate: every divergence must be a proven, arbitrated tie."""
    for v in verdicts:
        assert v["gap_ulps"] <= MAX_TIE_ULPS, (
            f"{label} diverged at index {v['position']} where the baseline's top-2 "
            f"gap was {v['gap']:.6f} ({v['gap_ulps']:.2f} ulp), beyond the "
            f"{MAX_TIE_ULPS}-ulp tie tolerance. The baseline was confident there, "
            f"so this is a defect rather than a tie-break."
        )
        assert v["reference_prefers"] != "neither", (
            f"{label} diverged at index {v['position']} and the fp32 reference "
            f"picked {v['decoded']['reference']!r}, which is neither candidate "
            f"({v['decoded']['candidate']!r} / {v['decoded']['baseline']!r}). "
            f"Both fp16 paths are wrong there, which is not a tie-break story."
        )
        if v["reference_prefers"] == "baseline":
            ratio = v["candidate_error"] / max(v["baseline_error"], 1e-12)
            assert v["candidate_error"] <= FP32_ERROR_RATIO * v["baseline_error"], (
                f"{label} diverged at index {v['position']}, the fp32 reference "
                f"sided with the baseline, and the candidate is {ratio:.2f}x further "
                f"from it. That is an accuracy regression, not a tie-break."
            )
