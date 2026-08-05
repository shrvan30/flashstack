"""Shared helpers for judging a greedy-generation divergence.

The question every parity test asks is the same: when the engine's tokens differ
from stock transformers', is that a defect or a coin-flip? These helpers answer it
in the two ways that actually carry information — how close the top two logits
were *in the precision the model ran at*, and which token an fp32 reference
prefers.
"""

from __future__ import annotations

import torch


def fp16_ulp(value: float) -> float:
    """Size of one fp16 ulp at `value`'s magnitude.

    fp16 has a 10-bit significand, so absolute resolution scales with the
    exponent: an ulp is ~0.00098 near 1.0 but 0.015625 near 20.0. Any "are these
    two logits distinguishable?" test therefore has to be expressed in ulps, not
    in a fixed absolute epsilon.
    """
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
    for i in range(min(len(left), right.__len__())):
        if left[i] != right[i]:
            return i
    return None
