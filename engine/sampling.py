"""Token sampling: greedy, temperature, top-p, repetition penalty, stop conditions.

Everything here is pure PyTorch on a logits tensor and has no CUDA or kernel
dependency, so it is unit-testable on CPU. The functions take and return plain
tensors rather than reaching into a runner, which is what lets the server's
behaviour be tested without loading a model.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

# Below this, temperature scaling would amplify logits into overflow; the
# OpenAI API treats temperature 0 as "be deterministic", so that is what it means.
GREEDY_TEMPERATURE_EPSILON = 1e-5


@dataclass
class SamplingParams:
    """One request's decoding configuration.

    Defaults are the deterministic ones: temperature 0 is greedy, top_p 1.0 keeps
    the full distribution, and the repetition penalty is off at 1.0. The phase
    contract asks for the penalty to be optional and off by default, because it
    silently changes outputs and would confound the parity tests.
    """

    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 128
    repetition_penalty: float = 1.0
    stop_token_ids: set[int] = field(default_factory=set)
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.temperature < 0:
            raise ValueError(f"temperature must be >= 0, got {self.temperature}")
        if not 0 < self.top_p <= 1:
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}")
        if self.max_tokens < 1:
            raise ValueError(f"max_tokens must be >= 1, got {self.max_tokens}")
        if self.repetition_penalty <= 0:
            raise ValueError(
                f"repetition_penalty must be > 0, got {self.repetition_penalty}"
            )

    @property
    def is_greedy(self) -> bool:
        return self.temperature < GREEDY_TEMPERATURE_EPSILON


def apply_repetition_penalty(
    logits: torch.Tensor, previous_tokens: torch.Tensor, penalty: float
) -> torch.Tensor:
    """Divide already-seen tokens' logits by `penalty`, or multiply if negative.

    The sign split is the CTRL formulation and it is not an implementation detail:
    dividing a negative logit by a penalty > 1 would move it *up*, rewarding the
    repetition it is meant to discourage. Multiplying instead pushes it further
    down, so the penalty reduces probability regardless of the logit's sign.
    """
    if penalty == 1.0 or previous_tokens.numel() == 0:
        return logits

    out = logits.clone()
    seen = torch.unique(previous_tokens.to(out.device))
    values = out[seen]
    out[seen] = torch.where(values > 0, values / penalty, values * penalty)
    return out


def top_p_filter(probabilities: torch.Tensor, top_p: float) -> torch.Tensor:
    """Zero out the tail of the distribution beyond cumulative mass `top_p`.

    The token that *crosses* the threshold is kept, not dropped. Excluding it
    would make `top_p` slightly below any token's own probability select nothing,
    and it is the standard nucleus-sampling convention. The result is
    renormalised, so the caller gets a valid distribution.
    """
    if top_p >= 1.0:
        return probabilities

    sorted_probabilities, sorted_indices = torch.sort(probabilities, descending=True, dim=-1)
    cumulative = torch.cumsum(sorted_probabilities, dim=-1)

    # Drop everything strictly after the first index whose cumulative mass
    # reaches top_p; shifting by one is what keeps that crossing token.
    remove = cumulative - sorted_probabilities > top_p
    sorted_probabilities = sorted_probabilities.masked_fill(remove, 0.0)

    filtered = torch.zeros_like(probabilities)
    filtered.scatter_(-1, sorted_indices, sorted_probabilities)
    total = filtered.sum(dim=-1, keepdim=True)
    return filtered / total


def sample(
    logits: torch.Tensor,
    params: SamplingParams,
    previous_tokens: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Pick one token per row. `logits` is `(vocab,)` or `(batch, vocab)`.

    Returns a 1-D int64 tensor of length `batch` (or 1 for a flat input).
    """
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
    if logits.dim() != 2:
        raise ValueError(f"expected (vocab,) or (batch, vocab) logits, got {tuple(logits.shape)}")

    # Sampling maths runs in fp32 whatever the model produced: softmax over a
    # 150k-entry fp16 vocabulary loses meaningful mass in the tail, which is
    # exactly the region top-p is deciding about.
    working = logits.float()

    if previous_tokens is not None and params.repetition_penalty != 1.0:
        rows = [
            apply_repetition_penalty(working[i], previous_tokens, params.repetition_penalty)
            for i in range(working.shape[0])
        ]
        working = torch.stack(rows)

    if params.is_greedy:
        return working.argmax(dim=-1)

    probabilities = torch.softmax(working / params.temperature, dim=-1)
    probabilities = top_p_filter(probabilities, params.top_p)
    return torch.multinomial(probabilities, num_samples=1, generator=generator).squeeze(-1)


def make_generator(seed: int | None, device: torch.device | str = "cpu") -> torch.Generator | None:
    """A seeded generator, or None when the request did not ask for reproducibility."""
    if seed is None:
        return None
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator


@dataclass
class StopState:
    """Tracks why and when a sequence should stop producing tokens."""

    params: SamplingParams
    produced: int = 0
    finish_reason: str | None = None

    def observe(self, token: int) -> bool:
        """Record a generated token; returns True when the sequence is finished."""
        self.produced += 1
        if token in self.params.stop_token_ids:
            # An EOS token is a stop signal, not output: the caller drops it.
            self.finish_reason = "stop"
            return True
        if self.produced >= self.params.max_tokens:
            self.finish_reason = "length"
            return True
        return False

    @property
    def finished(self) -> bool:
        return self.finish_reason is not None
