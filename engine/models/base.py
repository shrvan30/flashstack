"""The interface every model runner implements.

A runner owns three things: the weights, the KV cache, and the mapping from a
request's slot to its rows in that cache. It exposes exactly the two operations
an inference loop needs.

The asymmetry between them is the whole architecture in miniature. `prefill`
takes one sequence at a time, because the prefill kernel computes dense causal
attention over a single length and has no way to express a padded batch.
`decode_step` takes the whole batch at once, because the decode kernel reads
per-sequence lengths and handles ragged batches natively. Serving therefore
batches decode and serialises prefill — which is the right way round, since
decode is where the time goes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from engine.kv_cache import KVCache


class ModelRunner(ABC):
    """Weights + KV cache for one model, executing on the flashattn_cuda kernels."""

    cache: KVCache
    device: torch.device
    dtype: torch.dtype

    @property
    @abstractmethod
    def num_layers(self) -> int: ...

    @property
    @abstractmethod
    def num_heads(self) -> int:
        """Query heads."""

    @property
    @abstractmethod
    def num_kv_heads(self) -> int:
        """Distinct key/value heads; equals `num_heads` outside grouped-query models."""

    @property
    @abstractmethod
    def eos_token_ids(self) -> set[int]: ...

    @abstractmethod
    def prefill(self, input_ids: torch.Tensor, slot: int) -> torch.Tensor:
        """Run one prompt through the model and fill its cache rows.

        `input_ids` is a 1-D int tensor of prompt tokens. Returns the logits for
        the final position, shape `(vocab,)` — the only row a sampler needs.
        """

    @abstractmethod
    def decode_step(self, token_ids: torch.Tensor, slots: list[int]) -> torch.Tensor:
        """Advance every sequence in `slots` by one token.

        `token_ids` is a 1-D int tensor aligned with `slots`. Returns logits of
        shape `(len(slots), vocab)`.
        """

    # -- shared helpers ----------------------------------------------------

    def allocate(self) -> int:
        return self.cache.allocate()

    def free(self, slot: int) -> None:
        self.cache.free(slot)

    def generate_greedy(
        self, input_ids: torch.Tensor, max_new_tokens: int, stop_at_eos: bool = True
    ) -> list[int]:
        """Single-sequence greedy decode. Used by the parity tests and the demo."""
        slot = self.allocate()
        try:
            logits = self.prefill(input_ids, slot)
            produced: list[int] = []
            for _ in range(max_new_tokens):
                token = int(logits.argmax(-1).item())
                produced.append(token)
                if stop_at_eos and token in self.eos_token_ids:
                    break
                next_ids = torch.tensor([token], dtype=torch.long, device=self.device)
                logits = self.decode_step(next_ids, [slot])[0]
            return produced
        finally:
            self.free(slot)
