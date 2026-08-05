"""Preallocated KV cache for the decode kernel.

The cache is allocated once, at its maximum size, and never grows: every layer
gets a `(B_max, H_kv, S_max, 64)` fp16 tensor pair and requests are handed a
*slot* — a row of the batch dimension — for their lifetime. `flashattn_cuda.decode`
reads the whole allocation and is told each sequence's real length through
`seq_lens`, so unused tail rows cost memory but never cost correctness.

The layout is dictated by the kernel: it indexes `(batch, head, position, dim)`
with head-dim contiguous, so that is what is stored. Grouped-query models keep
only their `H_kv` distinct heads here and expand to the query-head count on the
way to the kernel — see `gathered()`.
"""

from __future__ import annotations

import torch

HEAD_DIM = 64


class KVCacheError(RuntimeError):
    """Raised on slot exhaustion, double-free or overflow of a sequence."""


class KVCache:
    """Per-layer key/value storage for up to `max_batch` concurrent sequences.

    Indexing and lifecycle are deliberately free of any CUDA dependency: the
    class works on CPU with float32 tensors, which is what makes it unit-testable
    without a GPU.
    """

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        max_batch: int = 8,
        max_seq: int = 2048,
        head_dim: int = HEAD_DIM,
        dtype: torch.dtype = torch.float16,
        device: torch.device | str = "cuda",
    ) -> None:
        if num_layers < 1 or num_kv_heads < 1:
            raise ValueError("num_layers and num_kv_heads must be positive")
        if max_batch < 1 or max_seq < 1:
            raise ValueError("max_batch and max_seq must be positive")

        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.max_batch = max_batch
        self.max_seq = max_seq
        self.head_dim = head_dim
        self.device = torch.device(device)
        self.dtype = dtype

        shape = (max_batch, num_kv_heads, max_seq, head_dim)
        self.keys = [
            torch.zeros(shape, dtype=dtype, device=self.device) for _ in range(num_layers)
        ]
        self.values = [
            torch.zeros(shape, dtype=dtype, device=self.device) for _ in range(num_layers)
        ]

        # Lengths live on the device because the decode kernel reads them there,
        # and int32 because that is the dtype its signature requires.
        self.seq_lens = torch.zeros(max_batch, dtype=torch.int32, device=self.device)
        self._free_slots = list(range(max_batch))
        self._live_slots: set[int] = set()

    # -- slot lifecycle ----------------------------------------------------

    def allocate(self) -> int:
        """Reserve a batch row for one sequence. Raises when the cache is full."""
        if not self._free_slots:
            raise KVCacheError(
                f"no free cache slot (max_batch={self.max_batch}); "
                "the scheduler must wait for a sequence to finish"
            )
        slot = self._free_slots.pop(0)
        self._live_slots.add(slot)
        self.seq_lens[slot] = 0
        return slot

    def free(self, slot: int) -> None:
        """Return a slot to the pool.

        The stored keys and values are left as they are: the next occupant starts
        at length 0, and the kernel only ever reads `[0, seq_lens[b])`, so stale
        data below a live sequence's length is unreachable. Zeroing 2048 rows per
        layer on every request completion would be pure waste.
        """
        if slot not in self._live_slots:
            raise KVCacheError(f"slot {slot} is not allocated")
        self._live_slots.discard(slot)
        self.seq_lens[slot] = 0
        self._free_slots.append(slot)

    @property
    def free_slots(self) -> int:
        return len(self._free_slots)

    @property
    def live_slots(self) -> list[int]:
        return sorted(self._live_slots)

    def length(self, slot: int) -> int:
        return int(self.seq_lens[slot].item())

    # -- writes ------------------------------------------------------------

    def append(self, layer: int, slot: int, key: torch.Tensor, value: torch.Tensor) -> None:
        """Write `n` new positions for one sequence at one layer.

        `key`/`value` are `(H_kv, n, head_dim)`. The sequence length is *not*
        advanced here — `advance()` does that once per step, after every layer has
        been written, because all layers share one length.
        """
        self._check_slot(slot)
        if key.shape != value.shape:
            raise ValueError(
                f"key/value shape mismatch: {tuple(key.shape)} vs {tuple(value.shape)}"
            )
        if key.dim() != 3 or key.shape[0] != self.num_kv_heads or key.shape[2] != self.head_dim:
            raise ValueError(
                f"expected key of shape ({self.num_kv_heads}, n, {self.head_dim}), "
                f"got {tuple(key.shape)}"
            )

        start = self.length(slot)
        n = key.shape[1]
        if start + n > self.max_seq:
            raise KVCacheError(
                f"sequence in slot {slot} would reach {start + n} tokens, "
                f"past the cache's max_seq={self.max_seq}"
            )

        self.keys[layer][slot, :, start : start + n] = key.to(self.dtype)
        self.values[layer][slot, :, start : start + n] = value.to(self.dtype)

    def advance(self, slot: int, n: int = 1) -> None:
        """Extend a sequence's length by `n` positions, after writing every layer."""
        self._check_slot(slot)
        new_length = self.length(slot) + n
        if new_length > self.max_seq:
            raise KVCacheError(
                f"sequence in slot {slot} would reach {new_length} tokens, "
                f"past the cache's max_seq={self.max_seq}"
            )
        self.seq_lens[slot] = new_length

    # -- reads -------------------------------------------------------------

    def gathered(
        self, layer: int, slots: list[int], num_query_heads: int, pending: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Build the `(B, H, S, d)` key/value views the decode kernel expects.

        `pending` counts positions that have been written with `append` but not
        yet committed with `advance`. A decode step writes the new token's K and V
        to every layer before any layer attends, because all layers share one
        length and advancing mid-loop would send later layers to the wrong offset.
        The lengths handed to the kernel must nonetheless include that token, or a
        sequence would not attend to itself — so `pending=1` is what a decode step
        passes.

        Returns `(keys, values, seq_lens, max_len)` for the requested slots, in
        the order given. Three things happen here, all forced by the kernel's
        contract:

        * **Slot gather.** The kernel's batch dimension is positional, so a batch
          of slots [3, 0, 5] has to become rows 0, 1, 2 of the tensor it reads.
        * **Truncation to `max_len`.** The kernel treats `k_cache.size(2)` as the
          cache stride, so handing it the full 2048-row allocation makes every
          block walk the whole thing. Passing only the rows any sequence in the
          batch actually occupies is the same computation over a smaller stride.
        * **Head expansion.** Grouped-query models store `H_kv` heads but the
          kernel has no notion of grouping, so the heads are repeated out to
          `num_query_heads` here.
        """
        if not slots:
            raise ValueError("no slots given")
        for slot in slots:
            self._check_slot(slot)
        if num_query_heads % self.num_kv_heads != 0:
            raise ValueError(
                f"{num_query_heads} query heads is not a multiple of "
                f"{self.num_kv_heads} key/value heads"
            )

        if pending < 0:
            raise ValueError("pending must not be negative")

        index = torch.tensor(slots, dtype=torch.long, device=self.device)
        lengths = self.seq_lens[index] + pending
        max_len = int(lengths.max().item())
        if max_len == 0:
            raise KVCacheError("every sequence in the batch is empty; nothing to attend to")
        if max_len > self.max_seq:
            raise KVCacheError(
                f"batch reaches {max_len} tokens, past max_seq={self.max_seq}"
            )

        keys = self.keys[layer].index_select(0, index)[:, :, :max_len]
        values = self.values[layer].index_select(0, index)[:, :, :max_len]

        repeat = num_query_heads // self.num_kv_heads
        if repeat > 1:
            # repeat_interleave, not repeat: query head h must map to kv head
            # h // repeat, so each kv head's copies have to be adjacent.
            keys = keys.repeat_interleave(repeat, dim=1)
            values = values.repeat_interleave(repeat, dim=1)

        return keys.contiguous(), values.contiguous(), lengths.contiguous(), max_len

    def reset(self) -> None:
        """Free every slot. Does not reallocate storage."""
        self._live_slots.clear()
        self._free_slots = list(range(self.max_batch))
        self.seq_lens.zero_()

    def memory_bytes(self) -> int:
        """Total bytes held by the key and value tensors."""
        per_tensor = self.max_batch * self.num_kv_heads * self.max_seq * self.head_dim
        itemsize = torch.empty((), dtype=self.dtype).element_size()
        return 2 * self.num_layers * per_tensor * itemsize

    def _check_slot(self, slot: int) -> None:
        if slot not in self._live_slots:
            raise KVCacheError(f"slot {slot} is not allocated")

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"KVCache(layers={self.num_layers}, kv_heads={self.num_kv_heads}, "
            f"max_batch={self.max_batch}, max_seq={self.max_seq}, "
            f"free={self.free_slots}, {self.memory_bytes() / 2**20:.0f} MiB)"
        )
