"""GPT-2 runner: HuggingFace weights, flashattn_cuda attention, our own KV cache.

The module tree is loaded with transformers and then driven directly rather than
called through `GPT2LMHeadModel.forward`. That is what makes a real decode path
possible: the cache, the position offsets and the choice of prefill-vs-decode
kernel all have to be ours, and none of them are reachable from the outside of
the library's forward.

Everything that is not attention — the embeddings, both layer norms, the MLP, the
LM head — stays exactly the library's module, called as-is.
"""

from __future__ import annotations

import flashattn_cuda
import torch

from engine.kv_cache import KVCache
from engine.models.base import ModelRunner
from engine.patching import merge_heads, split_qkv_heads

DEFAULT_MODEL = "gpt2"


class GPT2Runner(ModelRunner):
    """H = H_kv = 12, head_dim = 64, learned positional embeddings."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        max_batch: int = 8,
        max_seq: int = 1024,
        device: str = "cuda",
    ) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model_name
        self.device = torch.device(device)
        self.dtype = torch.float16

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = (
            AutoModelForCausalLM.from_pretrained(model_name, dtype=self.dtype)
            .to(self.device)
            .eval()
        )
        self.model = model
        self.transformer = model.transformer
        self.blocks = model.transformer.h
        config = model.config

        self._num_layers = config.num_hidden_layers
        self._num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.split_size = config.hidden_size
        self.scale = self.head_dim**-0.5

        if self.head_dim != 64:
            raise ValueError(f"the kernel needs head_dim=64, this model has {self.head_dim}")

        # GPT-2's learned position embeddings are only defined up to n_positions,
        # so the cache must not outrun them.
        max_seq = min(max_seq, config.max_position_embeddings)

        self.cache = KVCache(
            num_layers=self._num_layers,
            num_kv_heads=self._num_heads,
            max_batch=max_batch,
            max_seq=max_seq,
            head_dim=self.head_dim,
            dtype=self.dtype,
            device=self.device,
        )

        eos = config.eos_token_id
        self._eos = {eos} if isinstance(eos, int) else set(eos or [])

    @property
    def num_layers(self) -> int:
        return self._num_layers

    @property
    def num_heads(self) -> int:
        return self._num_heads

    @property
    def num_kv_heads(self) -> int:
        return self._num_heads

    @property
    def eos_token_ids(self) -> set[int]:
        return self._eos

    # -- forward -----------------------------------------------------------

    def _embed(self, token_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        return self.transformer.wte(token_ids) + self.transformer.wpe(positions)

    def _head(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.model.lm_head(self.transformer.ln_f(hidden))

    @torch.no_grad()
    def prefill(self, input_ids: torch.Tensor, slot: int) -> torch.Tensor:
        input_ids = input_ids.to(self.device).view(1, -1)
        seq_len = input_ids.shape[1]
        if self.cache.length(slot) != 0:
            raise ValueError(f"slot {slot} already holds a sequence; free it first")

        positions = torch.arange(seq_len, device=self.device).unsqueeze(0)
        hidden = self._embed(input_ids, positions)

        for layer, block in enumerate(self.blocks):
            residual = hidden
            normed = block.ln_1(hidden)

            query, key, value = split_qkv_heads(
                block.attn.c_attn(normed), self.split_size, self._num_heads, self.head_dim
            )
            # Written before the kernel call so the cache holds this prompt even
            # if a later layer raises; the kernel does not read the cache here.
            self.cache.append(layer, slot, key[0], value[0])

            attn_out = flashattn_cuda.prefill(query, key, value, True, self.scale)
            hidden = residual + block.attn.c_proj(merge_heads(attn_out))

            hidden = hidden + block.mlp(block.ln_2(hidden))

        self.cache.advance(slot, seq_len)
        return self._head(hidden)[0, -1]

    @torch.no_grad()
    def decode_step(self, token_ids: torch.Tensor, slots: list[int]) -> torch.Tensor:
        token_ids = token_ids.to(self.device).view(-1, 1)
        batch = token_ids.shape[0]
        if batch != len(slots):
            raise ValueError(f"{batch} tokens for {len(slots)} slots")

        # Each sequence sits at a different position, so the position embedding
        # is per-row: the next token's index is the current length.
        positions = torch.tensor(
            [[self.cache.length(slot)] for slot in slots], device=self.device
        )
        hidden = self._embed(token_ids, positions)

        for layer, block in enumerate(self.blocks):
            residual = hidden
            normed = block.ln_1(hidden)

            query, key, value = split_qkv_heads(
                block.attn.c_attn(normed), self.split_size, self._num_heads, self.head_dim
            )
            for row, slot in enumerate(slots):
                self.cache.append(layer, slot, key[row], value[row])

            # pending=1: the token written just above is part of this step's
            # context, but the length is only committed once every layer is done.
            keys, values, lengths, _ = self.cache.gathered(
                layer, slots, num_query_heads=self._num_heads, pending=1
            )
            attn_out = flashattn_cuda.decode(query, keys, values, lengths, self.scale)
            hidden = residual + block.attn.c_proj(merge_heads(attn_out))

            hidden = hidden + block.mlp(block.ln_2(hidden))

        for slot in slots:
            self.cache.advance(slot, 1)

        return self._head(hidden)[:, -1]
