"""Qwen2.5-0.5B-Instruct runner: RoPE, grouped-query attention, chat formatting.

Three things differ from GPT-2 and each one is a real design point:

* **Rotary position embeddings** instead of a learned table. Position enters
  through a rotation applied to Q and K, so it must be applied *before* the
  kernel and — critically — K must be cached *after* rotation, because a cached
  key belongs to the absolute position it was written at.
* **Grouped-query attention**: 14 query heads share 2 key/value heads. The cache
  stores the 2 real heads; they are expanded to 14 on the way into the kernel,
  which has no notion of grouping.
* **Chat formatting** through the tokenizer's template, since this is an instruct
  model and raw continuation is not how it is meant to be prompted.

The model ships in bf16 and is cast to fp16, because the kernel's contract is
fp16. That is a narrowing of range, not of precision — both have the same 10-bit
significand — and 0.5B activations stay far inside fp16's range.
"""

from __future__ import annotations

import flashattn_cuda
import torch

from engine.kv_cache import KVCache
from engine.models.base import ModelRunner

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def build_rope_tables(
    max_seq: int, head_dim: int, theta: float, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute `(max_seq, head_dim)` cos/sin tables once, in fp32.

    Built in fp32 and kept in fp32 deliberately. The tables are indexed by
    absolute position, so an error here is a *systematic* rotation error that
    grows with context rather than averaging out, and they cost
    `2 * max_seq * 64 * 4` bytes — 1 MB at 2048 positions. There is no reason to
    economise on the one table the whole position encoding depends on.
    """
    inverse_frequency = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim)
    )
    positions = torch.arange(max_seq, dtype=torch.float32, device=device)
    frequencies = torch.outer(positions, inverse_frequency)
    emb = torch.cat((frequencies, frequencies), dim=-1)
    return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rope(
    query: torch.Tensor, key: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate Q and K by their positions. `cos`/`sin` broadcast over heads."""
    dtype = query.dtype
    query_f = query.float()
    key_f = key.float()
    rotated_q = query_f * cos + rotate_half(query_f) * sin
    rotated_k = key_f * cos + rotate_half(key_f) * sin
    return rotated_q.to(dtype), rotated_k.to(dtype)


class Qwen25Runner(ModelRunner):
    """H = 14 query heads, H_kv = 2, head_dim = 64, RoPE, SwiGLU MLP."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        max_batch: int = 8,
        max_seq: int = 2048,
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
        self.inner = model.model
        self.layers = model.model.layers
        config = model.config

        self._num_layers = config.num_hidden_layers
        self._num_heads = config.num_attention_heads
        self._num_kv_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scale = self.head_dim**-0.5
        self.repeat = self._num_heads // self._num_kv_heads

        if self.head_dim != 64:
            raise ValueError(f"the kernel needs head_dim=64, this model has {self.head_dim}")

        # transformers 5.x moved rope_theta into a rope_parameters dict; fall back
        # to the old attribute so this keeps working across both layouts.
        rope = getattr(config, "rope_parameters", None) or {}
        theta = rope.get("rope_theta", getattr(config, "rope_theta", 1000000.0))
        self.cos_table, self.sin_table = build_rope_tables(
            max_seq, self.head_dim, float(theta), self.device
        )

        self.cache = KVCache(
            num_layers=self._num_layers,
            num_kv_heads=self._num_kv_heads,
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
        return self._num_kv_heads

    @property
    def eos_token_ids(self) -> set[int]:
        return self._eos

    def apply_chat_template(self, messages: list[dict], add_generation_prompt: bool = True):
        """Render OpenAI-style messages into this model's prompt format."""
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt
        )
        return self.tokenizer(text, return_tensors="pt").input_ids[0].to(self.device)

    # -- forward -----------------------------------------------------------

    def _project(self, attn, normed: torch.Tensor, batch: int, seq_len: int):
        """Separate Q/K/V projections into `(B, H, N, d)` — Qwen does not fuse them."""
        query = (
            attn.q_proj(normed)
            .view(batch, seq_len, self._num_heads, self.head_dim)
            .transpose(1, 2)
        )
        key = (
            attn.k_proj(normed)
            .view(batch, seq_len, self._num_kv_heads, self.head_dim)
            .transpose(1, 2)
        )
        value = (
            attn.v_proj(normed)
            .view(batch, seq_len, self._num_kv_heads, self.head_dim)
            .transpose(1, 2)
        )
        return query, key, value

    def _expand_kv(self, x: torch.Tensor) -> torch.Tensor:
        """2 kv heads -> 14 query heads. repeat_interleave, so copies stay adjacent."""
        if self.repeat == 1:
            return x.contiguous()
        return x.repeat_interleave(self.repeat, dim=1).contiguous()

    def _head(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.model.lm_head(self.inner.norm(hidden))

    @torch.no_grad()
    def prefill(self, input_ids: torch.Tensor, slot: int) -> torch.Tensor:
        input_ids = input_ids.to(self.device).view(1, -1)
        seq_len = input_ids.shape[1]
        if self.cache.length(slot) != 0:
            raise ValueError(f"slot {slot} already holds a sequence; free it first")
        if seq_len > self.cache.max_seq:
            raise ValueError(f"prompt of {seq_len} tokens exceeds max_seq={self.cache.max_seq}")

        cos = self.cos_table[:seq_len].unsqueeze(0).unsqueeze(0)
        sin = self.sin_table[:seq_len].unsqueeze(0).unsqueeze(0)

        hidden = self.inner.embed_tokens(input_ids)

        for layer, block in enumerate(self.layers):
            residual = hidden
            normed = block.input_layernorm(hidden)

            query, key, value = self._project(block.self_attn, normed, 1, seq_len)
            query, key = apply_rope(query, key, cos, sin)

            # The rotated key is what gets cached: a key carries the position it
            # was written at, and re-rotating it later would double-apply RoPE.
            self.cache.append(layer, slot, key[0], value[0])

            attn_out = flashattn_cuda.prefill(
                query.contiguous(), self._expand_kv(key), self._expand_kv(value), True, self.scale
            )
            attn_out = attn_out.transpose(1, 2).contiguous().view(1, seq_len, -1)
            hidden = residual + block.self_attn.o_proj(attn_out)

            hidden = hidden + block.mlp(block.post_attention_layernorm(hidden))

        self.cache.advance(slot, seq_len)
        return self._head(hidden)[0, -1]

    @torch.no_grad()
    def decode_step(self, token_ids: torch.Tensor, slots: list[int]) -> torch.Tensor:
        token_ids = token_ids.to(self.device).view(-1, 1)
        batch = token_ids.shape[0]
        if batch != len(slots):
            raise ValueError(f"{batch} tokens for {len(slots)} slots")

        positions = torch.tensor(
            [self.cache.length(slot) for slot in slots], device=self.device
        )
        # (B, 1, 1, d) so it broadcasts across heads for a one-token step.
        cos = self.cos_table[positions].unsqueeze(1).unsqueeze(1)
        sin = self.sin_table[positions].unsqueeze(1).unsqueeze(1)

        hidden = self.inner.embed_tokens(token_ids)

        for layer, block in enumerate(self.layers):
            residual = hidden
            normed = block.input_layernorm(hidden)

            query, key, value = self._project(block.self_attn, normed, batch, 1)
            query, key = apply_rope(query, key, cos, sin)

            for row, slot in enumerate(slots):
                self.cache.append(layer, slot, key[row], value[row])

            keys, values, lengths, _ = self.cache.gathered(
                layer, slots, num_query_heads=self._num_heads, pending=1
            )
            attn_out = flashattn_cuda.decode(
                query.contiguous(), keys, values, lengths, self.scale
            )
            attn_out = attn_out.transpose(1, 2).contiguous().view(batch, 1, -1)
            hidden = residual + block.self_attn.o_proj(attn_out)

            hidden = hidden + block.mlp(block.post_attention_layernorm(hidden))

        for slot in slots:
            self.cache.advance(slot, 1)

        return self._head(hidden)[:, -1]
