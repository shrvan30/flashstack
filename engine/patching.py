"""Replace GPT-2's attention computation with the flashattn_cuda prefill kernel.

The patch swaps the operation, never the weights: `c_attn` and `c_proj` are
carried over by reference from the module being replaced, so a patched model has
exactly the parameters the stock model had. Only the score / softmax / PV
computation between them changes.

Scope note: this is the Phase 2 correctness milestone and it is prefill-only.
Every forward pass recomputes the whole sequence, so a patched model must be run
with `use_cache=False`. Incremental decoding against a KV cache needs the decode
kernel and lands with the engine's cache in Phase 3.
"""

from __future__ import annotations

import flashattn_cuda
import torch
from torch import nn

HEAD_DIM = 64


class FlashAttentionGPT2(nn.Module):
    """GPT-2 self-attention with the attention core running on flashattn_cuda.

    Replaces `GPT2Attention`. The surrounding contract is unchanged: it is called
    with the post-`ln_1` hidden states and returns `(attn_output, attn_weights)`,
    where the weights are always `None` — a fused attention kernel never
    materialises the score matrix, so there is nothing to hand back. GPT-2's
    block ignores that slot.
    """

    def __init__(self, attn: nn.Module) -> None:
        super().__init__()

        if getattr(attn, "is_cross_attention", False):
            raise ValueError("flashattn_cuda patching supports self-attention only")
        if attn.head_dim != HEAD_DIM:
            raise ValueError(
                f"the kernel is specialised for head_dim={HEAD_DIM}, got {attn.head_dim}"
            )

        # Carried by reference: patching must not copy or reinitialise weights.
        self.c_attn = attn.c_attn
        self.c_proj = attn.c_proj
        self.resid_dropout = attn.resid_dropout

        self.num_heads = attn.num_heads
        self.head_dim = attn.head_dim
        self.split_size = attn.split_size
        self.layer_idx = attn.layer_idx

        # 1/sqrt(64) = 0.125 for GPT-2. Taken from the module rather than
        # hardcoded so that a config using `scale_attn_by_inverse_layer_idx`
        # would be caught by the parity test instead of silently mis-scaled.
        self.scaling = float(attn.scaling)

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_values=None,
        attention_mask: torch.Tensor | None = None,
        use_cache: bool | None = False,
        **kwargs,
    ):
        if use_cache or past_key_values is not None:
            raise NotImplementedError(
                "the patched GPT-2 attention is prefill-only; run with use_cache=False. "
                "Incremental decoding against a KV cache arrives with the engine in Phase 3."
            )
        _reject_non_causal_mask(attention_mask)

        if hidden_states.dtype != torch.float16:
            raise TypeError(
                f"the kernel takes fp16 activations, got {hidden_states.dtype}; "
                "cast the model with .half()"
            )
        if not hidden_states.is_cuda:
            raise RuntimeError("the kernel is CUDA-only; move the model to a GPU")

        batch, seq_len, _ = hidden_states.shape

        # GPT-2 stores Q, K and V in one fused projection, so the kernel's three
        # inputs have to be carved out of a single output tensor before any of
        # them can be reshaped into the (B, H, N, 64) layout it requires.
        query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)

        query = self._to_bhnd(query, batch, seq_len)
        key = self._to_bhnd(key, batch, seq_len)
        value = self._to_bhnd(value, batch, seq_len)

        attn_output = flashattn_cuda.prefill(query, key, value, True, self.scaling)

        # (B, H, N, d) -> (B, N, H*d). The transpose leaves a non-contiguous
        # view, so the copy is explicit rather than left to reshape.
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq_len, -1)

        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)
        return attn_output, None

    def _to_bhnd(self, x: torch.Tensor, batch: int, seq_len: int) -> torch.Tensor:
        """(B, N, H*d) -> contiguous (B, H, N, d), which is the kernel's contract."""
        return (
            x.view(batch, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )


def _reject_non_causal_mask(attention_mask: torch.Tensor | None) -> None:
    """Fail loudly on any mask the kernel's built-in causal masking cannot express.

    The kernel applies a dense causal mask internally and takes no mask argument,
    so it is only equivalent to stock attention when the mask transformers built
    is exactly that. Padded batches are the case this rules out; they need either
    a mask-aware kernel or per-sequence lengths, and neither exists until the
    engine's cache in Phase 3.
    """
    if attention_mask is None:
        return

    mask = attention_mask
    if mask.dim() != 4 or mask.shape[-1] != mask.shape[-2]:
        raise NotImplementedError(
            f"unsupported attention mask of shape {tuple(mask.shape)}; "
            "the patched attention handles dense causal attention only"
        )

    seq_len = mask.shape[-1]
    causal = torch.ones(seq_len, seq_len, dtype=torch.bool, device=mask.device).tril()
    if mask.dtype == torch.bool:
        allowed = mask
    else:
        allowed = mask > torch.finfo(mask.dtype).min / 2

    if not bool((allowed == causal).all()):
        raise NotImplementedError(
            "the patched attention supports dense causal attention only; the mask "
            "supplied masks additional positions (padded batches are not supported "
            "until the engine's KV cache in Phase 3)"
        )


def patch_gpt2(model: nn.Module) -> nn.Module:
    """Swap every GPT-2 block's attention for the flashattn_cuda implementation.

    Modifies `model` in place and returns it. The model must already be fp16 and
    on a CUDA device.
    """
    base = getattr(model, "transformer", model)
    blocks = getattr(base, "h", None)
    if blocks is None:
        raise TypeError(
            f"{type(model).__name__} does not look like a GPT-2 model "
            "(no transformer.h block list)"
        )

    for block in blocks:
        if isinstance(block.attn, FlashAttentionGPT2):
            continue
        block.attn = FlashAttentionGPT2(block.attn)

    return model


def is_patched(model: nn.Module) -> bool:
    """True when every block of `model` runs on the kernel."""
    base = getattr(model, "transformer", model)
    blocks = getattr(base, "h", [])
    return len(blocks) > 0 and all(
        isinstance(block.attn, FlashAttentionGPT2) for block in blocks
    )
