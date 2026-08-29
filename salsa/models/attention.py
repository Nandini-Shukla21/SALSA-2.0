"""Multi-head attention for the compact SALSA 2.0 models.

One module covers both uses:

* **self-attention** -- queries, keys and values come from the same sequence,
  so ``kv_dim == query_dim`` and the parameter cost is ``4 d^2``;
* **cross-attention** -- queries come from the decoder (``d_d``) while keys and
  values come from the encoder output (``d_e``), costing
  ``2 d_d^2 + 2 d_e d_d``.

Design choices, all deliberate:

* **No biases.**  Modern pre-norm transformers gain nothing from attention
  biases, and every removed bias is budget spent on width instead.
* **RoPE on self-attention only.**  A rotary embedding encodes *relative*
  position, which is meaningful within one sequence.  Between a decoder step
  and an encoder position there is no shared position axis, so applying RoPE
  across them would inject a spurious relation.  RoPE has no parameters, so
  this choice does not affect the budget either way.
* **Masks mark keys, never queries.**  A key-padding mask can therefore never
  produce an all-masked attention row, which would yield NaNs.

Pure PyTorch, CPU-first: the only backend call is
``torch.nn.functional.scaled_dot_product_attention``, which has a CPU
implementation and needs no CUDA.
"""

from typing import Optional

import torch
from torch import Tensor, nn

from .embeddings import RotaryPositionalEmbedding, apply_rope

__all__ = ["MultiHeadAttention", "build_causal_mask", "build_key_padding_mask"]


def build_key_padding_mask(valid: Optional[Tensor], num_queries: int) -> Optional[Tensor]:
    """Turn a per-token validity mask into a broadcastable attention mask.

    Args:
        valid: Boolean tensor of shape ``(batch, num_keys)`` where True marks a
            real token, or None when every token is real.
        num_queries: Query length, used only for the broadcast shape.

    Returns:
        A boolean tensor of shape ``(batch, 1, 1, num_keys)`` where True means
        "this key may be attended to", or None when no masking is needed.
    """
    if valid is None:
        return None
    if valid.dim() != 2:
        raise ValueError(f"key padding mask must be (batch, keys), got {tuple(valid.shape)}.")
    return valid.to(torch.bool)[:, None, None, :]


def build_causal_mask(
    num_queries: int, num_keys: int, device: torch.device
) -> Tensor:
    """Return a lower-triangular mask allowing each query to see only the past.

    Args:
        num_queries: Number of query positions.
        num_keys: Number of key positions.
        device: Device for the mask.

    Returns:
        A boolean tensor of shape ``(1, 1, num_queries, num_keys)``; True means
        the position may be attended to.
    """
    mask = torch.ones(num_queries, num_keys, dtype=torch.bool, device=device).tril(
        diagonal=num_keys - num_queries
    )
    return mask[None, None, :, :]


class MultiHeadAttention(nn.Module):
    """Multi-head attention with optional rotary positional embeddings.

    Attributes:
        query_dim: Width of the query stream (and of the output).
        kv_dim: Width of the key/value stream.
        num_heads: Number of attention heads.
        head_dim: ``query_dim // num_heads``.
        use_rope: Whether rotary embeddings are applied to queries and keys.

    Parameters:
        ``query_dim*query_dim`` (Q) ``+ 2*kv_dim*query_dim`` (K, V)
        ``+ query_dim*query_dim`` (output), no biases.
    """

    def __init__(
        self,
        query_dim: int,
        num_heads: int,
        kv_dim: Optional[int] = None,
        dropout: float = 0.0,
        use_rope: bool = True,
        rope_base: float = 10000.0,
    ) -> None:
        """Create the attention block.

        Args:
            query_dim: Width of the query stream.
            num_heads: Number of heads; must divide ``query_dim``.
            kv_dim: Width of the key/value stream.  Defaults to ``query_dim``
                (self-attention).
            dropout: Attention dropout probability.
            use_rope: Apply rotary embeddings.  Must be False for
                cross-attention, where queries and keys have no shared position
                axis.
            rope_base: RoPE frequency base.

        Raises:
            ValueError: If ``num_heads`` does not divide ``query_dim``, or the
                resulting head dimension is odd while RoPE is enabled.
        """
        super().__init__()
        if num_heads < 1:
            raise ValueError(f"num_heads must be >= 1, got {num_heads}.")
        if query_dim % num_heads != 0:
            raise ValueError(
                f"query_dim ({query_dim}) must be divisible by num_heads ({num_heads})."
            )
        self.query_dim = int(query_dim)
        self.kv_dim = int(query_dim if kv_dim is None else kv_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.query_dim // self.num_heads
        self.dropout = float(dropout)
        self.use_rope = bool(use_rope)
        self.is_self_attention = self.kv_dim == self.query_dim

        self.q_proj = nn.Linear(self.query_dim, self.query_dim, bias=False)
        self.k_proj = nn.Linear(self.kv_dim, self.query_dim, bias=False)
        self.v_proj = nn.Linear(self.kv_dim, self.query_dim, bias=False)
        self.out_proj = nn.Linear(self.query_dim, self.query_dim, bias=False)

        self.rope: Optional[RotaryPositionalEmbedding] = None
        if self.use_rope:
            self.rope = RotaryPositionalEmbedding(self.head_dim, base=rope_base)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Xavier-uniform initialisation for every projection."""
        for module in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            nn.init.xavier_uniform_(module.weight)

    def _split_heads(self, x: Tensor) -> Tensor:
        """Reshape ``(batch, seq, dim)`` into ``(batch, heads, seq, head_dim)``."""
        batch, seq, _ = x.shape
        return x.view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: Tensor) -> Tensor:
        """Reshape ``(batch, heads, seq, head_dim)`` back to ``(batch, seq, dim)``."""
        batch, _, seq, _ = x.shape
        return x.transpose(1, 2).reshape(batch, seq, self.query_dim)

    def forward(
        self,
        query: Tensor,
        key_value: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        query_offset: int = 0,
        key_bias: Optional[Tensor] = None,
    ) -> Tensor:
        """Run attention.

        Args:
            query: Tensor of shape ``(batch, q_len, query_dim)``.
            key_value: Tensor of shape ``(batch, kv_len, kv_dim)``.  None means
                self-attention over ``query``.
            attn_mask: Boolean tensor broadcastable to
                ``(batch, heads, q_len, kv_len)``, where True means the key may
                be attended to.
            key_bias: Optional additive bias on the attention logits, indexed by
                KEY position only and broadcastable to
                ``(batch, heads, 1, kv_len)``.  ``None`` -- the default and the
                only value V1 ever passes -- leaves this call bit-identical to
                the unbiased implementation.
            query_offset: Position offset for the query's rotary embedding,
                used by incremental decoding in a later phase.

        Returns:
            Tensor of shape ``(batch, q_len, query_dim)``.

        Raises:
            ValueError: On inconsistent shapes.
        """
        source = query if key_value is None else key_value
        if query.dim() != 3 or source.dim() != 3:
            raise ValueError("query and key_value must be (batch, seq, dim) tensors.")
        if source.shape[-1] != self.kv_dim:
            raise ValueError(
                f"key/value width {source.shape[-1]} does not match kv_dim {self.kv_dim}."
            )

        q = self._split_heads(self.q_proj(query))
        k = self._split_heads(self.k_proj(source))
        v = self._split_heads(self.v_proj(source))

        if self.rope is not None:
            cos_q, sin_q = self.rope(
                q.shape[-2], q.device, q.dtype, offset=query_offset
            )
            q = apply_rope(q, cos_q, sin_q)
            cos_k, sin_k = self.rope(k.shape[-2], k.device, k.dtype)
            k = apply_rope(k, cos_k, sin_k)

        mask = attn_mask
        if key_bias is not None:
            # scaled_dot_product_attention ADDS a float mask to the logits, so a
            # boolean key mask has to become -inf / 0 before the bias joins it.
            if mask is not None and mask.dtype == torch.bool:
                mask = torch.zeros_like(mask, dtype=q.dtype).masked_fill(
                    ~mask, float("-inf")
                )
            mask = key_bias if mask is None else mask + key_bias

        context = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        return self.out_proj(self._merge_heads(context))

    def extra_repr(self) -> str:
        """Describe the attention block."""
        kind = "self" if self.is_self_attention else "cross"
        return (
            f"{kind}, query_dim={self.query_dim}, kv_dim={self.kv_dim}, "
            f"heads={self.num_heads}, head_dim={self.head_dim}, rope={self.use_rope}"
        )
