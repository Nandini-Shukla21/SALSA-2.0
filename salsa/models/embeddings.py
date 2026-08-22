"""Token embeddings and rotary positional embeddings (RoPE).

Why RoPE and not a learned position table
-----------------------------------------
The attack has to work at several lattice dimensions -- n = 30, 50, 70, 90, 128
-- and under two representations whose sequence lengths differ (62 vs 91 tokens
at n=30, 258 vs 385 at n=128).  A learned absolute position table would:

* add ``max_len * d_model`` parameters that scale with the largest ``n``;
* leave positions never seen during training completely untrained, so a model
  trained at n=30 could not be evaluated at n=128 at all.

RoPE has **zero trainable parameters** and is defined by a closed-form function
of the position index, so one checkpoint covers every ``n`` and both
representations.  ``inv_freq`` is registered as a non-persistent buffer: it
depends only on the head dimension, never on the sequence length, and it does
not enter the checkpoint.

This module is pure PyTorch tensor arithmetic -- no CUDA-specific code.
"""

from typing import Tuple

import torch
from torch import Tensor, nn

__all__ = ["RotaryPositionalEmbedding", "TokenEmbedding", "apply_rope"]


class TokenEmbedding(nn.Module):
    """Vocabulary embedding table.

    Attributes:
        vocab_size: Number of tokens.
        dim: Embedding width.

    Parameters:
        ``vocab_size * dim`` (one weight matrix, no bias).
    """

    def __init__(self, vocab_size: int, dim: int, padding_idx: int = 0) -> None:
        """Create the embedding table.

        Args:
            vocab_size: Number of tokens in the vocabulary.
            dim: Embedding width.
            padding_idx: Index whose embedding stays at zero.  This is the
                ``<pad>`` id fixed by :class:`~salsa.data.encoding.Vocabulary`.

        Raises:
            ValueError: If ``vocab_size`` or ``dim`` is not positive.
        """
        super().__init__()
        if vocab_size < 1 or dim < 1:
            raise ValueError(
                f"vocab_size and dim must be positive, got {vocab_size} and {dim}."
            )
        self.vocab_size = int(vocab_size)
        self.dim = int(dim)
        self.padding_idx = int(padding_idx)
        self.weight = nn.Parameter(torch.empty(self.vocab_size, self.dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialise the table with a small normal, zeroing the pad row."""
        nn.init.normal_(self.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.weight[self.padding_idx].zero_()

    def forward(self, token_ids: Tensor) -> Tensor:
        """Look up embeddings.

        Args:
            token_ids: Integer tensor of shape ``(batch, seq)``.

        Returns:
            A float tensor of shape ``(batch, seq, dim)``.
        """
        return nn.functional.embedding(token_ids, self.weight, self.padding_idx)

    def extra_repr(self) -> str:
        """Describe the table."""
        return f"vocab_size={self.vocab_size}, dim={self.dim}"


class RotaryPositionalEmbedding(nn.Module):
    """Rotary positional embeddings, with **no trainable parameters**.

    The cosine/sine tables are computed on demand from the requested sequence
    length, so the module never stores anything sized to a maximum length and
    works unchanged at any ``n``.

    Example:
        >>> rope = RotaryPositionalEmbedding(head_dim=64)
        >>> sum(p.numel() for p in rope.parameters())
        0
        >>> cos, sin = rope(385, torch.device("cpu"), torch.float32)
        >>> cos.shape
        torch.Size([385, 32])
    """

    def __init__(self, head_dim: int, base: float = 10000.0) -> None:
        """Create the rotary embedding.

        Args:
            head_dim: Per-head dimension.  Must be even.
            base: Frequency base (the RoPE ``theta``).

        Raises:
            ValueError: If ``head_dim`` is not a positive even number.
        """
        super().__init__()
        if head_dim < 2 or head_dim % 2 != 0:
            raise ValueError(f"head_dim must be a positive even number, got {head_dim}.")
        self.head_dim = int(head_dim)
        self.base = float(base)
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim)
        )
        # Not persistent: derived from head_dim, so it never enters a checkpoint
        # and never ties a checkpoint to a sequence length.
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        offset: int = 0,
    ) -> Tuple[Tensor, Tensor]:
        """Return the cosine and sine tables for ``seq_len`` positions.

        Args:
            seq_len: Number of positions required.
            device: Device to build the tables on.
            dtype: Output dtype.
            offset: First position index.  Non-zero offsets support incremental
                decoding in a later phase.

        Returns:
            ``(cos, sin)``, each of shape ``(seq_len, head_dim // 2)``.

        Raises:
            ValueError: If ``seq_len`` is negative or ``offset`` is negative.
        """
        if seq_len < 0 or offset < 0:
            raise ValueError("seq_len and offset must be non-negative.")
        positions = torch.arange(
            offset, offset + seq_len, device=device, dtype=torch.float32
        )
        frequencies = torch.outer(positions, self.inv_freq.to(device=device))
        return frequencies.cos().to(dtype), frequencies.sin().to(dtype)

    def extra_repr(self) -> str:
        """Describe the rotary embedding."""
        return f"head_dim={self.head_dim}, base={self.base}, trainable_params=0"


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Apply a rotary embedding to a query or key tensor.

    Adjacent dimension pairs ``(2i, 2i+1)`` are rotated by the angle assigned
    to their position, which makes the dot product between a query at position
    ``m`` and a key at position ``n`` depend only on ``m - n``.

    Args:
        x: Tensor of shape ``(batch, heads, seq, head_dim)``.
        cos: Cosine table of shape ``(seq, head_dim // 2)``.
        sin: Sine table of shape ``(seq, head_dim // 2)``.

    Returns:
        The rotated tensor, same shape as ``x``.

    Raises:
        ValueError: If the shapes are inconsistent.
    """
    if x.dim() != 4:
        raise ValueError(f"expected (batch, heads, seq, head_dim), got {tuple(x.shape)}.")
    seq_len, head_dim = x.shape[-2], x.shape[-1]
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even, got {head_dim}.")
    if cos.shape != (seq_len, head_dim // 2) or sin.shape != cos.shape:
        raise ValueError(
            f"cos/sin must have shape ({seq_len}, {head_dim // 2}), got "
            f"{tuple(cos.shape)} and {tuple(sin.shape)}."
        )

    even, odd = x[..., 0::2], x[..., 1::2]
    cos = cos.to(dtype=x.dtype).unsqueeze(0).unsqueeze(0)
    sin = sin.to(dtype=x.dtype).unsqueeze(0).unsqueeze(0)
    rotated_even = even * cos - odd * sin
    rotated_odd = even * sin + odd * cos
    return torch.stack((rotated_even, rotated_odd), dim=-1).flatten(-2)
