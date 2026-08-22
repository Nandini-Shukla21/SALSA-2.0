"""The compact SALSA 2.0 encoder-decoder models.

Two architectures share one implementation, selected by
:class:`ModelSpec`:

``compact_transformer`` (control, phase-4 candidate A)
    ``encoder_layers`` distinct layers, each executed once.

``gated_universal_transformer`` (approved primary, phase-4 candidate B)
    Fewer parameter sets than passes: layer ``t`` of a loop is
    ``layers[t % len(layers)]``, so **the same trainable module is reused**.
    Each layer is wrapped in a learned copy gate.  Raising ``encoder_loops``
    changes compute but not the parameter count, which makes effective depth a
    free experimental variable at a fixed budget.

Common decisions (fixed for this codebase, recorded in every run's metadata via
:meth:`SalsaTransformer.describe`):

* pre-norm blocks with **RMSNorm** (a single weight vector, no bias);
* **no biases** on any linear layer except the copy gate, where the bias is
  what lets the gate learn a default open/closed position;
* **GELU feed-forward** with two matrices, hidden width ``ffn_multiplier * d``;
* **RoPE** on self-attention only -- zero positional parameters;
* asymmetric widths: the encoder reads ``n`` coordinates, the decoder emits
  two digits, so nearly 90% of the budget sits in the encoder.

The task interface is teacher-forced:
``forward(src_ids, tgt_ids) -> logits (batch, tgt_len, vocab)``.
Autoregressive generation and beam search belong to a later phase.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from torch import Tensor, nn

from .attention import MultiHeadAttention, build_causal_mask, build_key_padding_mask
from .embeddings import TokenEmbedding

__all__ = [
    "ARCHITECTURES",
    "CopyGate",
    "DecoderLayer",
    "EncoderLayer",
    "FeedForward",
    "ModelSpec",
    "RMSNorm",
    "SalsaTransformer",
    "build_model",
]

#: Architecture names accepted by :func:`build_model`.
ARCHITECTURES = ("compact_transformer", "gated_universal_transformer")


# --------------------------------------------------------------------------- #
# Specification
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ModelSpec:
    """Every hyper-parameter that determines the model and its size.

    Attributes:
        vocab_size: Token count.  SALSA 2.0 ships 85 (4 specials + 81 digits).
        encoder_dim: Encoder width.
        decoder_dim: Decoder width.
        encoder_layers: Number of distinct encoder parameter sets.
        decoder_layers: Number of distinct decoder parameter sets.
        encoder_heads: Encoder attention heads.
        decoder_heads: Decoder attention heads.
        encoder_loops: Encoder passes at runtime (>= ``encoder_layers``).
        decoder_loops: Decoder passes at runtime (>= ``decoder_layers``).
        gated: Wrap each layer in a learned copy gate.
        ffn_multiplier: Feed-forward width as a multiple of the model width.
        dropout: Residual/FFN dropout.
        attention_dropout: Dropout inside attention.
        tie_embeddings: Share the decoder embedding with the output projection.
        use_rope: Use rotary positional embeddings.
        rope_base: RoPE frequency base.
        norm_eps: RMSNorm epsilon.
        pad_id: Padding token id, kept at zero in the embedding table.
        arch: Architecture name, for reporting.
    """

    vocab_size: int = 85
    encoder_dim: int = 512
    decoder_dim: int = 128
    encoder_layers: int = 1
    decoder_layers: int = 1
    encoder_heads: int = 8
    decoder_heads: int = 4
    encoder_loops: int = 2
    decoder_loops: int = 2
    gated: bool = True
    ffn_multiplier: float = 4.0
    dropout: float = 0.0
    attention_dropout: float = 0.0
    tie_embeddings: bool = False
    use_rope: bool = True
    rope_base: float = 10000.0
    norm_eps: float = 1e-6
    pad_id: int = 0
    arch: str = "gated_universal_transformer"

    def __post_init__(self) -> None:
        """Validate the specification eagerly."""
        if self.arch not in ARCHITECTURES:
            raise ValueError(f"arch must be one of {list(ARCHITECTURES)}, got {self.arch!r}.")
        if self.vocab_size < 2:
            raise ValueError(f"vocab_size must be >= 2, got {self.vocab_size}.")
        for name in ("encoder_dim", "decoder_dim", "encoder_layers", "decoder_layers",
                     "encoder_heads", "decoder_heads", "encoder_loops", "decoder_loops"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1, got {getattr(self, name)}.")
        for heads, dim, label in (
            (self.encoder_heads, self.encoder_dim, "encoder"),
            (self.decoder_heads, self.decoder_dim, "decoder"),
        ):
            if dim % heads != 0:
                raise ValueError(f"{label}_dim {dim} is not divisible by {heads} heads.")
            if self.use_rope and (dim // heads) % 2 != 0:
                raise ValueError(
                    f"{label} head dimension {dim // heads} must be even for RoPE."
                )
        if self.encoder_loops < self.encoder_layers:
            raise ValueError("encoder_loops must be >= encoder_layers.")
        if self.decoder_loops < self.decoder_layers:
            raise ValueError("decoder_loops must be >= decoder_layers.")
        if self.ffn_multiplier <= 0:
            raise ValueError("ffn_multiplier must be > 0.")

    @property
    def encoder_ffn_dim(self) -> int:
        """Encoder feed-forward hidden width."""
        return int(self.ffn_multiplier * self.encoder_dim)

    @property
    def decoder_ffn_dim(self) -> int:
        """Decoder feed-forward hidden width."""
        return int(self.ffn_multiplier * self.decoder_dim)

    @property
    def shares_encoder_weights(self) -> bool:
        """True when the encoder reuses parameter sets across passes."""
        return self.encoder_loops > self.encoder_layers

    @property
    def shares_decoder_weights(self) -> bool:
        """True when the decoder reuses parameter sets across passes."""
        return self.decoder_loops > self.decoder_layers

    @classmethod
    def from_config(cls, config, vocab_size: Optional[int] = None) -> "ModelSpec":
        """Build a specification from a :class:`~salsa.utils.config.Config`.

        Args:
            config: The experiment configuration.
            vocab_size: Vocabulary size.  Defaults to the size implied by the
                configured encoding, so the model always matches its codec.

        Returns:
            The corresponding :class:`ModelSpec`.
        """
        model = config.model
        if vocab_size is None:
            from ..data.encoding import LatticeCodec

            vocab_size = LatticeCodec.from_config(config).vocabulary.size
        return cls(
            vocab_size=int(vocab_size),
            encoder_dim=int(model.encoder_dim),
            decoder_dim=int(model.decoder_dim),
            encoder_layers=int(model.encoder_layers),
            decoder_layers=int(model.decoder_layers),
            encoder_heads=int(model.encoder_heads),
            decoder_heads=int(model.decoder_heads),
            encoder_loops=int(config.resolved_encoder_loops),
            decoder_loops=int(config.resolved_decoder_loops),
            gated=bool(model.gated),
            ffn_multiplier=float(model.ffn_multiplier),
            dropout=float(model.dropout),
            attention_dropout=float(model.attention_dropout),
            tie_embeddings=bool(model.tie_embeddings),
            arch=str(model.arch),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return a serialisable view for run metadata."""
        from dataclasses import asdict

        payload = asdict(self)
        payload["encoder_ffn_dim"] = self.encoder_ffn_dim
        payload["decoder_ffn_dim"] = self.decoder_ffn_dim
        payload["shares_encoder_weights"] = self.shares_encoder_weights
        payload["shares_decoder_weights"] = self.shares_decoder_weights
        return payload


# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #
class RMSNorm(nn.Module):
    """Root-mean-square layer normalisation.

    Cheaper than LayerNorm (no mean subtraction, no bias) and it halves the
    normalisation parameter cost, which matters when every parameter is
    budgeted.

    Parameters: ``dim`` (one weight vector).
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        """Create the norm.

        Args:
            dim: Width to normalise.
            eps: Numerical floor inside the square root.
        """
        super().__init__()
        self.dim = int(dim)
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(self.dim))

    def forward(self, x: Tensor) -> Tensor:
        """Normalise the last dimension and rescale.

        Args:
            x: Tensor whose last dimension is ``dim``.

        Returns:
            The normalised tensor, same shape and dtype as ``x``.
        """
        dtype = x.dtype
        x32 = x.to(torch.float32)
        normed = x32 * torch.rsqrt(x32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (normed.to(dtype)) * self.weight

    def extra_repr(self) -> str:
        """Describe the norm."""
        return f"dim={self.dim}, eps={self.eps}"


class FeedForward(nn.Module):
    """Two-matrix GELU feed-forward block, no biases.

    Parameters: ``2 * dim * hidden_dim``.
    """

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        """Create the feed-forward block.

        Args:
            dim: Model width.
            hidden_dim: Inner width.
            dropout: Dropout applied after the activation and the output.
        """
        super().__init__()
        self.dim = int(dim)
        self.hidden_dim = int(hidden_dim)
        self.w1 = nn.Linear(self.dim, self.hidden_dim, bias=False)
        self.w2 = nn.Linear(self.hidden_dim, self.dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Xavier-uniform initialisation."""
        nn.init.xavier_uniform_(self.w1.weight)
        nn.init.xavier_uniform_(self.w2.weight)

    def forward(self, x: Tensor) -> Tensor:
        """Apply the feed-forward transform."""
        return self.dropout(self.w2(self.dropout(nn.functional.gelu(self.w1(x)))))

    def extra_repr(self) -> str:
        """Describe the block."""
        return f"dim={self.dim}, hidden_dim={self.hidden_dim}"


class CopyGate(nn.Module):
    """Learned copy gate for shared/looped layers.

    Following the Neural Data Router (Csordas et al., 2021), the gate decides
    per position and per channel how much of the layer's output to keep versus
    how much of its input to copy forward::

        g   = sigmoid(W [x ; h] + b)
        out = g * h + (1 - g) * x

    This is what lets a looped layer leave some positions untouched on some
    iterations instead of being forced to transform everything on every pass.
    The gate is genuinely learned: ``W`` and ``b`` receive gradients, and the
    bias is initialised at zero so the gate starts at an even mix rather than
    collapsing to either the identity or the transform.

    Parameters: ``2 * dim * dim`` (weight) ``+ dim`` (bias).
    """

    def __init__(self, dim: int) -> None:
        """Create the gate.

        Args:
            dim: Model width.
        """
        super().__init__()
        self.dim = int(dim)
        self.proj = nn.Linear(2 * self.dim, self.dim, bias=True)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Xavier weights, zero bias (an even 0.5 mix at initialisation)."""
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def gate_values(self, x: Tensor, h: Tensor) -> Tensor:
        """Return the gate activations themselves.

        Exposed so tests and diagnostics can inspect what the gate learned.

        Args:
            x: Layer input, shape ``(..., dim)``.
            h: Layer output, same shape.

        Returns:
            Gate values in ``(0, 1)``, same shape.
        """
        return torch.sigmoid(self.proj(torch.cat((x, h), dim=-1)))

    def forward(self, x: Tensor, h: Tensor) -> Tensor:
        """Blend the layer input and output.

        Args:
            x: Layer input.
            h: Layer output.

        Returns:
            ``g * h + (1 - g) * x``.
        """
        gate = self.gate_values(x, h)
        return gate * h + (1.0 - gate) * x

    def extra_repr(self) -> str:
        """Describe the gate."""
        return f"dim={self.dim}"


# --------------------------------------------------------------------------- #
# Layers
# --------------------------------------------------------------------------- #
class EncoderLayer(nn.Module):
    """Pre-norm self-attention + feed-forward block, optionally copy-gated."""

    def __init__(self, spec: ModelSpec) -> None:
        """Create the layer from a :class:`ModelSpec`."""
        super().__init__()
        dim = spec.encoder_dim
        self.norm1 = RMSNorm(dim, spec.norm_eps)
        self.self_attn = MultiHeadAttention(
            query_dim=dim,
            num_heads=spec.encoder_heads,
            dropout=spec.attention_dropout,
            use_rope=spec.use_rope,
            rope_base=spec.rope_base,
        )
        self.norm2 = RMSNorm(dim, spec.norm_eps)
        self.ffn = FeedForward(dim, spec.encoder_ffn_dim, spec.dropout)
        self.dropout = nn.Dropout(spec.dropout)
        self.gate: Optional[CopyGate] = CopyGate(dim) if spec.gated else None

    def forward(self, x: Tensor, attn_mask: Optional[Tensor] = None) -> Tensor:
        """Run one encoder pass.

        Args:
            x: Tensor of shape ``(batch, seq, encoder_dim)``.
            attn_mask: Optional boolean key mask.

        Returns:
            Tensor of the same shape as ``x``.
        """
        residual = x
        h = x + self.dropout(self.self_attn(self.norm1(x), attn_mask=attn_mask))
        h = h + self.ffn(self.norm2(h))
        return self.gate(residual, h) if self.gate is not None else h


class DecoderLayer(nn.Module):
    """Pre-norm self-attention + cross-attention + FFN, optionally copy-gated."""

    def __init__(self, spec: ModelSpec) -> None:
        """Create the layer from a :class:`ModelSpec`."""
        super().__init__()
        dim = spec.decoder_dim
        self.norm1 = RMSNorm(dim, spec.norm_eps)
        self.self_attn = MultiHeadAttention(
            query_dim=dim,
            num_heads=spec.decoder_heads,
            dropout=spec.attention_dropout,
            use_rope=spec.use_rope,
            rope_base=spec.rope_base,
        )
        self.norm2 = RMSNorm(dim, spec.norm_eps)
        # Cross-attention has no shared position axis, so no rotary embedding.
        self.cross_attn = MultiHeadAttention(
            query_dim=dim,
            num_heads=spec.decoder_heads,
            kv_dim=spec.encoder_dim,
            dropout=spec.attention_dropout,
            use_rope=False,
        )
        self.norm3 = RMSNorm(dim, spec.norm_eps)
        self.ffn = FeedForward(dim, spec.decoder_ffn_dim, spec.dropout)
        self.dropout = nn.Dropout(spec.dropout)
        self.gate: Optional[CopyGate] = CopyGate(dim) if spec.gated else None

    def forward(
        self,
        x: Tensor,
        memory: Tensor,
        self_mask: Optional[Tensor] = None,
        cross_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Run one decoder pass.

        Args:
            x: Tensor of shape ``(batch, tgt_len, decoder_dim)``.
            memory: Encoder output, ``(batch, src_len, encoder_dim)``.
            self_mask: Causal (and padding) mask for self-attention.
            cross_mask: Key mask over the encoder output.

        Returns:
            Tensor of the same shape as ``x``.
        """
        residual = x
        h = x + self.dropout(self.self_attn(self.norm1(x), attn_mask=self_mask))
        h = h + self.dropout(
            self.cross_attn(self.norm2(h), key_value=memory, attn_mask=cross_mask)
        )
        h = h + self.ffn(self.norm3(h))
        return self.gate(residual, h) if self.gate is not None else h


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class SalsaTransformer(nn.Module):
    """Compact encoder-decoder model for the SALSA ``a -> b`` task.

    Example:
        >>> spec = ModelSpec()                       # approved gated UT
        >>> model = SalsaTransformer(spec)
        >>> src = torch.randint(0, 85, (2, 62))
        >>> tgt = torch.randint(0, 85, (2, 4))
        >>> model(src, tgt).shape
        torch.Size([2, 4, 85])
    """

    def __init__(self, spec: ModelSpec) -> None:
        """Build the model described by ``spec``."""
        super().__init__()
        self.spec = spec

        self.encoder_embedding = TokenEmbedding(
            spec.vocab_size, spec.encoder_dim, padding_idx=spec.pad_id
        )
        self.decoder_embedding = TokenEmbedding(
            spec.vocab_size, spec.decoder_dim, padding_idx=spec.pad_id
        )
        self.embedding_dropout = nn.Dropout(spec.dropout)

        # One module per *parameter set*.  A pass reuses layers[t % len(layers)],
        # so with encoder_layers=1 and encoder_loops=T the same trainable module
        # runs T times and the parameter count is independent of T.
        self.encoder_layers = nn.ModuleList(
            EncoderLayer(spec) for _ in range(spec.encoder_layers)
        )
        self.decoder_layers = nn.ModuleList(
            DecoderLayer(spec) for _ in range(spec.decoder_layers)
        )
        self.encoder_norm = RMSNorm(spec.encoder_dim, spec.norm_eps)
        self.decoder_norm = RMSNorm(spec.decoder_dim, spec.norm_eps)

        self.output_projection: Optional[nn.Linear]
        if spec.tie_embeddings:
            self.output_projection = None
        else:
            self.output_projection = nn.Linear(
                spec.decoder_dim, spec.vocab_size, bias=False
            )
            nn.init.xavier_uniform_(self.output_projection.weight)

    # -- properties --------------------------------------------------------- #
    @property
    def vocab_size(self) -> int:
        """Vocabulary size the model was built for."""
        return self.spec.vocab_size

    @property
    def device(self) -> torch.device:
        """Device currently holding the parameters."""
        return next(self.parameters()).device

    def encoder_layer_for(self, step: int) -> EncoderLayer:
        """Return the encoder module used at pass ``step``."""
        return self.encoder_layers[step % len(self.encoder_layers)]

    def decoder_layer_for(self, step: int) -> DecoderLayer:
        """Return the decoder module used at pass ``step``."""
        return self.decoder_layers[step % len(self.decoder_layers)]

    # -- forward ------------------------------------------------------------ #
    def encode(self, src_ids: Tensor, src_valid: Optional[Tensor] = None) -> Tensor:
        """Encode the token sequence of ``a``.

        Args:
            src_ids: Integer tensor ``(batch, src_len)``.
            src_valid: Optional boolean mask ``(batch, src_len)``; True marks a
                real token.  Sequences are fixed-length in the standard SALSA
                setting, so this is normally None.

        Returns:
            Encoder output of shape ``(batch, src_len, encoder_dim)``.

        Raises:
            ValueError: If ``src_ids`` is not 2-D.
        """
        if src_ids.dim() != 2:
            raise ValueError(f"src_ids must be (batch, seq), got {tuple(src_ids.shape)}.")
        mask = build_key_padding_mask(src_valid, src_ids.shape[1])
        x = self.embedding_dropout(self.encoder_embedding(src_ids))
        for step in range(self.spec.encoder_loops):
            x = self.encoder_layer_for(step)(x, attn_mask=mask)
        return self.encoder_norm(x)

    def decode(
        self,
        memory: Tensor,
        tgt_ids: Tensor,
        src_valid: Optional[Tensor] = None,
        causal: bool = True,
    ) -> Tensor:
        """Decode ``b`` given the encoder output, teacher-forced.

        Args:
            memory: Encoder output ``(batch, src_len, encoder_dim)``.
            tgt_ids: Target token ids ``(batch, tgt_len)``.
            src_valid: Optional encoder-side validity mask.
            causal: Apply a causal mask so position ``t`` cannot see ``> t``.

        Returns:
            Logits of shape ``(batch, tgt_len, vocab_size)``.

        Raises:
            ValueError: On inconsistent shapes.
        """
        if tgt_ids.dim() != 2:
            raise ValueError(f"tgt_ids must be (batch, seq), got {tuple(tgt_ids.shape)}.")
        if memory.dim() != 3 or memory.shape[-1] != self.spec.encoder_dim:
            raise ValueError(
                f"memory must be (batch, src_len, {self.spec.encoder_dim}), got "
                f"{tuple(memory.shape)}."
            )
        if memory.shape[0] != tgt_ids.shape[0]:
            raise ValueError("memory and tgt_ids must share the batch dimension.")

        tgt_len = tgt_ids.shape[1]
        self_mask = (
            build_causal_mask(tgt_len, tgt_len, tgt_ids.device) if causal else None
        )
        cross_mask = build_key_padding_mask(src_valid, tgt_len)

        x = self.embedding_dropout(self.decoder_embedding(tgt_ids))
        for step in range(self.spec.decoder_loops):
            x = self.decoder_layer_for(step)(
                x, memory, self_mask=self_mask, cross_mask=cross_mask
            )
        x = self.decoder_norm(x)
        if self.output_projection is None:
            return nn.functional.linear(x, self.decoder_embedding.weight)
        return self.output_projection(x)

    def forward(
        self,
        src_ids: Tensor,
        tgt_ids: Tensor,
        src_valid: Optional[Tensor] = None,
        causal: bool = True,
    ) -> Tensor:
        """Teacher-forced forward pass: encoded ``a`` and ``b`` in, logits out.

        The trainer will feed ``tgt_ids = target[:, :-1]`` and score against
        ``target[:, 1:]``; this method makes no assumption about that slicing.

        Args:
            src_ids: Encoded ``a``, shape ``(batch, src_len)``.
            tgt_ids: Encoded ``b``, shape ``(batch, tgt_len)``.
            src_valid: Optional encoder-side validity mask.
            causal: Apply the causal mask on decoder self-attention.

        Returns:
            Logits of shape ``(batch, tgt_len, vocab_size)``.
        """
        memory = self.encode(src_ids, src_valid=src_valid)
        return self.decode(memory, tgt_ids, src_valid=src_valid, causal=causal)

    # -- reporting ---------------------------------------------------------- #
    def describe(self) -> Dict[str, Any]:
        """Return a serialisable description for run metadata."""
        description = self.spec.to_dict()
        description.update(
            {
                "name": self.name,
                "trainable_parameters": sum(
                    p.numel() for p in self.parameters() if p.requires_grad
                ),
                "norm": "RMSNorm",
                "activation": "gelu",
                "attention_bias": False,
                "positional_encoding": "rope" if self.spec.use_rope else "none",
                "cross_attention_rope": False,
            }
        )
        return description

    @property
    def name(self) -> str:
        """Human-readable model name used in reports."""
        return (
            "Salsa2-GatedUT"
            if self.spec.arch == "gated_universal_transformer"
            else "Salsa2-CompactTransformer"
        )

    def extra_repr(self) -> str:
        """Describe the model."""
        spec = self.spec
        return (
            f"arch={spec.arch}, vocab={spec.vocab_size}, "
            f"enc={spec.encoder_dim}x{spec.encoder_layers}L/{spec.encoder_loops}p, "
            f"dec={spec.decoder_dim}x{spec.decoder_layers}L/{spec.decoder_loops}p, "
            f"gated={spec.gated}"
        )


def build_model(config, vocab_size: Optional[int] = None) -> SalsaTransformer:
    """Build the model described by an experiment configuration.

    Args:
        config: A :class:`~salsa.utils.config.Config`.
        vocab_size: Override the vocabulary size; by default it is taken from
            the configured encoding so the model always matches its codec.

    Returns:
        An initialised :class:`SalsaTransformer` on the CPU.
    """
    return SalsaTransformer(ModelSpec.from_config(config, vocab_size=vocab_size))
