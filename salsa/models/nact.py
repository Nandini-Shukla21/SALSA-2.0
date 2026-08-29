"""Salsa2-NACT: the numerical-aware compact transformer approved in phase 14.

What changes relative to V1
---------------------------
Only the encoder's **input front end**.  The encoder body, the whole decoder and
the output interface are the V1 modules, unmodified, so a NACT checkpoint speaks
exactly the same target language as a GatedUT checkpoint and every downstream
script -- training, evaluation, phase-10 recovery, phase-12 generalization --
runs against it without a line of change.

The front end replaces ``width`` digit tokens per coordinate with **one learned
coordinate representation**::

    e_i = sum_w DigitEmb[w, digit_w(a_i)]                  # the V1 tokens, kept
        + W_num @ [ 1[a_i == 0],                           # zero indicator
                    centered(a_i) / (q/2),                 # signed residue
                    cos(2*pi*a_i/q), sin(2*pi*a_i/q) ]     # k=1 character
        + b_num
        + CoordEmb[i]                                      # ABSOLUTE coordinate
        + 1[a_i == 0] * ZeroVector                         # learned zero feature

so the encoder sequence is ``<bos> e_0 ... e_{n-1} <eos>``, of length ``n + 2``
instead of ``2n + 2`` (representation R) or ``3n + 1`` (representation P).

Why these features and not others (phase 14, with the measurements behind them):

* ``b = sum_i a_i s_i + e mod q`` is integer addition followed by one reduction,
  so the integer magnitude of ``a_i`` is the quantity being added.  The centered
  residue supplies it directly and symmetric about zero.
* ``cos``/``sin`` at ``k=1`` are the characters of ``Z_q``: addition mod q
  becomes addition of angles.  This is the only genuinely non-affine feature in
  the set, and the only one that expresses wraparound.
* The zero indicator is a *measured* repair, not a hunch.  In base-81 lsb-first,
  "this coordinate is zero" is a conjunction across two token positions --
  ``digit0 == 0`` holds for 4 of 251 residues and ``digit1 == 0`` for 81, but
  only one residue satisfies both -- so the encoder currently has to spend
  attention computing it, once per coordinate, on every pass.
* ``b = sum_i a_i s_i`` binds coordinate ``i`` to secret bit ``s_i``, so the task
  needs **absolute** coordinate identity.  RoPE supplies only relative position.

Rejected as redundant, and deliberately absent: the normalized residue ``x/q``
(an affine function of the centered residue), the modular distance to zero (a
monotone function of the ``k=1`` cosine) and a separate magnitude feature (the
same quantity renamed).

The input is read from the **V1 token layout**.  NACT takes the same
``src_ids`` tensor the codec already produces, recovers the digits from the
positions the codec put them in, and reconstructs ``a_i`` exactly.  Nothing about
the data pipeline changes, and the reconstruction is lossless by construction:
a test asserts it against the codec for both representations.

Sparse attention bias
---------------------
Each encoder head owns one learned scalar ``beta_h`` added to the attention
logit of every **key** at a zero coordinate::

    logit(i, j) += beta_h * 1[a_j == 0]

Initialised at exactly zero, so at step 0 the model is identical to one without
the bias and any suppression of zeros is *learned*, never hard-coded.  Eight
scalars in total at ``encoder_heads = 8``.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
from torch import Tensor, nn

from .attention import MultiHeadAttention, build_causal_mask, build_key_padding_mask
from .embeddings import TokenEmbedding
from .transformer import CopyGate, DecoderLayer, FeedForward, RMSNorm

__all__ = ["NactSpec", "NactFrontEnd", "NactEncoderLayer", "SalsaNact", "build_nact"]

#: Order of the numerical scalars fed to the projection.  Fixed, because the
#: parameter count and every report depend on it.
NUMERICAL_FEATURES: Tuple[str, ...] = (
    "zero_indicator",
    "signed_centered_residue",
    "cos_2pi_x_over_q",
    "sin_2pi_x_over_q",
)


@dataclass
class NactSpec:
    """Specification for a NACT model.

    The transformer fields mirror :class:`~salsa.models.transformer.ModelSpec`
    exactly; the remainder describe the front end.

    Attributes:
        vocab_size: Output vocabulary (85 for Salsa 2.0).
        encoder_dim: Encoder width.
        decoder_dim: Decoder width.
        encoder_layers: Distinct encoder parameter sets.
        decoder_layers: Distinct decoder parameter sets.
        encoder_heads: Encoder attention heads.
        decoder_heads: Decoder attention heads.
        encoder_loops: Encoder passes; reuses the parameter sets, costs nothing.
        decoder_loops: Decoder passes.
        gated: Copy gate on every layer.
        ffn_multiplier: FFN hidden width as a multiple of the model width.
        dropout: Residual/FFN dropout.
        attention_dropout: Attention dropout.
        tie_embeddings: Share the decoder embedding with the output projection.
        use_rope: Rotary embeddings on self-attention.
        rope_base: RoPE frequency base.
        norm_eps: RMSNorm epsilon.
        pad_id: Padding id.
        arch: Architecture name.
        q: Modulus of the lattice problem.
        base: Digit base of the input encoding.
        digit_width: Digits per coordinate.
        digit_id_offset: Vocabulary id of digit value 0.
        lsb_first: Digit order of the input encoding.
        separator: Whether the input layout carries separators (representation P).
        max_coordinates: Largest ``n`` the coordinate embedding covers.
        include_bos_eos: Whether the input layout is framed by ``<bos>``/``<eos>``.
    """

    vocab_size: int = 85
    encoder_dim: int = 512
    decoder_dim: int = 128
    encoder_layers: int = 1
    decoder_layers: int = 1
    encoder_heads: int = 8
    decoder_heads: int = 4
    encoder_loops: int = 4
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
    arch: str = "nact"
    # -- front end ---------------------------------------------------------- #
    q: int = 251
    base: int = 81
    digit_width: int = 2
    digit_id_offset: int = 4
    lsb_first: bool = True
    separator: bool = False
    max_coordinates: int = 128
    include_bos_eos: bool = True

    def __post_init__(self) -> None:
        """Validate eagerly, with the same strictness as V1."""
        if self.arch != "nact":
            raise ValueError(f"NactSpec.arch must be 'nact', got {self.arch!r}.")
        if self.vocab_size < 2:
            raise ValueError(f"vocab_size must be >= 2, got {self.vocab_size}.")
        for name in ("encoder_dim", "decoder_dim", "encoder_layers", "decoder_layers",
                     "encoder_heads", "decoder_heads", "encoder_loops", "decoder_loops",
                     "base", "digit_width", "max_coordinates"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1, got {getattr(self, name)}.")
        if self.q < 2:
            raise ValueError(f"q must be >= 2, got {self.q}.")
        for heads, dim, label in ((self.encoder_heads, self.encoder_dim, "encoder"),
                                  (self.decoder_heads, self.decoder_dim, "decoder")):
            if dim % heads != 0:
                raise ValueError(f"{label}_dim {dim} is not divisible by {heads} heads.")
            if self.use_rope and (dim // heads) % 2 != 0:
                raise ValueError(
                    f"{label} head dimension {dim // heads} must be even for RoPE.")
        if self.encoder_loops < self.encoder_layers:
            raise ValueError("encoder_loops must be >= encoder_layers.")
        if self.decoder_loops < self.decoder_layers:
            raise ValueError("decoder_loops must be >= decoder_layers.")
        if self.ffn_multiplier <= 0:
            raise ValueError("ffn_multiplier must be > 0.")
        if self.digit_id_offset + self.base > self.vocab_size:
            raise ValueError(
                f"digit ids {self.digit_id_offset}..{self.digit_id_offset + self.base - 1} "
                f"do not fit in a vocabulary of {self.vocab_size}.")

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

    @property
    def num_numerical_features(self) -> int:
        """How many numerical scalars are projected per coordinate."""
        return len(NUMERICAL_FEATURES)

    def coordinates_for(self, source_length: int) -> int:
        """Recover ``n`` from a V1-layout input length.

        Args:
            source_length: Length of the codec's ``src_ids`` sequence.

        Returns:
            The lattice dimension.

        Raises:
            ValueError: If the length is not a valid layout for this spec.
        """
        specials = 2 if self.include_bos_eos else 0
        body = int(source_length) - specials
        if self.separator:
            # width digits per coordinate plus (n - 1) separators.
            n, remainder = divmod(body + 1, self.digit_width + 1)
            if remainder or n < 1:
                raise ValueError(
                    f"source length {source_length} is not a valid representation-P "
                    f"layout for width {self.digit_width}.")
            return n
        n, remainder = divmod(body, self.digit_width)
        if remainder or n < 1:
            raise ValueError(
                f"source length {source_length} is not a valid representation-R "
                f"layout for width {self.digit_width}.")
        return n

    def encoder_length_for(self, n: int) -> int:
        """Encoder sequence length NACT uses for dimension ``n``."""
        return int(n) + (2 if self.include_bos_eos else 0)

    @classmethod
    def from_config(cls, config, vocab_size: Optional[int] = None) -> "NactSpec":
        """Build a specification from an experiment configuration.

        The front-end fields are taken from the *codec*, so the model can never
        disagree with the encoding it is trained on.
        """
        from ..data.encoding import LatticeCodec

        codec = LatticeCodec.from_config(config)
        model = config.model
        if vocab_size is None:
            vocab_size = codec.vocabulary.size
        digit_ids = [codec.vocabulary.token_to_id[t] for t in codec.vocabulary.digit_tokens]
        offset = min(digit_ids)
        if digit_ids != list(range(offset, offset + len(digit_ids))):
            raise ValueError(
                "NACT assumes digit token ids are contiguous; this vocabulary's are not.")
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
            arch="nact",
            q=int(config.lwe.q),
            base=int(codec.input_encoder.base),
            digit_width=int(codec.input_encoder.width),
            digit_id_offset=int(offset),
            lsb_first=bool(config.encoding.digit_order == "lsb_first"),
            separator=bool(codec.separator),
            max_coordinates=int(max(128, config.lwe.n)),
            include_bos_eos=bool(codec.include_bos_eos),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return a serialisable view for run metadata."""
        from dataclasses import asdict

        payload = asdict(self)
        payload["encoder_ffn_dim"] = self.encoder_ffn_dim
        payload["decoder_ffn_dim"] = self.decoder_ffn_dim
        payload["shares_encoder_weights"] = self.shares_encoder_weights
        payload["shares_decoder_weights"] = self.shares_decoder_weights
        payload["numerical_features"] = list(NUMERICAL_FEATURES)
        return payload


class NactFrontEnd(nn.Module):
    """Builds one encoder token per lattice coordinate from the V1 token layout.

    Parameters:
        ``digit_embedding``  ``width * base * d``
        ``special_embedding``  ``4 * d``
        ``numerical_projection``  ``F * d + d``
        ``coordinate_embedding``  ``max_coordinates * d``
        ``zero_vector``  ``d``
    """

    def __init__(self, spec: NactSpec) -> None:
        """Create the front end."""
        super().__init__()
        self.spec = spec
        dim = spec.encoder_dim

        self.digit_embedding = nn.Parameter(
            torch.empty(spec.digit_width, spec.base, dim))
        self.special_embedding = nn.Parameter(torch.empty(4, dim))
        self.numerical_projection = nn.Linear(spec.num_numerical_features, dim, bias=True)
        self.coordinate_embedding = nn.Parameter(
            torch.empty(spec.max_coordinates, dim))
        self.zero_vector = nn.Parameter(torch.zeros(dim))

        # Digit value for each vocabulary id, -1 where the id is not a digit.
        # Buffer, not a parameter: it is a property of the codec, not learned.
        table = torch.full((spec.vocab_size,), -1, dtype=torch.long)
        for value in range(spec.base):
            table[spec.digit_id_offset + value] = value
        self.register_buffer("digit_value_table", table, persistent=False)

        # Powers of the base in the order the digits appear in the sequence.
        exponents = torch.arange(spec.digit_width, dtype=torch.long)
        if not spec.lsb_first:
            exponents = torch.flip(exponents, dims=(0,))
        self.register_buffer(
            "digit_weights", (spec.base ** exponents).to(torch.long), persistent=False)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Deterministic initialisation.

        Embeddings get the usual ``N(0, 0.02)``; the zero vector starts at zero
        so an untrained model treats a zero coordinate exactly like any other,
        and any special handling of zeros has to be learned.
        """
        nn.init.normal_(self.digit_embedding, mean=0.0, std=0.02)
        nn.init.normal_(self.special_embedding, mean=0.0, std=0.02)
        nn.init.normal_(self.coordinate_embedding, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.numerical_projection.weight)
        nn.init.zeros_(self.numerical_projection.bias)
        nn.init.zeros_(self.zero_vector)

    def digit_columns(self, n: int) -> Tensor:
        """Column indices of the digit tokens in the V1 layout, shape ``(n, width)``."""
        spec = self.spec
        start = 1 if spec.include_bos_eos else 0
        stride = spec.digit_width + (1 if spec.separator else 0)
        starts = start + stride * torch.arange(n, device=self.digit_weights.device)
        offsets = torch.arange(spec.digit_width, device=self.digit_weights.device)
        return starts[:, None] + offsets[None, :]

    def decode_values(self, src_ids: Tensor) -> Tuple[Tensor, Tensor]:
        """Recover per-coordinate digits and integer values from ``src_ids``.

        Args:
            src_ids: The codec's input ids, shape ``(batch, L_v1)``.

        Returns:
            ``(digits, values)`` of shapes ``(batch, n, width)`` and
            ``(batch, n)``.

        Raises:
            ValueError: If a digit position does not hold a digit token.
        """
        spec = self.spec
        n = spec.coordinates_for(src_ids.shape[1])
        columns = self.digit_columns(n).reshape(-1)
        digit_ids = src_ids.index_select(1, columns).view(-1, n, spec.digit_width)
        digits = self.digit_value_table[digit_ids]
        if bool((digits < 0).any()):
            raise ValueError(
                "src_ids contains a non-digit token where a digit was expected; "
                "the sequence layout does not match this model's spec.")
        values = (digits * self.digit_weights).sum(dim=-1) % spec.q
        return digits, values

    def numerical_features(self, values: Tensor) -> Tensor:
        """Compute the four kept scalars, shape ``(batch, n, 4)``.

        All four are functions of the public input alone.  Nothing here sees the
        secret, the error or the target.
        """
        spec = self.spec
        q = float(spec.q)
        half = spec.q // 2
        floats = values.to(torch.float32)

        zero = (values == 0).to(torch.float32)
        centered = ((values + half) % spec.q - half).to(torch.float32) / (q / 2.0)
        angle = 2.0 * torch.pi * floats / q
        return torch.stack((zero, centered, torch.cos(angle), torch.sin(angle)), dim=-1)

    def forward(self, src_ids: Tensor) -> Tuple[Tensor, Tensor]:
        """Build the encoder input.

        Args:
            src_ids: The codec's input ids, shape ``(batch, L_v1)``.

        Returns:
            ``(embeddings, zero_indicator)`` of shapes ``(batch, n+2, d)`` and
            ``(batch, n+2)``.  The indicator is 0 at ``<bos>``/``<eos>``.

        Raises:
            ValueError: If ``n`` exceeds ``max_coordinates``.
        """
        spec = self.spec
        digits, values = self.decode_values(src_ids)
        batch, n = values.shape
        if n > spec.max_coordinates:
            raise ValueError(
                f"n = {n} exceeds max_coordinates = {spec.max_coordinates}; the "
                "coordinate embedding table would need extending.")

        # sum over digit slots, each with its own table
        embedded = torch.zeros(
            batch, n, spec.encoder_dim, dtype=self.digit_embedding.dtype,
            device=src_ids.device)
        for slot in range(spec.digit_width):
            embedded = embedded + self.digit_embedding[slot][digits[:, :, slot]]

        features = self.numerical_features(values)
        embedded = embedded + self.numerical_projection(features)
        embedded = embedded + self.coordinate_embedding[:n].unsqueeze(0)
        zero = features[..., 0]
        embedded = embedded + zero.unsqueeze(-1) * self.zero_vector

        if not spec.include_bos_eos:
            return embedded, zero

        bos = self.special_embedding[1].view(1, 1, -1).expand(batch, 1, -1)
        eos = self.special_embedding[2].view(1, 1, -1).expand(batch, 1, -1)
        sequence = torch.cat((bos, embedded, eos), dim=1)
        padded_zero = torch.cat(
            (torch.zeros(batch, 1, device=src_ids.device),
             zero,
             torch.zeros(batch, 1, device=src_ids.device)), dim=1)
        return sequence, padded_zero

    def extra_repr(self) -> str:
        """Describe the front end."""
        spec = self.spec
        return (f"q={spec.q}, base={spec.base}, width={spec.digit_width}, "
                f"lsb_first={spec.lsb_first}, separator={spec.separator}, "
                f"max_coordinates={spec.max_coordinates}, "
                f"features={list(NUMERICAL_FEATURES)}")


class NactEncoderLayer(nn.Module):
    """V1's encoder layer plus a learned per-head bias on zero-coordinate keys.

    Structurally identical to :class:`~salsa.models.transformer.EncoderLayer`;
    the only addition is ``sparse_attention_bias``, one scalar per head, applied
    to KEY positions only and initialised at zero.
    """

    def __init__(self, spec: NactSpec) -> None:
        """Create the layer."""
        super().__init__()
        dim = spec.encoder_dim
        self.norm1 = RMSNorm(dim, spec.norm_eps)
        self.self_attn = MultiHeadAttention(
            query_dim=dim, num_heads=spec.encoder_heads,
            dropout=spec.attention_dropout, use_rope=spec.use_rope,
            rope_base=spec.rope_base)
        self.norm2 = RMSNorm(dim, spec.norm_eps)
        self.ffn = FeedForward(dim, spec.encoder_ffn_dim, spec.dropout)
        self.dropout = nn.Dropout(spec.dropout)
        self.gate: Optional[CopyGate] = CopyGate(dim) if spec.gated else None
        # Deterministic zero init: at step 0 this is exactly the unbiased layer.
        self.sparse_attention_bias = nn.Parameter(torch.zeros(spec.encoder_heads))

    def key_bias(self, zero_indicator: Optional[Tensor]) -> Optional[Tensor]:
        """Additive attention bias of shape ``(batch, heads, 1, seq)``."""
        if zero_indicator is None:
            return None
        return (self.sparse_attention_bias.view(1, -1, 1, 1)
                * zero_indicator[:, None, None, :])

    def forward(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor] = None,
        zero_indicator: Optional[Tensor] = None,
    ) -> Tensor:
        """Run one encoder pass.

        Args:
            x: Tensor of shape ``(batch, seq, encoder_dim)``.
            attn_mask: Optional boolean key mask.
            zero_indicator: Per-position zero flags, ``(batch, seq)``.

        Returns:
            Tensor of the same shape as ``x``.
        """
        residual = x
        h = x + self.dropout(self.self_attn(
            self.norm1(x), attn_mask=attn_mask,
            key_bias=self.key_bias(zero_indicator)))
        h = h + self.ffn(self.norm2(h))
        return self.gate(residual, h) if self.gate is not None else h


class SalsaNact(nn.Module):
    """Salsa2-NACT V2.  Same public interface as :class:`SalsaTransformer`.

    Example:
        >>> model = SalsaNact(NactSpec())                    # doctest: +SKIP
        >>> src = torch.randint(4, 85, (2, 26))              # V1 layout, n=12
        >>> model(src, torch.randint(4, 85, (2, 4))).shape   # doctest: +SKIP
        torch.Size([2, 4, 85])
    """

    def __init__(self, spec: NactSpec) -> None:
        """Build the model described by ``spec``."""
        super().__init__()
        self.spec = spec

        self.front_end = NactFrontEnd(spec)
        self.decoder_embedding = TokenEmbedding(
            spec.vocab_size, spec.decoder_dim, padding_idx=spec.pad_id)
        self.embedding_dropout = nn.Dropout(spec.dropout)

        self.encoder_layers = nn.ModuleList(
            NactEncoderLayer(spec) for _ in range(spec.encoder_layers))
        self.decoder_layers = nn.ModuleList(
            DecoderLayer(spec) for _ in range(spec.decoder_layers))
        self.encoder_norm = RMSNorm(spec.encoder_dim, spec.norm_eps)
        self.decoder_norm = RMSNorm(spec.decoder_dim, spec.norm_eps)

        self.output_projection: Optional[nn.Linear]
        if spec.tie_embeddings:
            self.output_projection = None
        else:
            self.output_projection = nn.Linear(
                spec.decoder_dim, spec.vocab_size, bias=False)
            nn.init.xavier_uniform_(self.output_projection.weight)

    # -- properties --------------------------------------------------------- #
    def describe(self) -> Dict[str, Any]:
        """Return a serialisable description for run metadata."""
        description = self.spec.to_dict()
        description.update({
            "name": self.name,
            "trainable_parameters": sum(
                p.numel() for p in self.parameters() if p.requires_grad),
            "norm": "RMSNorm",
            "activation": "gelu",
            "attention_bias": False,
            "positional_encoding": "rope" if self.spec.use_rope else "none",
            "cross_attention_rope": False,
            "input_front_end": "coordinate tokens with numerical features",
            "encoder_tokens_per_coordinate": 1,
            "sparse_attention_bias": "learned, one scalar per encoder head, zero-init",
        })
        return description

    @property
    def name(self) -> str:
        """Human-readable model name, as used in run metadata."""
        return "Salsa2-NACT"

    @property
    def vocab_size(self) -> int:
        """Vocabulary size the model was built for."""
        return self.spec.vocab_size

    @property
    def device(self) -> torch.device:
        """Device currently holding the parameters."""
        return next(self.parameters()).device

    def encoder_layer_for(self, step: int) -> NactEncoderLayer:
        """Return the encoder module used at pass ``step``."""
        return self.encoder_layers[step % len(self.encoder_layers)]

    def decoder_layer_for(self, step: int) -> DecoderLayer:
        """Return the decoder module used at pass ``step``."""
        return self.decoder_layers[step % len(self.decoder_layers)]

    # -- forward ------------------------------------------------------------ #
    def encode(self, src_ids: Tensor, src_valid: Optional[Tensor] = None) -> Tensor:
        """Encode the source sequence.

        Args:
            src_ids: The codec's input ids, ``(batch, L_v1)``.
            src_valid: Optional validity mask.  Must already be in NACT's own
                ``n + 2`` coordinate frame; the V1 token frame has a different
                length and is rejected rather than silently misapplied.

        Returns:
            Encoder memory of shape ``(batch, n + 2, encoder_dim)``.

        Raises:
            ValueError: If ``src_valid`` has the wrong length.
        """
        if src_ids.dim() != 2:
            raise ValueError(f"src_ids must be (batch, seq), got {tuple(src_ids.shape)}.")
        x, zero_indicator = self.front_end(src_ids)
        if src_valid is not None and src_valid.shape[-1] != x.shape[1]:
            raise ValueError(
                f"src_valid has length {src_valid.shape[-1]} but NACT's encoder "
                f"sequence is {x.shape[1]} long. Pass a mask in the coordinate "
                "frame, or None.")
        x = self.embedding_dropout(x)
        mask = build_key_padding_mask(src_valid, x.shape[1])
        for step in range(self.spec.encoder_loops):
            x = self.encoder_layer_for(step)(
                x, attn_mask=mask, zero_indicator=zero_indicator)
        return self.encoder_norm(x)

    def decode(
        self,
        memory: Tensor,
        tgt_ids: Tensor,
        src_valid: Optional[Tensor] = None,
        causal: bool = True,
    ) -> Tensor:
        """Decode ``b`` given the encoder output, teacher-forced.

        Byte-for-byte the same logic as V1's decoder -- same modules, same
        masks, same output vocabulary -- so ``greedy_decode`` and the trainer
        call it with no change.

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
                f"{tuple(memory.shape)}.")
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

        Args:
            src_ids: The codec's input ids, ``(batch, L_v1)``.
            tgt_ids: Encoded ``b``, ``(batch, tgt_len)``.
            src_valid: Optional encoder-side validity mask, in the coordinate frame.
            causal: Apply the causal mask on decoder self-attention.

        Returns:
            Logits of shape ``(batch, tgt_len, vocab_size)``.
        """
        memory = self.encode(src_ids, src_valid=src_valid)
        return self.decode(memory, tgt_ids, src_valid=src_valid, causal=causal)


def build_nact(config, vocab_size: Optional[int] = None) -> SalsaNact:
    """Build a NACT model from an experiment configuration."""
    return SalsaNact(NactSpec.from_config(config, vocab_size=vocab_size))
