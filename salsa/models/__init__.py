"""Compact CPU-first models for the SALSA ``a -> b`` task, and their accounting.

Two architectures, both inside the 4-5M trainable-parameter budget at V=85:

``gated_universal_transformer`` -- the approved primary (phase-4 candidate B).
    Encoder 512 wide / 8 heads / **1 shared layer** run ``encoder_loops`` times;
    decoder 128 wide / 4 heads / 1 shared layer run ``decoder_loops`` times;
    learned copy gate on every layer.  4,131,200 parameters, independent of the
    loop counts.

``nact`` -- Salsa2-NACT V2 (phase 14).  The V1 gated encoder body and decoder
    unchanged, with the digit-token input front end replaced by one learned
    coordinate representation carrying numerical features, an absolute
    coordinate embedding and a learned per-head bias on zero-coordinate keys.
    4,241,288 parameters, likewise independent of the loop counts and of ``n``.

``compact_transformer`` -- the matched-budget control (candidate A).
    Encoder 384 wide / 6 heads / 2 distinct layers; decoder 128 wide / 4 heads /
    2 distinct layers; no sharing, no gate.  4,251,520 parameters.

Both use RMSNorm, no biases, GELU feed-forward and RoPE, so the parameter count
is invariant in the lattice dimension ``n`` and in the choice of representation
R or P -- one checkpoint covers n = 30, 50, 70, 90, 128.

Typical use::

    from salsa.models import build_model, parameter_report, format_parameter_report
    from salsa.utils import load_config

    config = load_config("configs/target_4_5m.yaml")
    model = build_model(config)
    print(format_parameter_report(parameter_report(
        model, config.model.max_parameters, config.model.min_parameters)))
"""

from .attention import MultiHeadAttention, build_causal_mask, build_key_padding_mask
from .embeddings import RotaryPositionalEmbedding, TokenEmbedding, apply_rope
from .parameter_count import (
    COMPONENTS,
    ParameterReport,
    analytical_breakdown,
    analytical_parameter_count,
    check_budget,
    count_all_parameters,
    count_trainable_parameters,
    format_parameter_report,
    parameter_breakdown,
    parameter_report,
    report_from_config,
    unclassified_parameters,
)
from .nact import (
    NUMERICAL_FEATURES,
    NactEncoderLayer,
    NactFrontEnd,
    NactSpec,
    SalsaNact,
    build_nact,
)
from .transformer import (
    ARCHITECTURES,
    CopyGate,
    DecoderLayer,
    EncoderLayer,
    FeedForward,
    ModelSpec,
    RMSNorm,
    SalsaTransformer,
    build_model,
)

__all__ = [
    # building blocks
    "TokenEmbedding",
    "RotaryPositionalEmbedding",
    "apply_rope",
    "MultiHeadAttention",
    "build_causal_mask",
    "build_key_padding_mask",
    "RMSNorm",
    "FeedForward",
    "CopyGate",
    "EncoderLayer",
    "DecoderLayer",
    # models
    "ARCHITECTURES",
    "ModelSpec",
    "SalsaTransformer",
    "build_model",
    # NACT V2
    "NactSpec",
    "NactFrontEnd",
    "NactEncoderLayer",
    "SalsaNact",
    "build_nact",
    "NUMERICAL_FEATURES",
    # accounting
    "COMPONENTS",
    "ParameterReport",
    "analytical_breakdown",
    "analytical_parameter_count",
    "check_budget",
    "count_all_parameters",
    "count_trainable_parameters",
    "format_parameter_report",
    "parameter_breakdown",
    "parameter_report",
    "report_from_config",
    "unclassified_parameters",
]
