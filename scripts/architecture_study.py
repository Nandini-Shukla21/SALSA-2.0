"""Phase 14: architecture study for Salsa 2.0 v2.  Design only -- nothing is trained.

No model is built or modified here except to *verify* that the analytical
parameter formulas reproduce the shipped 4,131,200-parameter GatedUT exactly.
That check is the anchor for every other count in this file: a formula that can
reproduce the real model is a formula that can be trusted for a model that does
not exist yet.

What is computed rather than asserted
-------------------------------------
- Exact parameter counts and breakdowns for every candidate.
- Sequence length, MACs and training activation memory at n in {12,30,50,70,90,128}
  under both representations R and P.
- The attention-versus-projection crossover, which decides whether a state-space
  component is motivated at all (hypothesis H4).
- The ordinal content of each digit position in the base-81 lsb-first encoding,
  which decides which numerical features are meaningful and which are redundant.

Everything here is arithmetic on public quantities.  No secret, no checkpoint,
no training data.
"""

import csv
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

OUTPUT_DIR = REPO_ROOT / "results" / "architecture_v2"

BYTES_FP32 = 4
V = 85               # Salsa 2.0 vocabulary: PAD BOS EOS SEP + 81 digits
BASE = 81
Q = 251
WIDTH = 2            # digits per integer at base 81 over Z_251
N_MAX = 128          # largest dimension the v2 design must support
DIMENSIONS = (12, 30, 50, 70, 90, 128)
BUDGET = (4_000_000, 5_000_000)


# --------------------------------------------------------------------------- #
# Sequence lengths
# --------------------------------------------------------------------------- #
def input_length(n: int, representation: str, tokens_per_coordinate: int) -> int:
    """Encoder input length for a dimension and representation.

    R packs ``width`` digit tokens per coordinate with no separator; P adds one
    separator per coordinate.  A coordinate-token front end collapses each
    coordinate to a single position, which is what ``tokens_per_coordinate=1``
    expresses.

    Args:
        n: Lattice dimension.
        representation: ``"R"`` or ``"P"``.
        tokens_per_coordinate: Positions the model spends per coordinate.

    Returns:
        Number of encoder positions including ``<bos>`` and ``<eos>``.

    Raises:
        ValueError: On an unknown representation.
    """
    if representation == "R":
        return n * tokens_per_coordinate + 2
    if representation == "P":
        # One separator per coordinate; a coordinate-token front end absorbs it.
        extra = 1 if tokens_per_coordinate > 1 else 0
        return n * (tokens_per_coordinate + extra) + 1
    raise ValueError(f"representation must be 'R' or 'P', got {representation!r}.")


# --------------------------------------------------------------------------- #
# Candidate description
# --------------------------------------------------------------------------- #
@dataclass
class Candidate:
    """One architecture under consideration.

    Attributes:
        key: Short identifier.
        name: Display name.
        mixing: How positions exchange information.
        encoder_dim: Encoder width.
        decoder_dim: Decoder width.
        encoder_heads: Attention heads in the encoder (0 if no attention).
        decoder_heads: Attention heads in the decoder.
        encoder_sets: Distinct encoder parameter sets.
        decoder_sets: Distinct decoder parameter sets.
        encoder_loops: Encoder passes (free: reuses the parameter sets).
        decoder_loops: Decoder passes.
        ffn_multiplier: FFN hidden width as a multiple of the model width.
        gated: Whether a copy gate is present.
        tokens_per_coordinate: 2 for digit tokens, 1 for a coordinate token.
        numerical_front_end: Whether the numerical feature front end is present.
        conv_kernel: Depthwise convolution kernel width, 0 if absent.
        glu_multiplier: Gated-MLP hidden multiple, 0 if absent.
        ssm_state: State-space state size, 0 if absent.
    """

    key: str
    name: str
    mixing: str
    encoder_dim: int
    decoder_dim: int
    encoder_heads: int
    decoder_heads: int
    encoder_sets: int = 1
    decoder_sets: int = 1
    encoder_loops: int = 2
    decoder_loops: int = 2
    ffn_multiplier: float = 4.0
    gated: bool = True
    tokens_per_coordinate: int = 2
    numerical_front_end: bool = False
    conv_kernel: int = 0
    glu_multiplier: float = 0.0
    ssm_state: int = 0
    notes: str = ""


#: Numerical features, and whether they survive the redundancy analysis.
#: A feature is kept only if it is (a) a function of the public input alone,
#: (b) not an affine function of a feature already present, and (c) tied to a
#: measured failure mode rather than to a hunch.
NUMERICAL_FEATURES = [
    {
        "feature": "signed centered residue  ((x + q//2) mod q - q//2) / (q/2)",
        "count": 1,
        "verdict": "KEEP",
        "reason": (
            "b = sum_{i in supp} a_i + e mod q is integer addition followed by one "
            "reduction, so the integer magnitude of a_i is the quantity the model "
            "must actually add. Centering puts it symmetric about zero, which suits "
            "RMSNorm and bias-free FFNs, and matches the centered error term."),
    },
    {
        "feature": "normalized residue x/q",
        "count": 0,
        "verdict": "REJECT - redundant",
        "reason": (
            "An affine function of the signed centered residue. A linear layer "
            "converts one to the other, so including both adds parameters and no "
            "information."),
    },
    {
        "feature": "Fourier pair(s) cos/sin(2*pi*k*x/q), k = 1..K",
        "count": 2,
        "verdict": "KEEP (K=1 only)",
        "reason": (
            "The characters of Z_q are the irreducible representations of the group "
            "the task actually lives in: addition mod q becomes addition of angles. "
            "This is the one genuinely non-affine feature that encodes wraparound, "
            "which no polynomial in x can express. K=1 is kept; higher harmonics are "
            "left for the model to build, since q=251 is prime and the k=1 pair "
            "already separates every residue."),
    },
    {
        "feature": "modular distance to zero  min(x, q-x)",
        "count": 0,
        "verdict": "REJECT - redundant",
        "reason": (
            "Determined by the k=1 Fourier pair: min(x, q-x) is a monotone function "
            "of cos(2*pi*x/q). Adding it separately is duplication."),
    },
    {
        "feature": "magnitude feature |centered(x)|",
        "count": 0,
        "verdict": "REJECT - duplicate",
        "reason": "It is the modular distance to zero under another name.",
    },
    {
        "feature": "zero / nonzero indicator  1[x == 0]",
        "count": 1,
        "verdict": "KEEP",
        "reason": (
            "The single most defensible sparsity feature. In base-81 lsb-first, "
            "'this coordinate is zero' is a CONJUNCTION across two token positions "
            "(digit0 == 0 AND digit1 == 0), which the encoder must spend attention "
            "to compute. Phase 12 measured that the model stops using its input "
            "somewhere between 8 and 6 nonzero coordinates; this makes the defining "
            "bit of that regime a free input rather than something to be inferred."),
    },
    {
        "feature": "absolute coordinate index embedding",
        "count": 0,
        "verdict": "KEEP as an embedding table, not a scalar",
        "reason": (
            "b = sum_i a_i s_i binds coordinate i to secret bit s_i, so the task "
            "needs ABSOLUTE coordinate identity. RoPE supplies only RELATIVE "
            "position. This is a real gap in the current model and it costs "
            "n_max * d, once, independent of n."),
    },
    {
        "feature": "digit embedding (existing LatticeCodec tokens)",
        "count": 0,
        "verdict": "KEEP",
        "reason": (
            "Retained unchanged so the v2 model stays compatible with the existing "
            "codec, the existing target format and the existing recovery interface, "
            "and so a token-only ablation arm remains available."),
    },
]

KEPT_SCALAR_FEATURES = sum(f["count"] for f in NUMERICAL_FEATURES)


# --------------------------------------------------------------------------- #
# Parameter accounting
# --------------------------------------------------------------------------- #
def parameter_breakdown(c: Candidate) -> Dict[str, int]:
    """Exact trainable-parameter breakdown for a candidate.

    The transformer terms are the formulas already validated against the shipped
    model in :mod:`salsa.models.parameter_count`; the new terms are stated
    explicitly so each can be checked by hand.

    Args:
        c: The candidate.

    Returns:
        A component -> count mapping.  No padding terms; every entry is a real
        tensor the implementation would have to allocate.
    """
    de, dd = c.encoder_dim, c.decoder_dim
    le, ld = c.encoder_sets, c.decoder_sets
    fe, fd = int(c.ffn_multiplier * de), int(c.ffn_multiplier * dd)
    parts: Dict[str, int] = {}

    # -- input front end ---------------------------------------------------- #
    if c.numerical_front_end:
        # One embedding table per digit slot, so slot 0 and slot 1 are
        # distinguishable without spending a sequence position on the difference.
        parts["encoder_digit_embeddings"] = WIDTH * BASE * de
        parts["encoder_special_embeddings"] = 4 * de          # pad bos eos sep
        parts["numerical_feature_projection"] = KEPT_SCALAR_FEATURES * de + de
        parts["coordinate_embedding"] = N_MAX * de
        parts["zero_coordinate_vector"] = de
        parts["sparse_attention_bias"] = c.encoder_heads      # one scalar per head
    else:
        parts["encoder_embedding"] = V * de

    # -- encoder ------------------------------------------------------------ #
    if c.encoder_heads:
        parts["encoder_attention"] = le * 4 * de * de
    if c.conv_kernel:
        parts["encoder_depthwise_conv"] = le * (c.conv_kernel * de + de)
    if c.ssm_state:
        # Diagonal state-space block: in/out projections plus A, B, C, D.
        parts["encoder_ssm"] = le * (2 * de * de + 3 * de * c.ssm_state + de)
    if c.glu_multiplier:
        g = int(c.glu_multiplier * de)
        parts["encoder_gated_mlp"] = le * 3 * de * g
    else:
        parts["encoder_ffn"] = le * 2 * de * fe
    if c.gated:
        parts["encoder_copy_gate"] = le * (2 * de * de + de)
    norms = 2 + (1 if c.conv_kernel else 0) + (1 if c.ssm_state else 0)
    parts["encoder_normalization"] = le * norms * de + de

    # -- decoder (unchanged across candidates: it is 10% of the budget) ------ #
    parts["decoder_embedding"] = V * dd
    parts["decoder_self_attention"] = ld * 4 * dd * dd
    parts["decoder_cross_attention"] = ld * (2 * dd * dd + 2 * de * dd)
    parts["decoder_ffn"] = ld * 2 * dd * fd
    if c.gated:
        parts["decoder_copy_gate"] = ld * (2 * dd * dd + dd)
    parts["decoder_normalization"] = ld * 3 * dd + dd
    parts["output_projection"] = dd * V
    return parts


def total_parameters(c: Candidate) -> int:
    """Total trainable parameters for a candidate."""
    return sum(parameter_breakdown(c).values())


# --------------------------------------------------------------------------- #
# Compute and memory
# --------------------------------------------------------------------------- #
def encoder_macs_per_pass(c: Candidate, length: int) -> Dict[str, int]:
    """Multiply-accumulates for one encoder pass over ``length`` positions.

    Reported as MACs; FLOPs are twice this.  Elementwise work (norms, GELU,
    sigmoids) is excluded: it is O(L d) with a small constant and does not change
    any of the conclusions drawn from these numbers.
    """
    de = c.encoder_dim
    out: Dict[str, int] = {}
    if c.encoder_heads:
        out["qkv_o_projections"] = 4 * length * de * de
        out["attention_scores_and_values"] = 2 * length * length * de
    if c.conv_kernel:
        out["depthwise_conv"] = length * de * c.conv_kernel
    if c.ssm_state:
        out["ssm"] = 2 * length * de * de + 3 * length * de * c.ssm_state
    if c.glu_multiplier:
        out["gated_mlp"] = 3 * length * de * int(c.glu_multiplier * de)
    else:
        out["ffn"] = 2 * length * de * int(c.ffn_multiplier * de)
    if c.gated:
        out["copy_gate"] = 2 * length * de * de
    return out


def encoder_activation_floats(c: Candidate, length: int) -> int:
    """Floats retained for backward, per sample, across all encoder passes.

    Accounting, per pass and per position, in units of the model width:
    normalised input, q, k, v, attention output, output projection, second
    norm, FFN output and the two copy-gate tensors -> 11 d; the FFN keeps its
    hidden activation before and after GELU -> 2 f.  Attention additionally
    keeps one probability matrix per head -> H L^2.  Loops are unrolled, so
    every pass contributes.
    """
    de = c.encoder_dim
    fe = int(c.glu_multiplier * de) if c.glu_multiplier else int(c.ffn_multiplier * de)
    per_pass = 11 * length * de + 2 * length * fe
    if c.encoder_heads:
        per_pass += c.encoder_heads * length * length
    if c.conv_kernel:
        per_pass += length * de
    if c.ssm_state:
        per_pass += length * de * (1 + c.ssm_state)
    return per_pass * c.encoder_loops


def decoder_activation_floats(c: Candidate, out_length: int, src_length: int) -> int:
    """Floats retained for backward in the decoder, per sample."""
    dd = c.decoder_dim
    fd = int(c.ffn_multiplier * dd)
    per_pass = (
        14 * out_length * dd
        + 2 * out_length * fd
        + c.decoder_heads * out_length * out_length      # self-attention probs
        + c.decoder_heads * out_length * src_length      # cross-attention probs
    )
    return per_pass * c.decoder_loops


# --------------------------------------------------------------------------- #
# Evidence for the representation analysis
# --------------------------------------------------------------------------- #
def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation with tie-averaged ranks."""
    def ranks(values: np.ndarray) -> np.ndarray:
        order = np.argsort(values, kind="mergesort")
        result = np.empty(values.size, dtype=np.float64)
        result[order] = np.arange(1, values.size + 1, dtype=np.float64)
        for value in np.unique(values):
            mask = values == value
            if mask.sum() > 1:
                result[mask] = result[mask].mean()
        return result

    rx, ry = ranks(np.asarray(x, float)), ranks(np.asarray(y, float))
    rx, ry = rx - rx.mean(), ry - ry.mean()
    denominator = math.sqrt(float((rx ** 2).sum()) * float((ry ** 2).sum()))
    return float((rx * ry).sum() / denominator) if denominator else 0.0


def encoding_evidence() -> Dict[str, Any]:
    """Measure what each digit position of the base-81 lsb-first code carries.

    This is the factual basis for the numerical-feature recommendation: if the
    first token the encoder reads per coordinate carries almost no ordinal
    information about the value, then supplying the value numerically is not
    feature engineering, it is repairing a representation defect.
    """
    values = np.arange(Q, dtype=np.int64)
    digit0 = values % BASE               # least significant, read FIRST
    digit1 = values // BASE              # most significant, read SECOND

    zero_digit0 = int((digit0 == 0).sum())
    zero_digit1 = int((digit1 == 0).sum())
    both_zero = int(((digit0 == 0) & (digit1 == 0)).sum())

    ring = np.minimum(values, Q - values)
    return {
        "modulus": Q, "base": BASE, "digits_per_integer": WIDTH,
        "spearman_value_vs_digit0_lsb_read_first": round(_spearman(values, digit0), 6),
        "spearman_value_vs_digit1_msb_read_second": round(_spearman(values, digit1), 6),
        "spearman_ring_distance_vs_digit0": round(_spearman(ring, digit0), 6),
        "spearman_ring_distance_vs_digit1": round(_spearman(ring, digit1), 6),
        "residues_with_digit0_equal_zero": zero_digit0,
        "residues_with_digit1_equal_zero": zero_digit1,
        "residues_with_both_digits_zero": both_zero,
        "probability_digit0_zero": round(zero_digit0 / Q, 6),
        "probability_digit1_zero": round(zero_digit1 / Q, 6),
        "zero_detection_is_a_conjunction_across_positions": both_zero == 1,
        "interpretation": (
            "The least significant digit is read FIRST under lsb_first and its rank "
            "correlation with the value it encodes is essentially zero, so the first "
            "token per coordinate is close to ordinally uninformative. Detecting "
            "'this coordinate is zero' requires both positions to be zero at once: "
            f"digit0 == 0 alone happens for {zero_digit0} of {Q} residues and "
            f"digit1 == 0 alone for {zero_digit1}, but only {both_zero} residue "
            "satisfies both. That conjunction is what the encoder currently has to "
            "spend attention on, once per coordinate."),
    }


def sparsity_token_statistics() -> List[Dict[str, Any]]:
    """Fraction of encoder positions occupied by zero digits, by sparsity.

    Under representation R a zero coordinate contributes ``width`` zero-digit
    tokens, so the encoder's input becomes overwhelmingly one repeated symbol as
    the input gets sparser.  This is the attention-dilution problem, quantified.
    """
    rows = []
    for n in DIMENSIONS:
        for nnz in (n, max(1, n // 2), max(1, n // 8), 1):
            zeros = n - nnz
            length_r = input_length(n, "R", WIDTH)
            rows.append({
                "n": n, "nnz": nnz,
                "zero_coordinates": zeros,
                "R_sequence_length": length_r,
                "R_zero_digit_tokens": zeros * WIDTH,
                "R_fraction_zero_digit_tokens": round(zeros * WIDTH / length_r, 6),
                "coordinate_token_length": input_length(n, "R", 1),
                "coordinate_token_fraction_zero": round(zeros / (n + 2), 6),
            })
    return rows


# --------------------------------------------------------------------------- #
# Candidates
# --------------------------------------------------------------------------- #
def build_candidates() -> List[Candidate]:
    """The architectures compared in this phase."""
    return [
        Candidate(
            key="A", name="Salsa2-GatedUT (current baseline)",
            mixing="full self-attention",
            encoder_dim=512, decoder_dim=128, encoder_heads=8, decoder_heads=4,
            encoder_loops=2, decoder_loops=2,
            tokens_per_coordinate=WIDTH, numerical_front_end=False,
            notes="Unchanged. Digit tokens, RoPE only, no numerical features.",
        ),
        Candidate(
            key="B", name="Salsa2-NACT (numerical-aware compact transformer)",
            mixing="full self-attention + sparse-aware attention bias",
            encoder_dim=512, decoder_dim=128, encoder_heads=8, decoder_heads=4,
            encoder_loops=4, decoder_loops=2,
            tokens_per_coordinate=1, numerical_front_end=True,
            notes="One token per coordinate; digit embeddings retained and "
                  "augmented with the kept numerical features; absolute "
                  "coordinate embedding; per-head zero-coordinate attention bias.",
        ),
        Candidate(
            key="B-lite", name="Salsa2-NACT at T_e=2 (identical parameters to B)",
            mixing="full self-attention + sparse-aware attention bias",
            encoder_dim=512, decoder_dim=128, encoder_heads=8, decoder_heads=4,
            encoder_loops=2, decoder_loops=2,
            tokens_per_coordinate=1, numerical_front_end=True,
            notes="Same weights as B. Loops are a runtime knob with no parameter "
                  "cost, so this row shows what the shorter sequence buys in "
                  "compute and memory when depth is held equal to A.",
        ),
        Candidate(
            key="C", name="Salsa2-Hybrid (depthwise conv + gated MLP + attention)",
            mixing="depthwise conv (local) + full self-attention (global)",
            encoder_dim=512, decoder_dim=128, encoder_heads=4, decoder_heads=4,
            encoder_loops=4, decoder_loops=2,
            tokens_per_coordinate=1, numerical_front_end=True,
            conv_kernel=5, glu_multiplier=2.5,
            notes="Same front end as B. FFN replaced by a gated MLP and a "
                  "depthwise convolution for cheap local mixing.",
        ),
        Candidate(
            key="C-alt", name="Salsa2-SSM (state-space instead of attention)",
            mixing="diagonal state-space scan, no attention in the encoder",
            encoder_dim=512, decoder_dim=128, encoder_heads=0, decoder_heads=4,
            encoder_loops=4, decoder_loops=2,
            tokens_per_coordinate=1, numerical_front_end=True,
            ssm_state=16, glu_multiplier=3.0,
            notes="Included to TEST hypothesis H4 rather than assume it. "
                  "Gated-MLP multiplier raised 2.5 -> 3.0, the smallest widening "
                  "that clears the 4M floor, so the comparison is budget-matched "
                  "rather than won on size. "
                  "Encoder has no content-based global mixing at all.",
        ),
    ]


# --------------------------------------------------------------------------- #
# Scaling
# --------------------------------------------------------------------------- #
def scaling_rows(candidates: List[Candidate], out_length: int = 4) -> List[Dict[str, Any]]:
    """Sequence length, compute and activation memory at every dimension."""
    rows = []
    for c in candidates:
        for representation in ("R", "P"):
            for n in DIMENSIONS:
                length = input_length(n, representation, c.tokens_per_coordinate)
                macs = encoder_macs_per_pass(c, length)
                attention_macs = macs.get("attention_scores_and_values", 0)
                projection_macs = sum(v for k, v in macs.items()
                                      if k != "attention_scores_and_values")
                total_macs = (sum(macs.values()) * c.encoder_loops)
                activations = (encoder_activation_floats(c, length)
                               + decoder_activation_floats(c, out_length, length))
                rows.append({
                    "section": "scaling",
                    "candidate": c.key,
                    "representation": representation,
                    "n": n,
                    "encoder_sequence_length": length,
                    "encoder_macs_per_sample": total_macs,
                    "encoder_gmacs_per_sample": round(total_macs / 1e9, 4),
                    "attention_share_of_encoder_macs": round(
                        attention_macs / max(attention_macs + projection_macs, 1), 6),
                    "activation_floats_per_sample": activations,
                    "activation_mib_batch_64": round(
                        activations * 64 * BYTES_FP32 / 2 ** 20, 3),
                })
    return rows


def attention_crossover(dim: int) -> Dict[str, Any]:
    """Sequence length at which attention cost equals projection cost.

    Per pass the projections cost ``4 L d^2`` MACs and the attention product
    costs ``2 L^2 d``.  They are equal at ``L = 2 d``.  Below that, attention is
    not the bottleneck and replacing it buys nothing.
    """
    crossover = 2 * dim
    return {
        "encoder_dim": dim,
        "crossover_sequence_length": crossover,
        "equivalent_n_representation_R_digit_tokens": (crossover - 2) // WIDTH,
        "equivalent_n_coordinate_tokens": crossover - 2,
        "derivation": "4*L*d^2 == 2*L^2*d  =>  L == 2*d",
    }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    """Run the study and write the artifacts."""
    # -- anchor: the formulas must reproduce the shipped model exactly ------- #
    from salsa.models.parameter_count import analytical_parameter_count
    from salsa.models.transformer import ModelSpec

    shipped = analytical_parameter_count(ModelSpec())
    candidates = build_candidates()
    baseline = total_parameters(candidates[0])
    assert shipped == 4_131_200, f"library formula drifted: {shipped}"
    assert baseline == 4_131_200, (
        f"this script's formula gives {baseline} for candidate A, "
        f"but the shipped model is 4,131,200")

    print("=" * 92)
    print("PHASE 14 - ARCHITECTURE STUDY FOR SALSA 2.0 v2   (design only, nothing trained)")
    print("=" * 92)
    print(f"  formula anchor: candidate A reproduces the shipped model exactly "
          f"({baseline:,} parameters)")
    print()

    print("-" * 92)
    print("PARAMETER COUNTS")
    print("-" * 92)
    print(f"  {'cand':<6}{'name':<52}{'params':>12}  budget")
    candidate_rows = []
    for c in candidates:
        parts = parameter_breakdown(c)
        total = sum(parts.values())
        status = ("OK" if BUDGET[0] <= total <= BUDGET[1]
                  else "FAILED BUDGET" if total > BUDGET[1] else "UNDER TARGET")
        print(f"  {c.key:<6}{c.name:<52}{total:>12,}  {status}")
        candidate_rows.append({"candidate": c, "parts": parts,
                               "total": total, "status": status})
    print()

    for row in candidate_rows:
        c, parts, total = row["candidate"], row["parts"], row["total"]
        print(f"  {c.key} breakdown:")
        for name, value in sorted(parts.items(), key=lambda kv: -kv[1]):
            if value:
                print(f"      {name:<34}{value:>11,}  {100 * value / total:5.1f}%")
        print(f"      {'TOTAL':<34}{total:>11,}   fp32 "
              f"{total * BYTES_FP32 / 2 ** 20:.2f} MiB")
        print()

    crossover = attention_crossover(512)
    print("-" * 92)
    print("H4 TEST - is attention actually the bottleneck at these lengths?")
    print("-" * 92)
    print(f"  projections cost 4*L*d^2, the attention product costs 2*L^2*d; "
          f"equal at L = 2d = {crossover['crossover_sequence_length']}")
    print(f"  that is n = {crossover['equivalent_n_representation_R_digit_tokens']} "
          f"under R with digit tokens, or n = "
          f"{crossover['equivalent_n_coordinate_tokens']} with coordinate tokens")
    print()

    scaling = scaling_rows(candidates)
    print("-" * 92)
    print("SCALING (representation R)")
    print("-" * 92)
    print(f"  {'cand':<7}{'n':>5}{'L_in':>7}{'GMACs/sample':>15}"
          f"{'attn share':>12}{'act MiB @bs64':>15}")
    for row in scaling:
        if row["representation"] == "R":
            print(f"  {row['candidate']:<7}{row['n']:>5}{row['encoder_sequence_length']:>7}"
                  f"{row['encoder_gmacs_per_sample']:>15.4f}"
                  f"{row['attention_share_of_encoder_macs']:>12.4f}"
                  f"{row['activation_mib_batch_64']:>15.1f}")
    print()

    evidence = encoding_evidence()
    print("-" * 92)
    print("ENCODING EVIDENCE (base 81, lsb_first, width 2 over Z_251)")
    print("-" * 92)
    print(f"  Spearman(value, digit0 read first)  = "
          f"{evidence['spearman_value_vs_digit0_lsb_read_first']:+.4f}")
    print(f"  Spearman(value, digit1 read second) = "
          f"{evidence['spearman_value_vs_digit1_msb_read_second']:+.4f}")
    print(f"  digit0 == 0 for {evidence['residues_with_digit0_equal_zero']}/{Q} residues, "
          f"digit1 == 0 for {evidence['residues_with_digit1_equal_zero']}/{Q}, "
          f"both for {evidence['residues_with_both_digits_zero']}")
    print()

    payload = {
        "phase": "14 - next-generation architecture study",
        "status": "DESIGN ONLY. Nothing trained. No model, generator, checkpoint or "
                  "recovery code modified. No model code written.",
        "formula_anchor": {
            "shipped_gatedut_parameters": shipped,
            "reproduced_by_this_script": baseline,
            "agree": shipped == baseline,
            "why_this_matters": ("Every count below comes from the same formulas that "
                                 "reproduce the real model exactly, so the counts for "
                                 "models that do not exist yet are trustworthy."),
        },
        "budget": {"min": BUDGET[0], "max": BUDGET[1],
                   "no_padding_parameters": True},
        "candidates": [
            {
                "key": row["candidate"].key,
                "name": row["candidate"].name,
                "mixing": row["candidate"].mixing,
                "notes": row["candidate"].notes,
                "encoder_dim": row["candidate"].encoder_dim,
                "decoder_dim": row["candidate"].decoder_dim,
                "encoder_heads": row["candidate"].encoder_heads,
                "decoder_heads": row["candidate"].decoder_heads,
                "encoder_parameter_sets": row["candidate"].encoder_sets,
                "decoder_parameter_sets": row["candidate"].decoder_sets,
                "encoder_loops": row["candidate"].encoder_loops,
                "decoder_loops": row["candidate"].decoder_loops,
                "tokens_per_coordinate": row["candidate"].tokens_per_coordinate,
                "numerical_front_end": row["candidate"].numerical_front_end,
                "parameter_breakdown": row["parts"],
                "parameter_count": row["total"],
                "fp32_size_mib": round(row["total"] * BYTES_FP32 / 2 ** 20, 3),
                "budget_status": row["status"],
            }
            for row in candidate_rows
        ],
        "attention_crossover": crossover,
        "scaling": scaling,
        "encoding_evidence": evidence,
        "sparsity_token_statistics": sparsity_token_statistics(),
        "numerical_features": NUMERICAL_FEATURES,
        "kept_scalar_feature_count": KEPT_SCALAR_FEATURES,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "architecture_comparison.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")

    columns = ["section", "candidate", "representation", "n", "encoder_sequence_length",
               "encoder_macs_per_sample", "encoder_gmacs_per_sample",
               "attention_share_of_encoder_macs", "activation_floats_per_sample",
               "activation_mib_batch_64", "parameter_count", "fp32_size_mib",
               "budget_status", "mixing", "tokens_per_coordinate",
               "numerical_front_end", "encoder_dim", "decoder_dim", "encoder_heads",
               "encoder_loops", "name"]
    with (OUTPUT_DIR / "architecture_comparison.csv").open(
            "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in candidate_rows:
            c = row["candidate"]
            writer.writerow({
                "section": "candidate", "candidate": c.key, "name": c.name,
                "parameter_count": row["total"],
                "fp32_size_mib": round(row["total"] * BYTES_FP32 / 2 ** 20, 3),
                "budget_status": row["status"], "mixing": c.mixing,
                "tokens_per_coordinate": c.tokens_per_coordinate,
                "numerical_front_end": c.numerical_front_end,
                "encoder_dim": c.encoder_dim, "decoder_dim": c.decoder_dim,
                "encoder_heads": c.encoder_heads, "encoder_loops": c.encoder_loops,
            })
        for row in scaling:
            writer.writerow(row)

    write_markdown(payload, OUTPUT_DIR / "architecture_comparison.md")

    print("-" * 92)
    for name in ("architecture_comparison.md", "architecture_comparison.json",
                 "architecture_comparison.csv"):
        print(f"  wrote {OUTPUT_DIR / name}")
    return 0


def write_markdown(payload: Dict[str, Any], path: Path) -> None:
    """Render the comparison document."""
    from textwrap import dedent

    candidates = {c["key"]: c for c in payload["candidates"]}
    scaling = payload["scaling"]
    evidence = payload["encoding_evidence"]
    crossover = payload["attention_crossover"]

    def scaling_table(representation: str) -> List[str]:
        lines = [
            f"| candidate | n | L_in ({representation}) | GMACs/sample | attention share | activation MiB @ batch 64 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for row in scaling:
            if row["representation"] == representation:
                lines.append(
                    f"| {row['candidate']} | {row['n']} | {row['encoder_sequence_length']} | "
                    f"{row['encoder_gmacs_per_sample']:.4f} | "
                    f"{row['attention_share_of_encoder_macs']:.4f} | "
                    f"{row['activation_mib_batch_64']:.1f} |")
        return lines

    lines: List[str] = [
        "# Phase 14 - Salsa 2.0 v2 architecture study",
        "",
        "**Design only.** Nothing was trained. No model, data generator, checkpoint or",
        "recovery code was modified, and no model code was written.",
        "",
        "## Anchor",
        "",
        "Every parameter count in this document comes from formulas that reproduce the",
        f"shipped GatedUT **exactly**: {payload['formula_anchor']['shipped_gatedut_parameters']:,} "
        "predicted, 4,131,200 measured. The script asserts",
        "this before computing anything else, so counts for models that do not exist yet",
        "rest on formulas known to be correct for a model that does.",
        "",
        "## 0. Where the current budget actually goes",
        "",
        "| component | parameters | share |",
        "|---|---:|---:|",
    ]
    baseline_parts = candidates["A"]["parameter_breakdown"]
    baseline_total = candidates["A"]["parameter_count"]
    for name, value in sorted(baseline_parts.items(), key=lambda kv: -kv[1]):
        if value:
            lines.append(f"| {name} | {value:,} | {100 * value / baseline_total:.1f}% |")
    lines += [
        "",
        "**The encoder is 89.9% of the model and the input representation is 1.1%.**",
        "Three encoder tensors -- FFN, attention, copy gate -- are 88.9% of the budget.",
        "Any redesign that spends parameters on the input representation is spending from",
        "the cheapest part of the model, and any redesign that shortens the sequence is",
        "attacking the expensive part without touching the parameter count at all.",
        "",
        "## 1. Candidates",
        "",
        "| key | name | mixing | tokens/coord | encoder | decoder | parameters | fp32 | budget |",
        "|---|---|---|---:|---|---|---:|---:|---|",
    ]
    for key, c in candidates.items():
        lines.append(
            f"| {key} | {c['name']} | {c['mixing']} | {c['tokens_per_coordinate']} | "
            f"d={c['encoder_dim']} H={c['encoder_heads']} sets={c['encoder_parameter_sets']} "
            f"T={c['encoder_loops']} | d={c['decoder_dim']} H={c['decoder_heads']} "
            f"sets={c['decoder_parameter_sets']} T={c['decoder_loops']} | "
            f"{c['parameter_count']:,} | {c['fp32_size_mib']:.2f} MiB | {c['budget_status']} |")

    lines += ["", "### Parameter breakdowns", ""]
    for key, c in candidates.items():
        lines += [f"**{key} - {c['name']}** ({c['parameter_count']:,})", "",
                  "| component | parameters | share |", "|---|---:|---:|"]
        for name, value in sorted(c["parameter_breakdown"].items(), key=lambda kv: -kv[1]):
            if value:
                lines.append(f"| {name} | {value:,} | "
                             f"{100 * value / c['parameter_count']:.1f}% |")
        lines.append("")

    lines += [
        "## 2. Hypothesis H4 - is a state-space component motivated?",
        "",
        "**Answer: no, and this is arithmetic rather than opinion.**",
        "",
        "Per encoder pass the four projections cost `4*L*d^2` MACs and the attention",
        f"product costs `2*L^2*d`. They are equal at `L = 2d = "
        f"{crossover['crossover_sequence_length']}`, which is "
        f"n = {crossover['equivalent_n_representation_R_digit_tokens']} under",
        f"representation R and n = {crossover['equivalent_n_coordinate_tokens']} with coordinate tokens.",
        "**Every dimension this project cares about, including n=128, sits far below that**",
        "**crossover**, so attention is not the bottleneck and removing it buys almost nothing.",
        "",
        "The measured attention share of encoder MACs confirms it: **6.7% at n=128 for",
        "candidate A and 3.5% for the coordinate-token models**. Removing attention",
        "entirely could not recover more than that, and would cost the only content-based",
        "mixing in the encoder.",
        "",
        "Candidate C-alt also turns out to be *worse* on the metric it was supposed to win:",
        "at n=128 it needs **2,216 MiB** of activation memory at batch 64 against 1,374 MiB",
        "for B and 1,493 MiB for A, because a diagonal SSM keeps a `d x N` state per",
        "position. The state-space option loses on parameters, on memory, and on inductive",
        "bias, and wins only a compute term that was never the bottleneck.",
        "",
        "Two further CPU-specific points argue against a state-space encoder here:",
        "",
        "1. A selective/diagonal scan is **sequential in L**. Attention at these lengths is",
        "   one dense `L x L` matmul, which BLAS executes at near-peak on every CPU this",
        "   project targets. A short scan is latency-bound and loses to a small matmul.",
        "2. The task needs **content-based** global mixing: `b = sum_i a_i s_i` requires",
        "   pairing coordinate i with secret bit i regardless of distance. A diagonal SSM",
        "   mixes by recency, not by content, which is the wrong inductive bias for a sum",
        "   over an unordered support.",
        "",
        "Candidate C-alt is included and costed precisely so this conclusion is a",
        "measured comparison rather than an assumption.",
        "",
        "## 3. Scaling to n = 128",
        "",
        "`B-lite` is not a fifth architecture: it is candidate B with `T_e=2` instead of 4.",
        "Loops reuse one parameter set, so B and B-lite have byte-for-byte identical",
        "weights and differ only in a runtime knob. The row is there to separate what the",
        "shorter sequence buys from what the extra depth spends.",
        "",
        "### Representation R (base 81, no separator)",
        "",
    ] + scaling_table("R") + [
        "",
        "### Representation P (base 81, with separator)",
        "",
    ] + scaling_table("P") + [
        "",
        "**Representation P costs 50% more positions than R for digit-token models**",
        "(`3n+1` against `2n+2`). At n=128 that is 385 positions against 258, **3.13 GMACs**",
        "**against 2.03, and 2,416 MiB of activations against 1,493 MiB** at batch 64 -- a",
        "54% compute surcharge. It buys nothing this project has measured: the phase-3.5",
        "audit established that the released code itself uses no separator. For a",
        "coordinate-token front end the separator is absorbed entirely -- there is nothing",
        "left to separate -- so P and R converge to the same length. **Recommendation: R.**",
        "",
        "## 4. Numerical representation - what is meaningful and what is not",
        "",
        "### Measured basis for the decision",
        "",
        f"- Spearman(value, digit0) = {evidence['spearman_value_vs_digit0_lsb_read_first']:+.4f}. "
        "Under `lsb_first` this is the token the encoder reads **first** for every",
        "  coordinate, and it carries little ordinal information about the value it helps",
        "  encode -- it is a sawtooth that resets three times across Z_q.",
        f"- Spearman(value, digit1) = {evidence['spearman_value_vs_digit1_msb_read_second']:+.4f}. "
        "All of the ordinal content is in the second token.",
        f"- `digit0 == 0` holds for {evidence['residues_with_digit0_equal_zero']} of {evidence['modulus']} residues and "
        f"`digit1 == 0` for {evidence['residues_with_digit1_equal_zero']}, but",
        f"  only **{evidence['residues_with_both_digits_zero']}** residue satisfies both. "
        "Detecting a zero coordinate is a",
        "  **conjunction across two sequence positions** that the encoder must spend",
        "  attention to compute, once per coordinate, on every forward pass.",
        "",
        "### Verdicts",
        "",
        "| feature | scalars | verdict | reason |",
        "|---|---:|---|---|",
    ]
    for feature in payload["numerical_features"]:
        lines.append(f"| `{feature['feature']}` | {feature['count']} | "
                     f"**{feature['verdict']}** | {feature['reason']} |")
    lines += [
        "",
        f"**Kept: {payload['kept_scalar_feature_count']} scalars per coordinate** -- the zero "
        "indicator, the signed centered",
        "residue, and the k=1 Fourier pair -- plus the existing digit embeddings and an",
        "absolute coordinate embedding. Everything else on the candidate list is either an",
        "affine function of what is already there (`x/q`) or the same quantity renamed",
        "(`magnitude`, `modular distance to zero`). Rejecting them is not conservatism; it",
        "is refusing to pay parameters for duplicated information.",
        "",
        "**Scientific caveat.** Adding numerical features changes the representation, so v2",
        "results are no longer directly comparable to the token-only setting the original",
        "SALSA uses. A token-only ablation arm must be kept and reported alongside, or the",
        "comparison to phases 6-12 is broken.",
        "",
        "## 5. Sparsity",
        "",
        "**How zeros are represented today.** Under R a zero coordinate becomes "
        f"{evidence['digits_per_integer']} copies",
        "of the same zero-digit token. At n=128 with a weight-3 secret-like sparse probe,",
        "250 of 258 encoder positions carry the identical symbol. Phase 11 measured the",
        "consequence directly: the zero-digit token fraction is 0.156 on training rows",
        "against 0.846 on probes.",
        "",
        "| n | nnz | R length | zero-digit tokens | fraction | coordinate-token length | fraction |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["sparsity_token_statistics"]:
        lines.append(
            f"| {row['n']} | {row['nnz']} | {row['R_sequence_length']} | "
            f"{row['R_zero_digit_tokens']} | {row['R_fraction_zero_digit_tokens']:.3f} | "
            f"{row['coordinate_token_length']} | {row['coordinate_token_fraction_zero']:.3f} |")
    lines += [
        "",
        "**Are zero and nonzero efficiently distinguishable?** Not today -- it is the",
        "two-position conjunction measured above. With a 1-bit input feature it is free.",
        "",
        "**Is attention diluted by zeros?** Yes, structurally. Softmax over L keys assigns",
        "mass to every position; when 97% of positions are an identical uninformative",
        "symbol, the informative keys must win by logit margin alone, and there is nothing",
        "in the architecture that lets the model cheaply say 'ignore these'. The proposed",
        "fix is one learned scalar per head added to the attention logit of any zero",
        f"coordinate ({candidates['B']['parameter_breakdown'].get('sparse_attention_bias', 0)} "
        "parameters in total), which lets the encoder learn to suppress or",
        "attend to zeros as the data demands, without hard-coding either.",
        "",
        "**Is coordinate-wise processing before global mixing useful?** Yes, and it is",
        "already the main structural change: folding the `width` digit tokens of a",
        "coordinate into one position is exactly per-coordinate processing before mixing.",
        "It halves the sequence, quarters the attention term, and makes position equal",
        "coordinate index so absolute identity becomes expressible.",
        "",
        "## 6. Hypotheses",
        "",
        "| hypothesis | verdict from this phase | basis |",
        "|---|---|---|",
        "| H1: GatedUT is sufficient, training strategy is the limit | **Not refuted, and partly supported** | Phase 11 rated insufficient training High. The n=12 run saw 100,032 samples against the paper's ~3.9M, and sample reuse was off. This phase cannot settle it, and a v2 architecture does not excuse leaving it untested. |",
        "| H2: numerical-aware encoding improves sparse generalization | **Plausible, mechanism identified, untested** | The zero indicator removes a measured two-position conjunction, and the sparse attention bias gives a mechanism for the dilution that phase 12 measured. Neither is evidence that it works. |",
        "| H3: a hybrid improves CPU efficiency without losing global interactions | **Weakly supported at best** | The attention share of MACs is small at every n <= 128, so there is little CPU efficiency to win. Candidate C costs about the same and adds two new block types. |",
        "| H4: a pure state-space model is not necessarily advantageous | **Supported** | Crossover at L = 2d = 1024, far above n=128. Plus: scans are sequential on CPU, and diagonal SSMs mix by recency where the task needs content-based mixing. |",
        "",
        "## 7. Recommendation",
        "",
        "### Recommended: candidate B, `Salsa2-NACT`",
        "",
        f"**{candidates['B']['parameter_count']:,} parameters** "
        f"({candidates['B']['fp32_size_mib']:.2f} MiB fp32), inside the 4-5M budget with no",
        "padding parameters.",
        "",
        "**1. Why should it handle sparse inputs better?** Three specific mechanisms, each",
        "tied to something measured rather than to intuition: the zero indicator turns a",
        "two-position conjunction into a 1-bit input; the per-head sparse attention bias",
        "gives the encoder an explicit, learnable way to discount zero coordinates instead",
        "of relying on logit margin; and the coordinate-token front end halves the number",
        "of positions a sparse input floods with identical symbols.",
        "",
        "**2. Why does it stay within 4-5M?** Because the encoder body is unchanged and it",
        "is 89.9% of the budget. The entire new front end costs about "
        f"{candidates['B']['parameter_count'] - candidates['A']['parameter_count']:,} "
        "parameters, spent in",
        "the cheapest region of the model. Encoder loops rise from 2 to 4, which increases",
        "effective depth at **zero** parameter cost -- that is what the shared layer is for.",
        "",
        "**3. Why is it CPU-friendly?** Halving the sequence halves everything linear in L,",
        "which is where essentially all the compute is. The `B-lite` row makes the size of",
        "that effect explicit -- **identical weights to B, loops held at T_e=2 to match A**:",
        "",
        "| n=128, representation R | A | B-lite (T_e=2) | B (T_e=4) |",
        "|---|---:|---:|---:|",
        "| encoder positions | 258 | 130 | 130 |",
        "| GMACs / sample | 2.03 | **0.99** | 1.98 |",
        "| activation MiB @ batch 64 | 1493 | **690** | 1374 |",
        "| effective encoder depth | 2 | 2 | **4** |",
        "",
        "So the shorter sequence is worth **51% of the compute and 54% of the activation**",
        "**memory at equal depth**, or -- the setting recommended here -- **twice the**",
        "**effective depth at the same compute as today**. It is one or the other, not both;",
        "T_e is a runtime knob costing no parameters, so the choice can be made per",
        "experiment. No new operator types either: embeddings, dense matmuls and softmax,",
        "all BLAS-friendly, no sequential scans, no custom kernels, no CUDA.",
        "",
        "**4. Why is it better suited to n=128?** Sequence length at n=128 falls from 258",
        "to 130, and activation memory at batch 64 from 1,493 MiB to 690 MiB at equal",
        "depth -- the difference between a CPU training run that fits comfortably and one",
        "that does not. The coordinate embedding is a fixed `n_max x d` table, so **the**",
        "**parameter count is identical at every n from 12 to 128**: one model covers the",
        "whole dimension sweep, and n=128 needs no re-architecting. Dimensions beyond 128",
        "would need the table extended, which is the one place n enters the parameter count.",
        "",
        "**5. What of the current GatedUT is retained?** The parameter-shared looped",
        "encoder, the copy gate, RMSNorm, the bias-free 2-matrix GELU FFN, RoPE, the",
        "encoder/decoder width asymmetry (512/128), the decoder in its entirety, and the",
        "existing `LatticeCodec` tokens and target format. Phase 11 found the supplied",
        "reference *disabled* looping and gating; keeping them keeps v2 closer to the",
        "published architecture than that fork.",
        "",
        "**6. What is replaced?** Only the input front end: `width` digit tokens per",
        "coordinate become one coordinate token; RoPE-only positioning gains an absolute",
        "coordinate embedding; and attention gains a per-head zero-coordinate bias. The",
        "encoder body, the decoder and the output interface are untouched.",
        "",
        "**7. What new research contribution?** A named, falsifiable claim: *that the",
        "sparse-input failure measured in phase 12 is a representation defect rather than a",
        "capacity defect, and that per-coordinate tokenisation with a zero indicator and a",
        "learned sparse attention bias moves the zero-lift crossing point toward sparser",
        "inputs at constant parameter count.* Phase 12 already established the measurement",
        "that would confirm or refute it, on the same axis, with the same baselines.",
        "",
        "### Exact specification",
        "",
        "```",
        "encoder   d = 512, heads = 8, parameter sets = 1, loops T_e = 4, gated, RMSNorm,",
        "          RoPE on self-attention, FFN multiplier 4 (hidden 2048), no biases",
        "decoder   d = 128, heads = 4, parameter sets = 1, loops T_d = 2, gated, RMSNorm,",
        "          RoPE on self-attention only, cross-attention unrotated, FFN hidden 512",
        "sharing   1 encoder set reused 4 times; 1 decoder set reused twice.",
        "          Loops cost no parameters and are runtime knobs.",
        "",
        "input     one token per coordinate, i = 0 .. n-1, built as",
        "            e_i = sum_{w<2} DigitEmb_w[digit_w(a_i)]",
        "                + W_num @ [ 1[a_i == 0],",
        "                            centered(a_i) / (q/2),",
        "                            cos(2*pi*a_i/q), sin(2*pi*a_i/q) ] + b_num",
        "                + CoordEmb[i]",
        "            zero coordinates additionally blend in a learned zero vector",
        "          plus <bos> and <eos> from a 4-entry special table -> L_in = n + 2",
        "          attention logits receive beta_head * 1[a_j == 0] on key j",
        "",
        "output    UNCHANGED. Same LatticeCodec target sequence <bos> d0 d1 <eos>,",
        "          same 85-token vocabulary, same greedy/beam decode, same",
        "          decode_generated_ids contract, so phase-10 recovery and phase-12",
        "          generalization scripts run against it without modification.",
        "```",
        "",
        f"**Expected parameter count: {candidates['B']['parameter_count']:,}** "
        f"(vs {candidates['A']['parameter_count']:,} today, "
        f"{candidates['B']['parameter_count'] - candidates['A']['parameter_count']:+,}). "
        "Exact PyTorch counting must",
        "confirm this before any training, using the existing three-way check in",
        "`salsa.models.parameter_count` (actual, component breakdown, analytical).",
        "",
        "### What this recommendation does not claim",
        "",
        "Nothing here has been trained. H2 is a mechanism, not a result. The honest",
        "sequencing is that H1 remains live and cheap to test: the current GatedUT has",
        "never been trained past 100,032 samples with sample reuse enabled, and phase 11",
        "rated insufficient training a High-confidence cause. **A v2 architecture should",
        "not be adopted on the strength of a design document while the training-budget",
        "explanation for the same failure is still untested.** The two are separable and",
        "should be separated.",
        "",
        "## Artifacts",
        "",
        "- `architecture_comparison.md` (this file)",
        "- `architecture_comparison.json`",
        "- `architecture_comparison.csv`",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
