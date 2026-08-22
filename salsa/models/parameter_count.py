"""Exact parameter accounting and hard budget enforcement.

The 4-5M parameter budget is a constraint on the research, not a slogan, so it
is checked three ways that must all agree:

1. **Actual** -- ``sum(p.numel() for p in model.parameters() if p.requires_grad)``,
   the ground truth.
2. **Component breakdown** -- every named parameter assigned to exactly one
   component, summing back to the actual total.  Nothing may land in ``other``
   unnoticed.
3. **Analytical** -- recomputed from the architecture formulas alone, without
   touching the model.

A disagreement between (1) and (3) means either the formulas or the
implementation is wrong; :func:`parameter_report` records it rather than
papering over it.

A model above ``max_parameters`` is reported as **FAILED BUDGET**.  It is never
silently accepted, and never trimmed automatically.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from torch import nn

from .transformer import ModelSpec, SalsaTransformer

__all__ = [
    "COMPONENTS",
    "ParameterReport",
    "analytical_breakdown",
    "analytical_parameter_count",
    "check_budget",
    "count_trainable_parameters",
    "format_parameter_report",
    "parameter_breakdown",
    "parameter_report",
]

#: Component names, in report order.  Every trainable parameter maps to exactly
#: one of these; ``other`` must stay empty for the shipped architectures.
COMPONENTS: Tuple[str, ...] = (
    "encoder_embedding",
    "decoder_embedding",
    "positional_encoding",
    "encoder_attention",
    "encoder_ffn",
    "encoder_copy_gate",
    "encoder_normalization",
    "decoder_self_attention",
    "decoder_cross_attention",
    "decoder_ffn",
    "decoder_copy_gate",
    "decoder_normalization",
    "output_projection",
    "other",
)

BYTES_FP32 = 4


# --------------------------------------------------------------------------- #
# Actual counts
# --------------------------------------------------------------------------- #
def count_trainable_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters.

    Args:
        model: Any module.

    Returns:
        ``sum(p.numel() for p in model.parameters() if p.requires_grad)``.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_all_parameters(model: nn.Module) -> int:
    """Return the total parameter count, trainable or frozen."""
    return sum(p.numel() for p in model.parameters())


def classify_parameter(name: str) -> str:
    """Map a parameter name to its component.

    Args:
        name: Dotted parameter name from ``model.named_parameters()``.

    Returns:
        One of :data:`COMPONENTS`.
    """
    if name.startswith("encoder_embedding"):
        return "encoder_embedding"
    if name.startswith("decoder_embedding"):
        return "decoder_embedding"
    if name.startswith("output_projection"):
        return "output_projection"

    if name.startswith("encoder_layers") or name.startswith("encoder_norm"):
        if ".self_attn." in name:
            return "encoder_attention"
        if ".ffn." in name:
            return "encoder_ffn"
        if ".gate." in name:
            return "encoder_copy_gate"
        if "norm" in name:
            return "encoder_normalization"
    if name.startswith("decoder_layers") or name.startswith("decoder_norm"):
        if ".cross_attn." in name:
            return "decoder_cross_attention"
        if ".self_attn." in name:
            return "decoder_self_attention"
        if ".ffn." in name:
            return "decoder_ffn"
        if ".gate." in name:
            return "decoder_copy_gate"
        if "norm" in name:
            return "decoder_normalization"
    return "other"


def parameter_breakdown(model: nn.Module, trainable_only: bool = True) -> Dict[str, int]:
    """Return a component-level parameter breakdown.

    Args:
        model: The model to inspect.
        trainable_only: Count only parameters with ``requires_grad``.

    Returns:
        A dictionary keyed by :data:`COMPONENTS`, whose values sum to the total.
    """
    counts: Dict[str, int] = {name: 0 for name in COMPONENTS}
    for name, parameter in model.named_parameters():
        if trainable_only and not parameter.requires_grad:
            continue
        counts[classify_parameter(name)] += parameter.numel()
    return counts


def unclassified_parameters(model: nn.Module) -> List[str]:
    """Return the names of parameters that fell into ``other``."""
    return [
        name
        for name, _ in model.named_parameters()
        if classify_parameter(name) == "other"
    ]


# --------------------------------------------------------------------------- #
# Analytical counts
# --------------------------------------------------------------------------- #
def analytical_breakdown(spec: ModelSpec) -> Dict[str, int]:
    """Compute the parameter breakdown from the architecture formulas alone.

    This function never touches a built model, so comparing it against
    :func:`parameter_breakdown` is a genuine cross-check.

    Formulas (V = vocab, d_e / d_d = widths, L = parameter sets, m = FFN
    multiplier)::

        encoder_embedding        V * d_e
        decoder_embedding        V * d_d
        positional_encoding      0                                  (RoPE)
        encoder_attention        L_e * 4 * d_e^2
        encoder_ffn              L_e * 2 * d_e * (m * d_e)
        encoder_copy_gate        L_e * (2 * d_e^2 + d_e)            if gated
        encoder_normalization    L_e * 2 * d_e + d_e
        decoder_self_attention   L_d * 4 * d_d^2
        decoder_cross_attention  L_d * (2 * d_d^2 + 2 * d_e * d_d)
        decoder_ffn              L_d * 2 * d_d * (m * d_d)
        decoder_copy_gate        L_d * (2 * d_d^2 + d_d)            if gated
        decoder_normalization    L_d * 3 * d_d + d_d
        output_projection        d_d * V                            if untied

    Note that no term depends on ``encoder_loops`` or ``decoder_loops``: passes
    reuse parameter sets, so effective depth is free of parameter cost.

    Args:
        spec: The model specification.

    Returns:
        A dictionary keyed by :data:`COMPONENTS`.
    """
    v = spec.vocab_size
    de, dd = spec.encoder_dim, spec.decoder_dim
    le, ld = spec.encoder_layers, spec.decoder_layers
    fe, fd = spec.encoder_ffn_dim, spec.decoder_ffn_dim

    counts: Dict[str, int] = {name: 0 for name in COMPONENTS}
    counts["encoder_embedding"] = v * de
    counts["decoder_embedding"] = v * dd
    counts["positional_encoding"] = 0
    counts["encoder_attention"] = le * 4 * de * de
    counts["encoder_ffn"] = le * 2 * de * fe
    counts["encoder_copy_gate"] = le * (2 * de * de + de) if spec.gated else 0
    counts["encoder_normalization"] = le * 2 * de + de
    counts["decoder_self_attention"] = ld * 4 * dd * dd
    counts["decoder_cross_attention"] = ld * (2 * dd * dd + 2 * de * dd)
    counts["decoder_ffn"] = ld * 2 * dd * fd
    counts["decoder_copy_gate"] = ld * (2 * dd * dd + dd) if spec.gated else 0
    counts["decoder_normalization"] = ld * 3 * dd + dd
    counts["output_projection"] = 0 if spec.tie_embeddings else dd * v
    return counts


def analytical_parameter_count(spec: ModelSpec) -> int:
    """Return the analytically predicted trainable parameter count."""
    return sum(analytical_breakdown(spec).values())


# --------------------------------------------------------------------------- #
# Budget
# --------------------------------------------------------------------------- #
def check_budget(
    count: int,
    max_parameters: Optional[int] = None,
    min_parameters: Optional[int] = None,
) -> str:
    """Classify a parameter count against the configured budget.

    Args:
        count: Measured trainable parameter count.
        max_parameters: Hard ceiling.  Exceeding it is a failed configuration.
        min_parameters: Optional floor, used to flag an under-sized model.

    Returns:
        ``"FAILED BUDGET"``, ``"UNDER TARGET"`` or ``"OK"``.
    """
    if max_parameters is not None and count > int(max_parameters):
        return "FAILED BUDGET"
    if min_parameters is not None and count < int(min_parameters):
        return "UNDER TARGET"
    return "OK"


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
@dataclass
class ParameterReport:
    """A complete, serialisable parameter accounting for one model.

    Attributes:
        name: Model name.
        spec: The specification the model was built from.
        trainable: Measured trainable parameter count.
        total: Measured total parameter count (trainable or not).
        breakdown: Measured per-component counts.
        analytical: Predicted count from the formulas.
        analytical_breakdown: Predicted per-component counts.
        budget_status: Result of :func:`check_budget`.
        max_parameters: The ceiling used, if any.
        min_parameters: The floor used, if any.
        unclassified: Parameter names that fell into ``other``.
    """

    name: str
    spec: ModelSpec
    trainable: int
    total: int
    breakdown: Dict[str, int]
    analytical: int
    analytical_breakdown: Dict[str, int]
    budget_status: str
    max_parameters: Optional[int] = None
    min_parameters: Optional[int] = None
    unclassified: List[str] = field(default_factory=list)

    @property
    def matches_analytical(self) -> bool:
        """True when the measured and predicted totals agree exactly."""
        return self.trainable == self.analytical

    @property
    def component_mismatches(self) -> Dict[str, Tuple[int, int]]:
        """Components where measurement and prediction disagree."""
        return {
            key: (self.breakdown[key], self.analytical_breakdown[key])
            for key in COMPONENTS
            if self.breakdown[key] != self.analytical_breakdown[key]
        }

    @property
    def fp32_bytes(self) -> int:
        """Model size in bytes at FP32."""
        return self.trainable * BYTES_FP32

    @property
    def fp32_megabytes(self) -> float:
        """Model size in megabytes at FP32."""
        return self.fp32_bytes / (1024.0 ** 2)

    @property
    def within_budget(self) -> bool:
        """True when the model is at or below the hard ceiling."""
        return self.budget_status != "FAILED BUDGET"

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable view for run metadata."""
        return {
            "name": self.name,
            "spec": self.spec.to_dict(),
            "trainable_parameters": self.trainable,
            "total_parameters": self.total,
            "analytical_parameters": self.analytical,
            "matches_analytical": self.matches_analytical,
            "breakdown": dict(self.breakdown),
            "analytical_breakdown": dict(self.analytical_breakdown),
            "budget_status": self.budget_status,
            "max_parameters": self.max_parameters,
            "min_parameters": self.min_parameters,
            "fp32_megabytes": round(self.fp32_megabytes, 4),
            "unclassified": list(self.unclassified),
        }


def parameter_report(
    model: SalsaTransformer,
    max_parameters: Optional[int] = None,
    min_parameters: Optional[int] = None,
) -> ParameterReport:
    """Measure a model and cross-check it against the analytical formulas.

    Args:
        model: The model to account for.
        max_parameters: Hard ceiling for the budget check.
        min_parameters: Optional floor.

    Returns:
        A :class:`ParameterReport`.
    """
    trainable = count_trainable_parameters(model)
    return ParameterReport(
        name=model.name,
        spec=model.spec,
        trainable=trainable,
        total=count_all_parameters(model),
        breakdown=parameter_breakdown(model),
        analytical=analytical_parameter_count(model.spec),
        analytical_breakdown=analytical_breakdown(model.spec),
        budget_status=check_budget(trainable, max_parameters, min_parameters),
        max_parameters=max_parameters,
        min_parameters=min_parameters,
        unclassified=unclassified_parameters(model),
    )


def report_from_config(config, model: Optional[SalsaTransformer] = None) -> ParameterReport:
    """Build (if needed) and account for the model described by a config.

    Args:
        config: A :class:`~salsa.utils.config.Config`.
        model: An already-built model; one is constructed when omitted.

    Returns:
        A :class:`ParameterReport` using the config's budget bounds.
    """
    from .transformer import build_model

    if model is None:
        model = build_model(config)
    return parameter_report(
        model,
        max_parameters=config.model.max_parameters,
        min_parameters=config.model.min_parameters,
    )


def format_parameter_report(report: ParameterReport, breakdown: bool = True) -> str:
    """Render a human-readable parameter report.

    Args:
        report: The report to render.
        breakdown: Include the per-component table.

    Returns:
        The report as a multi-line string.
    """
    spec = report.spec
    lines = [
        f"Model: {report.name}",
        f"Architecture: {spec.arch}",
        f"Vocabulary: {spec.vocab_size}",
        f"Encoder dimension: {spec.encoder_dim}",
        f"Decoder dimension: {spec.decoder_dim}",
        f"Encoder heads: {spec.encoder_heads}",
        f"Decoder heads: {spec.decoder_heads}",
        f"Encoder shared layers: {spec.encoder_layers}",
        f"Decoder shared layers: {spec.decoder_layers}",
        f"Encoder loops: {spec.encoder_loops}",
        f"Decoder loops: {spec.decoder_loops}",
        f"Copy gate: {spec.gated}",
        f"Positional encoding: {'RoPE (0 parameters)' if spec.use_rope else 'none'}",
        f"Trainable parameters: {report.trainable:,}",
        f"FP32 size: {report.fp32_megabytes:.2f} MB",
    ]
    if breakdown:
        lines.append("")
        lines.append(f"{'component':26} {'measured':>12} {'analytical':>12} {'match':>7}")
        lines.append("-" * 60)
        for key in COMPONENTS:
            measured = report.breakdown[key]
            predicted = report.analytical_breakdown[key]
            if measured == 0 and predicted == 0:
                continue
            flag = "ok" if measured == predicted else "MISMATCH"
            lines.append(f"{key:26} {measured:>12,} {predicted:>12,} {flag:>7}")
        lines.append("-" * 60)
        flag = "ok" if report.matches_analytical else "MISMATCH"
        lines.append(
            f"{'TOTAL':26} {report.trainable:>12,} {report.analytical:>12,} {flag:>7}"
        )
    lines.append("")
    ceiling = f"{report.max_parameters:,}" if report.max_parameters else "n/a"
    floor = f"{report.min_parameters:,}" if report.min_parameters else "n/a"
    lines.append(f"Budget: min {floor}, max {ceiling}  ->  {report.budget_status}")
    if report.unclassified:
        lines.append(f"UNCLASSIFIED PARAMETERS: {report.unclassified}")
    return "\n".join(lines)
