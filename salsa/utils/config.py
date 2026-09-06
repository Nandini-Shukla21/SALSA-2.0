"""Typed, strict, file-driven configuration system for Lightweight SALSA.

Design rules (derived from the research protocol):

* **Nothing is silent.**  Unknown keys raise :class:`ConfigError` instead of
  being ignored, so a typo can never quietly change ``q``, ``sigma`` or the
  secret distribution.
* **Everything is recorded.**  A config can be serialised back to YAML and
  fingerprinted (SHA-256) so any reported number can be traced to the exact
  settings that produced it.
* **Experiments are comparable.**  ``extends:`` inheritance lets every model
  configuration in a sweep share one byte-identical cryptographic block.
* **Assumptions live in the file, not the code.**  Defaults here mirror the
  SALSA paper (q = 251, sigma = 3, base-81 encoding) but every experiment must
  still state them explicitly in its YAML.

Typical use::

    from salsa.utils import load_config
    cfg = load_config("configs/target_4_5m.yaml", overrides=["training.batch_size=64"])
    warnings = cfg.validate()
"""

import copy
import dataclasses
import hashlib
import json
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Type, TypeVar

try:  # pragma: no cover - trivial import guard
    import yaml
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "PyYAML is required for the Lightweight SALSA configuration system. "
        "Install it with `python -m pip install pyyaml`."
    ) from exc


T = TypeVar("T")

#: Keys handled by the loader itself rather than by a dataclass section.
_META_KEYS = ("extends",)


class ConfigError(ValueError):
    """Raised for any malformed, unknown or invalid configuration entry."""


# --------------------------------------------------------------------------- #
# Configuration sections
# --------------------------------------------------------------------------- #
@dataclass
class ExperimentConfig:
    """Identity, provenance and reproducibility settings for one experiment.

    Attributes:
        name: Short experiment name; also the run directory name.
        seed: Master random seed.  All other seeds are derived from it.
        deterministic: Request deterministic kernels/algorithms where available.
        output_root: Root directory for run artefacts (relative paths allowed).
        notes: Free-form research notes recorded with the run metadata.
        tags: Short labels used to group runs when comparing experiments.
    """

    name: str = "unnamed"
    seed: int = 0
    deterministic: bool = True
    output_root: str = "results"
    notes: str = ""
    tags: List[str] = field(default_factory=list)


@dataclass
class LWEConfig:
    """The cryptographic problem instance.

    These values define the *science*, not the engineering.  They must be
    identical across every model in a controlled comparison.

    Attributes:
        structure: ``"lwe"`` for i.i.d. random rows, ``"rlwe"`` for rows drawn
            from a circulant matrix (the setting used in the SALSA paper).
        rlwe_variant: Ring structure used when ``structure="rlwe"``:
            ``"circulant"`` (the paper's wording) or ``"negacyclic"``
            (``Z_q[x]/(x^n+1)``, what deployed ring-LWE schemes use).
        n: Lattice dimension / secret length.
        q: Modulus (prime in the SALSA setting).
        sigma: Standard deviation of the discrete Gaussian error distribution.
        error_distribution: Name of the error distribution.
        secret_distribution: ``"binary"`` ({0,1}) or ``"ternary"`` ({-1,0,1}).
        hamming_weight: Exact number of non-zero secret coordinates, or None.
        density: Fraction of non-zero coordinates, or None.  Exactly one of
            ``hamming_weight`` / ``density`` must be given.
        max_a_fraction: Upper bound on the entries of ``a`` expressed as a
            fraction of ``q`` (1.0 = full range).  Reproduces the bounded-``a``
            ablation of the paper.
        num_train_samples: Number of distinct LWE instances generated for
            training (before any reuse).
        num_valid_samples: Number of held-out instances for validation.
        num_test_samples: Number of held-out instances for final testing.
        sample_reuse: How many times each training instance may be reused.
        num_secrets: Number of independent secrets to generate.
        secret_index: Index of the secret under attack.
    """

    structure: str = "rlwe"
    rlwe_variant: str = "circulant"
    n: int = 30
    q: int = 251
    sigma: float = 3.0
    error_distribution: str = "discrete_gaussian"
    secret_distribution: str = "binary"
    hamming_weight: Optional[int] = 3
    density: Optional[float] = None
    max_a_fraction: float = 1.0
    num_train_samples: int = 200_000
    num_valid_samples: int = 5_000
    num_test_samples: int = 5_000
    sample_reuse: int = 1
    num_secrets: int = 1
    secret_index: int = 0

    @property
    def resolved_hamming_weight(self) -> int:
        """Return the Hamming weight implied by this configuration."""
        if self.hamming_weight is not None:
            return int(self.hamming_weight)
        if self.density is not None:
            return int(round(self.density * self.n))
        raise ConfigError("lwe: neither 'hamming_weight' nor 'density' is set.")

    @property
    def resolved_density(self) -> float:
        """Return the secret density implied by this configuration."""
        return self.resolved_hamming_weight / float(self.n)


@dataclass
class EncodingConfig:
    """How integers in Z_q are turned into model tokens.

    Attributes:
        base: Positional base used to write integers as digit sequences.
        input_base: Optional override for the encoder side (None = ``base``).
        output_base: Optional override for the decoder side (None = ``base``).
        separator: Insert a separator token between coordinates of ``a``.
        include_bos_eos: Wrap sequences in begin/end-of-sequence tokens.
        balanced: Use a balanced (signed) digit representation.
        digit_order: "lsb_first" (the order the released SALSA code writes) or
            "msb_first".  Established by the phase-3.5 encoding audit.
        fixed_width: Pad every value to a fixed digit count.  Required for
            separator-free parsing and for the batch encoding path.
    """

    base: int = 81
    input_base: Optional[int] = None
    output_base: Optional[int] = None
    separator: bool = True
    include_bos_eos: bool = True
    balanced: bool = False
    digit_order: str = "msb_first"
    fixed_width: bool = True

    @property
    def resolved_input_base(self) -> int:
        """Base actually used to encode the model input."""
        return int(self.input_base if self.input_base is not None else self.base)

    @property
    def resolved_output_base(self) -> int:
        """Base actually used to encode the model output."""
        return int(self.output_base if self.output_base is not None else self.base)


@dataclass
class ModelConfig:
    """Capacity knobs and the hard parameter budget.

    The architecture itself is *not* fixed in phase 1.  These fields describe
    capacity in architecture-agnostic terms so that one sweep file can drive
    several compact designs later.

    Attributes:
        arch: Registered architecture name -- ``"compact_transformer"`` (plain,
            one parameter set per layer) or ``"gated_universal_transformer"``
            (shared layers reused across loops, with a learned copy gate).
        encoder_dim: Encoder model width.
        decoder_dim: Decoder model width.
        encoder_layers: Number of encoder blocks.
        decoder_layers: Number of decoder blocks.
        encoder_heads: Encoder attention heads.
        decoder_heads: Decoder attention heads.
        encoder_loops: Number of encoder passes at runtime.  ``None`` means one
            pass per layer (a plain transformer).  With shared layers this is
            the effective-depth knob: raising it changes compute but **not** the
            parameter count.
        decoder_loops: Same, for the decoder.
        gated: Wrap each layer in a learned copy gate (Universal Transformer).
        ffn_multiplier: Feed-forward width as a multiple of the model width.
        dropout: Dropout probability.
        attention_dropout: Dropout probability inside attention.
        tie_embeddings: Share decoder input/output embedding matrices.
        max_parameters: Hard ceiling.  A model above this must be reported as a
            FAILED BUDGET configuration, never silently accepted.
        target_parameters: Informational nominal size for this configuration.
        min_parameters: Optional lower bound used to flag under-sized models.
    """

    arch: str = "compact_transformer"
    #: NACT ablation variant; ignored unless arch == "nact".
    nact_variant: str = "full"
    encoder_dim: int = 256
    decoder_dim: int = 128
    encoder_layers: int = 2
    decoder_layers: int = 2
    encoder_heads: int = 4
    decoder_heads: int = 4
    encoder_loops: Optional[int] = None
    decoder_loops: Optional[int] = None
    gated: bool = False
    ffn_multiplier: float = 4.0
    dropout: float = 0.0
    attention_dropout: float = 0.0
    tie_embeddings: bool = False
    max_parameters: int = 5_000_000
    target_parameters: Optional[int] = None
    min_parameters: Optional[int] = None


@dataclass
class TrainingConfig:
    """Optimisation schedule and run control.

    Attributes:
        batch_size: Training batch size.
        eval_batch_size: Batch size used during validation.
        max_epochs: Maximum number of epochs.
        steps_per_epoch: Optimiser steps that constitute one epoch.
        optimizer: Optimiser name.
        lr: Peak learning rate.
        weight_decay: Decoupled weight decay.
        beta1: Adam beta1.
        beta2: Adam beta2.
        eps: Adam epsilon.
        warmup_steps: Linear warm-up steps before the decay schedule.
        min_lr: Floor of the decay schedule.
        lr_schedule: ``"cosine"``, ``"inverse_sqrt"`` or ``"constant"``.
        grad_clip: Gradient-norm clipping value (0 disables).
        label_smoothing: Cross-entropy label smoothing.
        num_workers: DataLoader worker processes (0 is safest on Windows).
        log_every: Log training statistics every N optimiser steps.
        eval_every: Run validation every N epochs.
        checkpoint_every: Save a periodic checkpoint every N epochs (0 = off).
        keep_last_checkpoints: How many periodic checkpoints to retain.
        resume: Resume from the run's ``last.pt`` checkpoint when present.
        early_stopping_patience: Stop after N evaluations without improvement.
        monitor_metric: Validation metric that drives best-checkpoint selection
            and early stopping.
        monitor_mode: ``"min"`` or ``"max"`` -- whether lower or higher is better
            for ``monitor_metric``.
        stop_on_secret_recovery: Terminate as soon as a secret is verified.
    """

    batch_size: int = 128
    eval_batch_size: int = 128
    max_epochs: int = 100
    steps_per_epoch: int = 1_000
    optimizer: str = "adamw"
    lr: float = 3e-4
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.98
    eps: float = 1e-9
    warmup_steps: int = 1_000
    min_lr: float = 1e-6
    lr_schedule: str = "cosine"
    grad_clip: float = 1.0
    label_smoothing: float = 0.0
    num_workers: int = 0
    log_every: int = 100
    eval_every: int = 1
    checkpoint_every: int = 1
    keep_last_checkpoints: int = 2
    resume: bool = True
    early_stopping_patience: int = 20
    monitor_metric: str = "valid_loss"
    monitor_mode: str = "min"
    stop_on_secret_recovery: bool = True


@dataclass
class EvaluationConfig:
    """Validation metrics and decoding settings.

    Attributes:
        tolerance: Accuracy tolerance tau as a fraction of q (SALSA uses 0.1).
        beam_size: Beam width (1 = greedy decoding, the SALSA default).
        length_penalty: Beam-search length penalty.
        early_stopping: Stop the beam once enough hypotheses are complete.
        max_eval_batches: Cap on validation batches (0 = evaluate all).
        report_bitwise: Report per-digit (bitwise) accuracy.
        report_percentiles: Report the |b - b_hat| / q percentile table.
    """

    tolerance: float = 0.1
    beam_size: int = 1
    length_penalty: float = 1.0
    early_stopping: bool = True
    max_eval_batches: int = 0
    report_bitwise: bool = True
    report_percentiles: bool = True


@dataclass
class RecoveryConfig:
    """Secret-recovery algorithms, evaluated independently of one another.

    Attributes:
        enable_direct: Run direct (chosen-``a``) recovery.
        enable_distinguisher: Run decision-LWE distinguisher recovery.
        direct_k_values: Multipliers K used to build the special inputs K*e_i.
        direct_k_random_count: Extra randomly drawn K values per attempt.
        binarization_methods: Thresholding rules turned into secret candidates.
        distinguisher_min_accuracy: Minimum acc_tau (in %) before running the
            distinguisher is worthwhile.
        distinguisher_max_samples: Cap on samples per coordinate.
        distinguisher_tolerance: Tolerance tau used by the distinguisher.
        run_every: Attempt recovery every N evaluations.
    """

    enable_direct: bool = True
    enable_distinguisher: bool = True
    direct_k_values: List[int] = field(default_factory=lambda: [1, 2, 3])
    direct_k_random_count: int = 0
    binarization_methods: List[str] = field(
        default_factory=lambda: ["mean", "median", "mode"]
    )
    #: How a single candidate is chosen from the K sweep. "aggregate" is the
    #: separation-weighted vote and is the safe default; "best_margin" is the
    #: single-K rule kept as a diagnostic (phase 26 showed it is degenerate at
    #: low separation, so it is guarded by min_separation).
    selection_rule: str = "aggregate"
    #: Smallest ring separation a K may have and still be eligible for
    #: best_margin selection. None derives it from lwe.sigma as floor(sigma)+1.
    min_separation: Optional[int] = None
    distinguisher_min_accuracy: float = 25.0
    distinguisher_max_samples: int = 400
    distinguisher_tolerance: float = 0.1
    run_every: int = 1


@dataclass
class VerificationConfig:
    """Mathematical residual verification of secret candidates.

    A candidate ``s_hat`` is accepted only when the residuals
    ``r = b - a . s_hat mod q`` behave like the error distribution rather than
    like uniform noise over Z_q.

    Attributes:
        num_samples: Number of held-out LWE samples used for verification.
        std_tolerance_factor: Accept when std(r) <= factor * sigma.
        use_centered_residuals: Map residuals into (-q/2, q/2] before computing
            statistics.
        also_check_complement: Also test the bit-flipped candidate.
    """

    num_samples: int = 1_000
    #: Whether the pipeline runs verification at all.
    enabled: bool = True
    #: Split label the fresh verification stream is derived from. Must differ
    #: from every training/validation split so the samples are genuinely fresh.
    split_label: str = "pipeline_verify"
    std_tolerance_factor: float = 2.0
    use_centered_residuals: bool = True
    also_check_complement: bool = True


@dataclass
class DeviceConfig:
    """Compute placement.  CPU is the default and the supported baseline.

    Attributes:
        prefer: ``"cpu"`` (default), ``"auto"``, ``"cuda"`` or ``"mps"``.
        allow_gpu_fallback: If a requested accelerator is unavailable, fall back
            to CPU with a warning instead of raising.
        threads: Torch intra-op thread count (None = library default).
        interop_threads: Torch inter-op thread count (None = library default).
        dtype: Compute dtype, ``"float32"`` by default for CPU stability.
        pin_memory: DataLoader pinned memory (ignored on CPU).
    """

    prefer: str = "cpu"
    allow_gpu_fallback: bool = True
    threads: Optional[int] = None
    interop_threads: Optional[int] = None
    dtype: str = "float32"
    pin_memory: bool = False


@dataclass
class PipelineConfig:
    """Where the end-to-end pipeline finds its inputs and puts its outputs.

    Nothing here changes the science; it is plumbing. The cryptographic
    instance lives in ``lwe``, the encoding in ``encoding``, the K sweep in
    ``recovery`` and the verification budget in ``verification``.

    Attributes:
        run_dir: Directory of a completed training run. The best checkpoint is
            located from that run's own metadata; the filename is never assumed.
        checkpoint: Explicit checkpoint path, overriding ``run_dir``. Normally
            left unset.
        expect_parameters: Parameter count the checkpoint must report. An
            identity gate, not a tuning knob; None skips the check.
        expect_arch: Architecture the checkpoint must report. None skips it.
        output_dir: Where the pipeline writes its result artifacts.
        label: Short name for this pipeline run, used in reports.
    """

    run_dir: Optional[str] = None
    checkpoint: Optional[str] = None
    expect_parameters: Optional[int] = None
    expect_arch: Optional[str] = None
    output_dir: str = "results/final_pipeline"
    label: str = "salsa2"


@dataclass
class Config:
    """Root configuration object for one Lightweight SALSA experiment."""

    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    lwe: LWEConfig = field(default_factory=LWEConfig)
    encoding: EncodingConfig = field(default_factory=EncodingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    recovery: RecoveryConfig = field(default_factory=RecoveryConfig)
    verification: VerificationConfig = field(default_factory=VerificationConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    device: DeviceConfig = field(default_factory=DeviceConfig)

    # -- serialisation ----------------------------------------------------- #
    def to_dict(self) -> Dict[str, Any]:
        """Return a plain, JSON/YAML-serialisable dictionary."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Config":
        """Build a :class:`Config` from a nested mapping, strictly."""
        return _build_dataclass(cls, data, path="")

    # -- derived helpers --------------------------------------------------- #
    @property
    def resolved_encoder_loops(self) -> int:
        """Encoder passes at runtime (defaults to one pass per layer)."""
        loops = self.model.encoder_loops
        return int(self.model.encoder_layers if loops is None else loops)

    @property
    def resolved_decoder_loops(self) -> int:
        """Decoder passes at runtime (defaults to one pass per layer)."""
        loops = self.model.decoder_loops
        return int(self.model.decoder_layers if loops is None else loops)

    @classmethod
    def from_yaml(cls, path: "Path | str") -> "Config":
        """Load a configuration from a single YAML file (no ``extends``)."""
        raw = _read_yaml(Path(path))
        raw.pop("extends", None)
        return cls.from_dict(raw)

    def fingerprint(self) -> str:
        """Return a stable short SHA-256 fingerprint of this configuration."""
        return config_fingerprint(self)

    # -- validation -------------------------------------------------------- #
    def validate(self) -> List[str]:
        """Validate the configuration.

        Returns:
            A list of non-fatal warning strings (e.g. ``"q is not prime"``).

        Raises:
            ConfigError: If any setting is invalid or internally inconsistent.
        """
        warnings: List[str] = []
        exp, lwe, enc = self.experiment, self.lwe, self.encoding
        mdl, trn, evl = self.model, self.training, self.evaluation
        rec, ver, dev = self.recovery, self.verification, self.device

        # -- experiment
        if exp.seed < 0:
            raise ConfigError("experiment.seed must be >= 0.")
        if not exp.name.strip():
            raise ConfigError("experiment.name must be a non-empty string.")

        # -- lwe
        if lwe.structure not in ("lwe", "rlwe"):
            raise ConfigError("lwe.structure must be 'lwe' or 'rlwe'.")
        if lwe.rlwe_variant not in ("circulant", "negacyclic"):
            raise ConfigError(
                "lwe.rlwe_variant must be 'circulant' or 'negacyclic'."
            )
        if lwe.n < 2:
            raise ConfigError("lwe.n must be >= 2.")
        if lwe.q < 3:
            raise ConfigError("lwe.q must be >= 3.")
        if not _is_prime(lwe.q):
            warnings.append(
                f"lwe.q={lwe.q} is not prime; the SALSA reference setting uses a "
                "prime modulus."
            )
        if lwe.sigma < 0:
            raise ConfigError("lwe.sigma must be >= 0.")
        if lwe.error_distribution not in (
            "discrete_gaussian",
            "rounded_gaussian",
            "none",
        ):
            raise ConfigError(
                "lwe.error_distribution must be 'discrete_gaussian', "
                "'rounded_gaussian' or 'none'."
            )
        if lwe.secret_distribution not in ("binary", "ternary"):
            raise ConfigError("lwe.secret_distribution must be 'binary' or 'ternary'.")
        if (lwe.hamming_weight is None) == (lwe.density is None):
            raise ConfigError(
                "lwe: set exactly one of 'hamming_weight' or 'density' "
                "(research assumptions must be explicit)."
            )
        weight = lwe.resolved_hamming_weight
        if not 1 <= weight <= lwe.n:
            raise ConfigError(
                f"lwe: resolved Hamming weight {weight} must satisfy 1 <= h <= n "
                f"({lwe.n})."
            )
        if weight < 2:
            warnings.append(
                "lwe: Hamming weight < 2 is a degenerate secret; the paper uses h >= 2."
            )
        if not 0.0 < lwe.max_a_fraction <= 1.0:
            raise ConfigError("lwe.max_a_fraction must be in (0, 1].")
        for name in ("num_train_samples", "num_valid_samples", "num_test_samples"):
            if getattr(lwe, name) <= 0:
                raise ConfigError(f"lwe.{name} must be > 0.")
        if lwe.sample_reuse < 1:
            raise ConfigError("lwe.sample_reuse must be >= 1.")
        if lwe.num_secrets < 1:
            raise ConfigError("lwe.num_secrets must be >= 1.")
        if not 0 <= lwe.secret_index < lwe.num_secrets:
            raise ConfigError("lwe.secret_index must be in [0, lwe.num_secrets).")

        # -- encoding
        for base_name, base_value in (
            ("base", enc.base),
            ("input_base", enc.resolved_input_base),
            ("output_base", enc.resolved_output_base),
        ):
            if base_value < 2:
                raise ConfigError(f"encoding.{base_name} must be >= 2.")
        if enc.digit_order not in ("msb_first", "lsb_first"):
            raise ConfigError(
                "encoding.digit_order must be 'msb_first' or 'lsb_first'."
            )
        if not enc.fixed_width and not enc.separator:
            raise ConfigError(
                "encoding: fixed_width=false requires separator=true, otherwise "
                "coordinate boundaries are ambiguous and decoding is not unique."
            )

        # -- model
        if mdl.max_parameters <= 0:
            raise ConfigError("model.max_parameters must be > 0.")
        if mdl.target_parameters is not None and mdl.target_parameters > mdl.max_parameters:
            raise ConfigError(
                "model.target_parameters must not exceed model.max_parameters."
            )
        if mdl.min_parameters is not None and mdl.min_parameters > mdl.max_parameters:
            raise ConfigError(
                "model.min_parameters must not exceed model.max_parameters."
            )
        for dim_name in ("encoder_dim", "decoder_dim"):
            if getattr(mdl, dim_name) <= 0:
                raise ConfigError(f"model.{dim_name} must be > 0.")
        for layer_name in ("encoder_layers", "decoder_layers"):
            if getattr(mdl, layer_name) <= 0:
                raise ConfigError(f"model.{layer_name} must be > 0.")
        for head_name, dim_name in (
            ("encoder_heads", "encoder_dim"),
            ("decoder_heads", "decoder_dim"),
        ):
            heads, dim = getattr(mdl, head_name), getattr(mdl, dim_name)
            if heads <= 0:
                raise ConfigError(f"model.{head_name} must be > 0.")
            if dim % heads != 0:
                raise ConfigError(
                    f"model.{dim_name} ({dim}) must be divisible by "
                    f"model.{head_name} ({heads})."
                )
        if mdl.arch not in ("compact_transformer", "gated_universal_transformer", "nact"):
            raise ConfigError(
                "model.arch must be 'compact_transformer', "
                "'gated_universal_transformer' or 'nact'."
            )
        if self.recovery.selection_rule not in ("aggregate", "best_margin"):
            raise ConfigError(
                "recovery.selection_rule must be 'aggregate' or 'best_margin', "
                f"got {self.recovery.selection_rule!r}."
            )
        if (self.recovery.min_separation is not None
                and self.recovery.min_separation < 1):
            raise ConfigError(
                "recovery.min_separation must be >= 1 when set, got "
                f"{self.recovery.min_separation}."
            )
        if self.verification.num_samples < 1:
            raise ConfigError(
                "verification.num_samples must be >= 1, got "
                f"{self.verification.num_samples}."
            )
        if mdl.nact_variant not in ("full", "one_token_only"):
            raise ConfigError(
                "model.nact_variant must be 'full' or 'one_token_only', "
                f"got {mdl.nact_variant!r}."
            )
        for loop_name, layer_name in (
            ("encoder_loops", "encoder_layers"),
            ("decoder_loops", "decoder_layers"),
        ):
            loops = getattr(mdl, loop_name)
            if loops is None:
                continue
            if loops < 1:
                raise ConfigError(f"model.{loop_name} must be >= 1 when set.")
            if loops < getattr(mdl, layer_name):
                raise ConfigError(
                    f"model.{loop_name}={loops} is below model.{layer_name}="
                    f"{getattr(mdl, layer_name)}: some layers would never execute "
                    "and their parameters would be dead weight."
                )
        for head_name, dim_name in (
            ("encoder_heads", "encoder_dim"),
            ("decoder_heads", "decoder_dim"),
        ):
            head_dim = getattr(mdl, dim_name) // getattr(mdl, head_name)
            if head_dim % 2 != 0:
                raise ConfigError(
                    f"model.{dim_name} / model.{head_name} = {head_dim} must be even "
                    "for rotary positional embeddings."
                )
        if mdl.ffn_multiplier <= 0:
            raise ConfigError("model.ffn_multiplier must be > 0.")
        for prob_name in ("dropout", "attention_dropout"):
            if not 0.0 <= getattr(mdl, prob_name) < 1.0:
                raise ConfigError(f"model.{prob_name} must be in [0, 1).")

        # -- training
        if trn.batch_size <= 0 or trn.eval_batch_size <= 0:
            raise ConfigError("training batch sizes must be > 0.")
        if trn.max_epochs <= 0 or trn.steps_per_epoch <= 0:
            raise ConfigError("training.max_epochs and steps_per_epoch must be > 0.")
        if trn.lr <= 0:
            raise ConfigError("training.lr must be > 0.")
        if trn.min_lr < 0 or trn.min_lr > trn.lr:
            raise ConfigError("training.min_lr must satisfy 0 <= min_lr <= lr.")
        if trn.warmup_steps < 0:
            raise ConfigError("training.warmup_steps must be >= 0.")
        if trn.lr_schedule not in ("cosine", "inverse_sqrt", "constant"):
            raise ConfigError(
                "training.lr_schedule must be 'cosine', 'inverse_sqrt' or 'constant'."
            )
        if trn.optimizer not in ("adamw", "adam", "sgd"):
            raise ConfigError("training.optimizer must be 'adamw', 'adam' or 'sgd'.")
        if trn.num_workers < 0:
            raise ConfigError("training.num_workers must be >= 0.")
        if trn.grad_clip < 0:
            raise ConfigError("training.grad_clip must be >= 0 (0 disables clipping).")
        if not 0.0 <= trn.label_smoothing < 1.0:
            raise ConfigError("training.label_smoothing must be in [0, 1).")
        if trn.eval_every < 1:
            raise ConfigError("training.eval_every must be >= 1.")
        if trn.monitor_mode not in ("min", "max"):
            raise ConfigError("training.monitor_mode must be 'min' or 'max'.")
        if not trn.monitor_metric.strip():
            raise ConfigError("training.monitor_metric must be a non-empty name.")

        # -- evaluation
        if not 0.0 < evl.tolerance <= 0.5:
            raise ConfigError("evaluation.tolerance must be in (0, 0.5].")
        if evl.beam_size < 1:
            raise ConfigError("evaluation.beam_size must be >= 1.")
        if evl.max_eval_batches < 0:
            raise ConfigError("evaluation.max_eval_batches must be >= 0.")

        # -- recovery
        if not rec.enable_direct and not rec.enable_distinguisher:
            warnings.append(
                "recovery: both algorithms are disabled; no secret recovery will be "
                "attempted."
            )
        if rec.enable_direct and not rec.direct_k_values and rec.direct_k_random_count <= 0:
            raise ConfigError(
                "recovery: direct recovery is enabled but no K values are configured."
            )
        for method in rec.binarization_methods:
            if method not in ("mean", "median", "mode", "softmax_mean"):
                raise ConfigError(f"recovery: unknown binarization method '{method}'.")
        if not 0.0 < rec.distinguisher_tolerance <= 0.5:
            raise ConfigError("recovery.distinguisher_tolerance must be in (0, 0.5].")
        if rec.distinguisher_max_samples <= 0:
            raise ConfigError("recovery.distinguisher_max_samples must be > 0.")
        if rec.run_every < 1:
            raise ConfigError("recovery.run_every must be >= 1.")

        # -- verification
        if ver.num_samples <= 0:
            raise ConfigError("verification.num_samples must be > 0.")
        if ver.std_tolerance_factor <= 0:
            raise ConfigError("verification.std_tolerance_factor must be > 0.")
        uniform_std = lwe.q / (12.0 ** 0.5)
        if lwe.sigma > 0 and ver.std_tolerance_factor * lwe.sigma >= uniform_std:
            warnings.append(
                "verification: the acceptance threshold "
                f"({ver.std_tolerance_factor * lwe.sigma:.1f}) is not separated from "
                f"the uniform-residual std ({uniform_std:.1f}); verification cannot "
                "discriminate a correct secret from a wrong one."
            )

        # -- device
        if dev.prefer not in ("cpu", "auto", "cuda", "mps"):
            raise ConfigError("device.prefer must be 'cpu', 'auto', 'cuda' or 'mps'.")
        if dev.dtype not in ("float32", "float16", "bfloat16"):
            raise ConfigError("device.dtype must be 'float32', 'float16' or 'bfloat16'.")
        if dev.threads is not None and dev.threads < 1:
            raise ConfigError("device.threads must be >= 1 when set.")
        if dev.interop_threads is not None and dev.interop_threads < 1:
            raise ConfigError("device.interop_threads must be >= 1 when set.")
        if dev.prefer == "cpu" and dev.dtype != "float32":
            warnings.append(
                f"device: dtype '{dev.dtype}' on CPU is often slower and less stable "
                "than float32."
            )

        return warnings


# --------------------------------------------------------------------------- #
# Loading helpers
# --------------------------------------------------------------------------- #
def _read_yaml(path: Path) -> Dict[str, Any]:
    """Read a YAML mapping from ``path``."""
    if not path.is_file():
        raise ConfigError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"Configuration file {path} must contain a YAML mapping.")
    return data


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``override`` into ``base`` (``override`` wins)."""
    merged: Dict[str, Any] = copy.deepcopy(dict(base))
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _resolve_raw(path: Path, _seen: Optional[List[Path]] = None) -> Dict[str, Any]:
    """Read a YAML file and recursively apply its ``extends`` parents."""
    path = path.resolve()
    seen = list(_seen or [])
    if path in seen:
        chain = " -> ".join(str(p) for p in seen + [path])
        raise ConfigError(f"Circular 'extends' chain detected: {chain}")
    seen.append(path)

    raw = _read_yaml(path)
    parents = raw.pop("extends", None)
    if parents is None:
        return raw
    if isinstance(parents, (str, Path)):
        parents = [parents]
    if not isinstance(parents, list):
        raise ConfigError(f"{path}: 'extends' must be a string or a list of strings.")

    merged: Dict[str, Any] = {}
    for parent in parents:
        parent_path = (path.parent / str(parent)).resolve()
        merged = _deep_merge(merged, _resolve_raw(parent_path, seen))
    return _deep_merge(merged, raw)


def _parse_scalar(text: str) -> Any:
    """Parse a CLI override value using YAML scalar rules."""
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError:
        return text


def apply_overrides(raw: Mapping[str, Any], overrides: Sequence[str]) -> Dict[str, Any]:
    """Apply ``section.key=value`` overrides to a raw configuration mapping.

    Args:
        raw: Nested configuration mapping.
        overrides: Strings such as ``"training.batch_size=64"`` or
            ``"experiment.tags=[sweep, cpu]"``.

    Returns:
        A new mapping with the overrides applied.

    Raises:
        ConfigError: If an override is not of the form ``key=value``.
    """
    result: Dict[str, Any] = copy.deepcopy(dict(raw))
    for item in overrides:
        if "=" not in item:
            raise ConfigError(
                f"Malformed override '{item}'; expected 'section.key=value'."
            )
        key, _, value = item.partition("=")
        key = key.strip()
        if not key:
            raise ConfigError(f"Malformed override '{item}'; empty key.")
        parts = key.split(".")
        cursor: Dict[str, Any] = result
        for part in parts[:-1]:
            nxt = cursor.get(part)
            if not isinstance(nxt, dict):
                nxt = {}
                cursor[part] = nxt
            cursor = nxt
        cursor[parts[-1]] = _parse_scalar(value.strip())
    return result


def load_config(
    path: "Path | str",
    overrides: Optional[Sequence[str]] = None,
    validate: bool = False,
) -> Config:
    """Load, merge, override and build a configuration.

    Args:
        path: Path to a YAML configuration file.  ``extends:`` entries are
            resolved relative to the file that declares them.
        overrides: Optional ``section.key=value`` strings applied last.
        validate: Run :meth:`Config.validate` before returning.

    Returns:
        The fully-resolved :class:`Config`.

    Raises:
        ConfigError: On unknown keys, bad types or (if requested) invalid values.
    """
    raw = _resolve_raw(Path(path))
    if overrides:
        raw = apply_overrides(raw, overrides)
    config = Config.from_dict(raw)
    if validate:
        config.validate()
    return config


def save_config(config: Config, path: "Path | str") -> Path:
    """Write ``config`` to ``path`` as YAML and return the path."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            config.to_dict(), handle, sort_keys=True, default_flow_style=False
        )
    return target


def config_fingerprint(config: Config, length: int = 16) -> str:
    """Return a stable SHA-256 fingerprint of a configuration.

    The fingerprint is computed over the canonical JSON serialisation, so it is
    independent of YAML formatting and key order.
    """
    payload = json.dumps(config.to_dict(), sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return digest[:length]


# --------------------------------------------------------------------------- #
# Strict dataclass construction
# --------------------------------------------------------------------------- #
def _build_dataclass(cls: Type[T], data: Mapping[str, Any], path: str) -> T:
    """Instantiate dataclass ``cls`` from ``data``, rejecting unknown keys."""
    if not isinstance(data, Mapping):
        raise ConfigError(
            f"{path or 'config'}: expected a mapping, got {type(data).__name__}."
        )
    hints = typing.get_type_hints(cls)
    fields = {f.name: f for f in dataclasses.fields(cls)}

    unknown = [k for k in data if k not in fields and k not in _META_KEYS]
    if unknown:
        where = path or "config"
        raise ConfigError(
            f"{where}: unknown key(s) {sorted(unknown)}. "
            f"Known keys: {sorted(fields)}. "
            "Unknown keys are rejected so experiment settings cannot drift silently."
        )

    kwargs: Dict[str, Any] = {}
    for name, dataclass_field in fields.items():
        if name not in data:
            continue
        child_path = f"{path}.{name}" if path else name
        kwargs[name] = _coerce(hints[name], data[name], child_path)
    return cls(**kwargs)  # type: ignore[call-arg]


def _coerce(annotation: Any, value: Any, path: str) -> Any:
    """Coerce ``value`` to ``annotation``, raising :class:`ConfigError` on failure."""
    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)

    # Optional[X] / Union[...]
    if origin is typing.Union:
        non_none = [a for a in args if a is not type(None)]  # noqa: E721
        if value is None:
            if type(None) in args:
                return None
            raise ConfigError(f"{path}: null is not allowed here.")
        if len(non_none) == 1:
            return _coerce(non_none[0], value, path)
        for candidate in non_none:  # best effort for wider unions
            try:
                return _coerce(candidate, value, path)
            except ConfigError:
                continue
        raise ConfigError(f"{path}: value {value!r} does not match {annotation}.")

    # containers
    if origin in (list, List):
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"{path}: expected a list, got {type(value).__name__}.")
        item_type = args[0] if args else Any
        return [_coerce(item_type, v, f"{path}[{i}]") for i, v in enumerate(value)]
    if origin in (dict, Dict):
        if not isinstance(value, Mapping):
            raise ConfigError(f"{path}: expected a mapping, got {type(value).__name__}.")
        key_type = args[0] if args else Any
        val_type = args[1] if len(args) > 1 else Any
        return {
            _coerce(key_type, k, f"{path}.<key>"): _coerce(val_type, v, f"{path}.{k}")
            for k, v in value.items()
        }
    if origin in (tuple, Tuple):
        if not isinstance(value, (list, tuple)):
            raise ConfigError(
                f"{path}: expected a sequence, got {type(value).__name__}."
            )
        return tuple(value)

    # nested dataclass sections
    if dataclasses.is_dataclass(annotation):
        return _build_dataclass(annotation, value, path)

    # scalars
    if annotation is Any:
        return value
    if annotation is bool:
        if isinstance(value, bool):
            return value
        raise ConfigError(f"{path}: expected a boolean, got {value!r}.")
    if annotation is int:
        if isinstance(value, bool):
            raise ConfigError(f"{path}: expected an integer, got a boolean.")
        if isinstance(value, int):
            return value
        if isinstance(value, float) and float(value).is_integer():
            return int(value)
        if isinstance(value, str):
            try:
                return int(value.strip())
            except ValueError as exc:
                raise ConfigError(
                    f"{path}: cannot read {value!r} as an integer."
                ) from exc
        raise ConfigError(f"{path}: expected an integer, got {value!r}.")
    if annotation is float:
        if isinstance(value, bool):
            raise ConfigError(f"{path}: expected a float, got a boolean.")
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value.strip())
            except ValueError as exc:
                raise ConfigError(f"{path}: cannot read {value!r} as a float.") from exc
        raise ConfigError(f"{path}: expected a float, got {value!r}.")
    if annotation is str:
        if isinstance(value, str):
            return value
        if isinstance(value, (int, float, bool)):
            return str(value)
        raise ConfigError(f"{path}: expected a string, got {value!r}.")

    return value


def _is_prime(number: int) -> bool:
    """Deterministic small-integer primality test (advisory checks only)."""
    if number < 2:
        return False
    if number % 2 == 0:
        return number == 2
    divisor = 3
    while divisor * divisor <= number:
        if number % divisor == 0:
            return False
        divisor += 2
    return True
