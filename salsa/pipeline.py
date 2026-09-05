"""The Salsa 2.0 end-to-end pipeline.

One entry point, :func:`run_salsa2_pipeline`, joins the stages that until now
lived in separate scripts::

    public LWE/RLWE data -> encoding -> NACT-F -> predict b
                         -> direct secret recovery -> candidate secret
                         -> independent residual verification -> result

Nothing is trained here.  The pipeline loads a completed checkpoint, verifies
its identity against the configuration, and runs inference, recovery and
verification.

Three results, kept apart on purpose
------------------------------------
The stages are reported separately because they have different epistemic
status, and collapsing them is how a recovery experiment fools itself:

``recovery``      produced from public probes and model predictions alone.  The
                  true secret is not in scope; :class:`~salsa.recovery.DirectRecovery`
                  has no parameter through which it could be passed.
``verification``  produced from fresh public samples and the candidate.  The
                  verifier's signature is ``(A, b, candidate, q, sigma)``; no
                  secret, model output or training label reaches it.
``evaluation``    the only stage that loads the true secret, and it runs last.
                  It scores what the earlier stages already decided; it never
                  feeds back into them.

The ordering is enforced by construction: the secret is not read until after
both a candidate and a verdict exist.
"""

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from .data import LatticeCodec, build_problem
from .models import build_model, count_trainable_parameters
from .recovery import DirectRecovery, probe_separation
from .training.metrics import chance_baselines
from .utils import load_config
from .verification import ACCEPTANCE_CRITERIA, verify_candidate

__all__ = [
    "PipelineResult",
    "PipelineError",
    "load_checkpoint",
    "locate_checkpoint",
    "run_salsa2_pipeline",
]


class PipelineError(RuntimeError):
    """Raised when the pipeline cannot proceed safely."""


# --------------------------------------------------------------------------- #
# Checkpoint discovery and identity
# --------------------------------------------------------------------------- #
def locate_checkpoint(run_dir: Path) -> Path:
    """Find a run's best checkpoint using the run's own metadata.

    The filename is never assumed: the run's ``summary.json`` names the epoch
    its monitor metric selected, and only then is the checkpoint directory
    consulted.

    Args:
        run_dir: A completed run directory.

    Returns:
        Path to the selected checkpoint.

    Raises:
        PipelineError: If no usable checkpoint is present.
    """
    summary_path = run_dir / "artifacts" / "summary.json"
    checkpoints = run_dir / "checkpoints"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text("utf-8"))
        if summary["state"].get("best_epoch") is not None and (checkpoints / "best.pt").is_file():
            return checkpoints / "best.pt"
    for name in ("best.pt", "last.pt"):
        if (checkpoints / name).is_file():
            return checkpoints / name
    raise PipelineError(f"no usable checkpoint under {checkpoints}")


def check_identity(checkpoint: Dict[str, Any], config, codec: LatticeCodec) -> List[str]:
    """Confirm the checkpoint is the model this configuration describes.

    Returns:
        A list of mismatch descriptions; empty means everything agrees.
    """
    saved, spec = checkpoint["config"], checkpoint["spec"]
    expectations = [
        ("n", config.lwe.n, saved["lwe"]["n"]),
        ("hamming_weight", config.lwe.resolved_hamming_weight, saved["lwe"]["hamming_weight"]),
        ("q", config.lwe.q, saved["lwe"]["q"]),
        ("sigma", config.lwe.sigma, saved["lwe"]["sigma"]),
        ("structure", config.lwe.structure, saved["lwe"]["structure"]),
        ("base", config.encoding.base, saved["encoding"]["base"]),
        ("digit_order", config.encoding.digit_order, saved["encoding"]["digit_order"]),
        ("separator", config.encoding.separator, saved["encoding"]["separator"]),
        ("fixed_width", config.encoding.fixed_width, saved["encoding"]["fixed_width"]),
        ("vocabulary", codec.vocabulary.size, spec["vocab_size"]),
        ("seed", config.experiment.seed, saved["experiment"]["seed"]),
        ("encoder_loops", config.resolved_encoder_loops, spec["encoder_loops"]),
        ("decoder_loops", config.resolved_decoder_loops, spec["decoder_loops"]),
    ]
    if config.pipeline.expect_parameters is not None:
        expectations.append(("parameter count", int(config.pipeline.expect_parameters),
                             checkpoint["parameter_count"]))
    if config.pipeline.expect_arch is not None:
        expectations.append(("architecture", str(config.pipeline.expect_arch), spec["arch"]))
    return [
        f"{name}: config/expected {expected!r} vs checkpoint {actual!r}"
        for name, expected, actual in expectations
        if expected != actual
    ]


def load_checkpoint(config) -> Dict[str, Any]:
    """Locate, load and identity-check the checkpoint this config points at.

    Args:
        config: A validated :class:`~salsa.utils.config.Config`.

    Returns:
        A dict with the loaded model, the raw checkpoint and provenance.

    Raises:
        PipelineError: If no checkpoint is configured, or its identity does not
            match the configuration.
    """
    if config.pipeline.checkpoint:
        path = Path(config.pipeline.checkpoint)
    elif config.pipeline.run_dir:
        path = locate_checkpoint(Path(config.pipeline.run_dir))
    else:
        raise PipelineError(
            "set pipeline.run_dir (preferred) or pipeline.checkpoint in the config.")
    if not path.is_file():
        raise PipelineError(f"checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    codec = LatticeCodec.from_config(config)
    mismatches = check_identity(checkpoint, config, codec)
    if mismatches:
        raise PipelineError(
            "checkpoint does not match the configuration:\n  "
            + "\n  ".join(mismatches))

    model = build_model(config)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    # Count BEFORE freezing: after requires_grad_(False) the trainable count is
    # zero by definition, which is a property of the freeze and not of the model.
    parameter_count = count_trainable_parameters(model)
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    return {
        "model": model,
        "codec": codec,
        "checkpoint": checkpoint,
        "path": path,
        "parameter_count": parameter_count,
        "trainable_after_freeze": count_trainable_parameters(model),
        "stored_parameter_count": checkpoint["parameter_count"],
        "config_fingerprint": checkpoint["config_fingerprint"],
        "best_epoch": checkpoint["state"]["best_epoch"],
        "samples_seen": checkpoint["state"]["samples_seen"],
        "architecture": checkpoint["spec"]["arch"],
        "spec": checkpoint["spec"],
    }


# --------------------------------------------------------------------------- #
# Result schema
# --------------------------------------------------------------------------- #
@dataclass
class PipelineResult:
    """Everything one end-to-end run produced.

    The three stages are separate fields, not merged, so a reader can always
    tell what was decided without the secret and what was scored with it.
    """

    label: str = ""
    config_path: str = ""
    config_fingerprint: str = ""
    checkpoint_fingerprint: str = ""
    architecture: str = ""
    model_name: str = ""
    parameter_count: int = 0
    checkpoint: str = ""
    best_epoch: Optional[int] = None
    samples_seen: Optional[int] = None
    instance: Dict[str, Any] = field(default_factory=dict)
    chance_baselines: Dict[str, Any] = field(default_factory=dict)
    # -- stage 1: recovery, secret-free ------------------------------------- #
    recovery_method: str = ""
    selection_rule: str = ""
    min_separation: int = 1
    k_values: List[int] = field(default_factory=list)
    k_separations: List[int] = field(default_factory=list)
    excluded_k: List[int] = field(default_factory=list)
    successful_k: List[int] = field(default_factory=list)
    selected_k: Optional[int] = None
    recovered_candidate: List[int] = field(default_factory=list)
    candidate_hamming_weight: int = 0
    probe_decode_validity: float = 0.0
    per_k: List[Dict[str, Any]] = field(default_factory=list)
    # -- stage 2: verification, secret-free --------------------------------- #
    verification_ran: bool = False
    verification_passed: Optional[bool] = None
    verification_samples: int = 0
    verification_split: str = ""
    verification_data_seed: Optional[int] = None
    residual_statistics: Dict[str, Any] = field(default_factory=dict)
    residual_baselines: Dict[str, Any] = field(default_factory=dict)
    verification_criteria: Dict[str, bool] = field(default_factory=dict)
    # -- stage 3: evaluation, the only stage that sees the secret ----------- #
    true_secret: List[int] = field(default_factory=list)
    coordinate_accuracy: float = 0.0
    hamming_distance: int = 0
    exact_recovery: bool = False
    baselines: Dict[str, Any] = field(default_factory=dict)
    # -- provenance --------------------------------------------------------- #
    runtime_seconds: float = 0.0
    stage_seconds: Dict[str, float] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable view."""
        return asdict(self)

    def summary_line(self) -> str:
        """One-line human summary."""
        verification = ("PASS" if self.verification_passed
                        else "FAIL" if self.verification_ran else "not run")
        return (f"{self.label}: n={self.instance.get('n')} h={self.instance.get('h')} "
                f"exact={'YES' if self.exact_recovery else 'NO'} "
                f"acc={self.coordinate_accuracy:.4f} "
                f"K={len(self.successful_k)}/{len(self.k_values)} "
                f"verification={verification}")


# --------------------------------------------------------------------------- #
# The pipeline
# --------------------------------------------------------------------------- #
def run_salsa2_pipeline(
    config_path: "str | Path",
    *,
    evaluate: bool = True,
    verify: Optional[bool] = None,
    output_dir: Optional["str | Path"] = None,
) -> PipelineResult:
    """Run data -> encoding -> model -> recovery -> verification -> result.

    Args:
        config_path: YAML configuration describing the instance, the encoding,
            the checkpoint, the K sweep and the verification budget.
        evaluate: Whether to run the final ground-truth comparison. Set False
            for a genuine attack setting, where no secret is available.
        verify: Override ``verification.enabled`` from the config.
        output_dir: Override ``pipeline.output_dir``.

    Returns:
        A :class:`PipelineResult`.

    Raises:
        PipelineError: On a missing or mismatched checkpoint.
    """
    started = time.perf_counter()
    stage_times: Dict[str, float] = {}

    config = load_config(config_path)
    config.validate()
    q, n, sigma = config.lwe.q, config.lwe.n, config.lwe.sigma
    weight = config.lwe.resolved_hamming_weight

    mark = time.perf_counter()
    loaded = load_checkpoint(config)
    stage_times["load_checkpoint"] = time.perf_counter() - mark
    model, codec = loaded["model"], loaded["codec"]

    # ---------------- STAGE 1: recovery.  No secret is in scope. ---------- #
    mark = time.perf_counter()
    k_values = [int(k) for k in config.recovery.direct_k_values]
    # The anchored rule is the only decision rule free of polarity ambiguity,
    # which is why phase 10 adopted it; the released code's thresholding rules
    # resolve polarity against the true secret and cannot be part of an attack.
    recovery = DirectRecovery(model, codec, method="anchor")
    report = recovery.recover(
        k_values,
        selection_rule=config.recovery.selection_rule,
        min_separation=config.recovery.min_separation,
        sigma=sigma,
    )
    candidate = report.primary_candidate
    stage_times["recovery"] = time.perf_counter() - mark

    per_k = []
    for entry in report.per_k:
        decoded = [o.decoded_b for o in entry.outcomes]
        readable = [v for v, o in zip(decoded, entry.outcomes) if o.decoded]
        per_k.append({
            "K": entry.K, "reduced_K": entry.reduced_K,
            "separation": entry.separation,
            "eligible_for_selection": entry.separation >= report.min_separation,
            "decode_validity": 1.0 - entry.decode_failure_rate,
            "mean_margin": entry.mean_margin,
            "unique_predictions": int(np.unique(readable).size) if readable else 0,
            "candidate": entry.candidate.tolist(),
            "candidate_hamming_weight": entry.recovered_weight,
            "predictions_by_coordinate": decoded,
        })

    result = PipelineResult(
        label=config.pipeline.label or config.experiment.name,
        config_path=str(config_path),
        config_fingerprint=config.fingerprint(),
        checkpoint_fingerprint=loaded["config_fingerprint"],
        architecture=loaded["architecture"],
        model_name=getattr(model, "name", type(model).__name__),
        parameter_count=loaded["parameter_count"],
        checkpoint=str(loaded["path"]),
        best_epoch=loaded["best_epoch"],
        samples_seen=loaded["samples_seen"],
        instance={"n": n, "h": weight, "q": q, "sigma": sigma,
                  "structure": config.lwe.structure,
                  "rlwe_variant": config.lwe.rlwe_variant,
                  "base": config.encoding.base,
                  "digit_order": config.encoding.digit_order,
                  "separator": config.encoding.separator,
                  "fixed_width": config.encoding.fixed_width,
                  "vocabulary": codec.vocabulary.size,
                  "encoder_sequence_length": n + 2 if loaded["architecture"] == "nact"
                  else codec.input_length,
                  "codec_input_length": codec.input_length,
                  "output_length": codec.output_length,
                  "tolerance_tau": config.evaluation.tolerance,
                  "search_space": _binomial(n, weight)},
        chance_baselines=chance_baselines(codec, config.evaluation.tolerance),
        recovery_method=report.method,
        selection_rule=report.selection_rule,
        min_separation=report.min_separation,
        k_values=k_values,
        k_separations=[probe_separation(k, q) for k in k_values],
        excluded_k=report.excluded_k,
        selected_k=report.selected_K,
        recovered_candidate=candidate.tolist(),
        candidate_hamming_weight=int(candidate.sum()),
        probe_decode_validity=float(np.mean([e["decode_validity"] for e in per_k])),
        per_k=per_k,
    )

    # ---------------- STAGE 2: verification.  Still no secret. ------------ #
    should_verify = config.verification.enabled if verify is None else bool(verify)
    if should_verify:
        mark = time.perf_counter()
        problem = build_problem(config, split=config.verification.split_label)
        labeled = problem.labeled_sample(
            config.verification.num_samples,
            rng=np.random.default_rng(problem.data_seed))
        # Only A and b cross into the verifier.
        verification = verify_candidate(
            labeled.public.A, labeled.public.b, candidate, q, sigma,
            baselines={"all_zeros": np.zeros(n, dtype=np.int64),
                       "all_ones": np.ones(n, dtype=np.int64)},
        )
        stage_times["verification"] = time.perf_counter() - mark
        result.verification_ran = True
        result.verification_passed = verification.passes
        result.verification_samples = verification.samples
        result.verification_split = config.verification.split_label
        result.verification_data_seed = problem.data_seed
        result.residual_statistics = verification.statistics
        result.residual_baselines = verification.baseline_statistics
        result.verification_criteria = verification.criteria

    # ---------------- STAGE 3: evaluation.  The secret enters HERE. ------- #
    if evaluate:
        mark = time.perf_counter()
        from .data.secrets import secret_from_config

        secret = secret_from_config(config)
        matches = int((candidate == secret).sum())
        result.true_secret = secret.tolist()
        result.coordinate_accuracy = matches / n
        result.hamming_distance = int((candidate != secret).sum())
        result.exact_recovery = bool(np.array_equal(candidate, secret))
        result.successful_k = [
            entry.K for entry in report.per_k
            if np.array_equal(entry.candidate, secret)
        ]
        free = (n - weight) / n
        result.baselines = {
            "all_zeros_coordinate_accuracy": free,
            "all_ones_coordinate_accuracy": weight / n,
            "warning": (f"A weight-{weight} secret over {n} coordinates gives an "
                        f"all-zeros guess {free:.4f} coordinate accuracy for free. "
                        "Only exact recovery is evidence."),
        }
        stage_times["evaluation"] = time.perf_counter() - mark

    result.stage_seconds = {k: round(v, 4) for k, v in stage_times.items()}
    result.runtime_seconds = round(time.perf_counter() - started, 4)
    result.notes = [
        "Recovery saw only public probes and model predictions; DirectRecovery "
        "has no parameter through which a secret could be passed.",
        f"Verification received only (A, b, candidate, q, sigma). Criteria: "
        f"{list(ACCEPTANCE_CRITERIA)}.",
        "The true secret was loaded only in the evaluation stage, after both a "
        "candidate and a verification verdict existed.",
    ]
    result.limitations = [
        f"Diagnostic scale: n={n}, h={weight} has C({n},{weight}) = "
        f"{_binomial(n, weight)} possible secrets.",
        "One secret, one checkpoint, one seed per configuration.",
        "Not a practical LWE break and not evidence about larger dimensions.",
    ]

    target = Path(output_dir or config.pipeline.output_dir)
    target.mkdir(parents=True, exist_ok=True)
    name = result.label.replace(" ", "_")
    (target / f"{name}_result.json").write_text(
        json.dumps(result.to_dict(), indent=2, default=str), encoding="utf-8")
    return result


def _binomial(n: int, k: int) -> int:
    """Number of weight-``k`` binary vectors of length ``n``."""
    import math

    return math.comb(int(n), int(k)) if 0 <= k <= n else 0
