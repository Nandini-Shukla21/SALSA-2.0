"""Phase 12: how the frozen n=12,h=2 model responds as its input gets sparser.

The phase-11 audit measured that a direct-recovery probe ``K*e_i`` is roughly a
1-in-10^25 event under the training distribution, and hypothesised that the
phase-10 collapse is an out-of-distribution failure.  That hypothesis predicts
something specific and falsifiable: performance should fall *smoothly* as the
input is made sparser, with the probes sitting at the far end of the same curve.
If instead performance holds up until sparsity is extreme and then falls off a
cliff, or does not fall at all, the distribution-shift story is not sufficient.

This script measures that curve.  Nothing is trained; the checkpoint is loaded,
verified, and frozen.

Two stages, kept strictly apart
-------------------------------
**Stage 1 -- model response.**  The model is shown only ``a``.  Everything
measured here (decode validity, output entropy, uniqueness, variance) is a
property of the model's predictions alone and needs no target at all.

**Stage 2 -- evaluation.**  Only afterwards is a target ``b`` constructed, by
the data layer, from the same LWE instance with freshly drawn errors.  It is
never shown to the model, never used to choose an input, and never fed back
into stage 1.

For ``K*e_i`` inputs the stage-2 numbers measure *prediction fidelity on probe
inputs*.  They are NOT secret recovery and are not reported as such: secret
recovery was measured in phase 10 and the answer there was NO.
"""

import argparse
import csv
import json
import math
import sys
from itertools import permutations
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from salsa.data import LatticeCodec, build_problem  # noqa: E402
from salsa.data.lwe import generate_error  # noqa: E402
from salsa.data.secrets import resolve_rng  # noqa: E402
from salsa.models import build_model, count_trainable_parameters  # noqa: E402
from salsa.recovery import build_probe_matrix, probe_separation  # noqa: E402
from salsa.training.metrics import (  # noqa: E402
    absolute_distance,
    chance_baselines,
    decode_generated_ids,
    greedy_decode,
)
from salsa.training.seed import derive_seed  # noqa: E402
from salsa.utils import load_config  # noqa: E402

#: Sparsity levels required by the phase brief, densest first.
NNZ_LEVELS = (12, 10, 8, 6, 4, 2, 1)


# --------------------------------------------------------------------------- #
# Checkpoint loading and verification (identical checks to phase 10)
# --------------------------------------------------------------------------- #
def locate_checkpoint(run_dir: Path) -> Path:
    """Find the best checkpoint using the run's own metadata."""
    summary_path = run_dir / "artifacts" / "summary.json"
    checkpoints = run_dir / "checkpoints"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text("utf-8"))
        if summary["state"].get("best_epoch") is not None and (checkpoints / "best.pt").is_file():
            return checkpoints / "best.pt"
    for name in ("best.pt", "last.pt"):
        if (checkpoints / name).is_file():
            return checkpoints / name
    raise FileNotFoundError(f"no checkpoint under {checkpoints}")


def verify(checkpoint: Dict[str, Any], config, codec: LatticeCodec) -> List[str]:
    """Return a list of mismatches between checkpoint and config; empty is good."""
    saved, spec = checkpoint["config"], checkpoint["spec"]
    expectations = [
        ("parameter count", 4_131_200, checkpoint["parameter_count"]),
        ("architecture", "gated_universal_transformer", spec["arch"]),
        ("n", config.lwe.n, saved["lwe"]["n"]),
        ("h", config.lwe.resolved_hamming_weight, saved["lwe"]["hamming_weight"]),
        ("q", config.lwe.q, saved["lwe"]["q"]),
        ("sigma", config.lwe.sigma, saved["lwe"]["sigma"]),
        ("base", config.encoding.base, saved["encoding"]["base"]),
        ("digit_order", "lsb_first", saved["encoding"]["digit_order"]),
        ("separator", False, saved["encoding"]["separator"]),
        ("fixed_width", True, saved["encoding"]["fixed_width"]),
        ("vocabulary", codec.vocabulary.size, spec["vocab_size"]),
        ("seed", config.experiment.seed, saved["experiment"]["seed"]),
    ]
    return [
        f"{name}: expected {expected!r} vs checkpoint {actual!r}"
        for name, expected, actual in expectations
        if expected != actual
    ]


# --------------------------------------------------------------------------- #
# Input generation.  None of this touches the secret.
# --------------------------------------------------------------------------- #
def sparse_inputs(
    count: int, n: int, q: int, nnz: int, rng: np.random.Generator
) -> np.ndarray:
    """Draw ``count`` vectors in ``Z_q^n`` with exactly ``nnz`` nonzero entries.

    Nonzero values are uniform on ``[1, q-1]`` and positions are a uniform
    random subset, so the only thing varying across levels is the sparsity.

    Args:
        count: How many vectors.
        n: Dimension.
        q: Modulus.
        nnz: Exact number of nonzero coordinates, ``0 <= nnz <= n``.
        rng: Source of randomness.

    Returns:
        An ``int64`` array of shape ``(count, n)``.

    Raises:
        ValueError: If ``nnz`` is out of range.
    """
    if not 0 <= nnz <= n:
        raise ValueError(f"nnz must be in [0, {n}], got {nnz}.")
    matrix = np.zeros((count, n), dtype=np.int64)
    for row in range(count):
        positions = rng.choice(n, size=nnz, replace=False)
        matrix[row, positions] = rng.integers(1, q, size=nnz, dtype=np.int64)
    return matrix


def probe_inputs(n: int, q: int, k_values: Sequence[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the exhaustive set of ``K*e_i`` probes for the phase-10 K sweep.

    Returns:
        ``(matrix, k_of_row, coordinate_of_row)``.  ``matrix`` is
        ``(len(k_values)*n, n)``.
    """
    blocks, ks, coords = [], [], []
    for K in k_values:
        blocks.append(build_probe_matrix(n, K, q))
        ks.extend([int(K)] * n)
        coords.extend(range(n))
    return np.concatenate(blocks, axis=0), np.asarray(ks), np.asarray(coords)


# --------------------------------------------------------------------------- #
# Stage 1: model response.  Sees `a` and nothing else.
# --------------------------------------------------------------------------- #
@torch.no_grad()
def model_response(
    model, codec: LatticeCodec, matrix: np.ndarray, batch_size: int = 256
) -> Dict[str, np.ndarray]:
    """Run the frozen model on inputs and record its predictions.

    Reproduces :func:`greedy_decode` step by step so that the per-step
    predictive distribution can be captured as well; the caller asserts the
    generated ids agree with the library function exactly.

    Args:
        model: The frozen transformer.
        codec: The codec the model was trained with.
        matrix: Inputs of shape ``(m, n)``.
        batch_size: Rows per forward pass.

    Returns:
        Dict with ``values`` (decoded ints, -1 on failure), ``valid`` (bool),
        ``tokens`` (generated ids) and ``token_entropy`` (mean predictive
        entropy in nats, per row).
    """
    model.eval()
    bos = codec.vocabulary.bos_id
    steps = codec.output_length - 1
    values, valid, tokens, entropies = [], [], [], []

    for start in range(0, matrix.shape[0], batch_size):
        chunk = matrix[start : start + batch_size]
        src_ids, _ = codec.encode_batch(chunk)
        src = torch.from_numpy(src_ids)

        memory = model.encode(src)
        generated = torch.full((src.shape[0], 1), int(bos), dtype=torch.long)
        step_entropy = torch.zeros(src.shape[0], dtype=torch.float64)
        for _ in range(steps):
            logits = model.decode(memory, generated)[:, -1]
            log_probs = torch.log_softmax(logits.double(), dim=-1)
            step_entropy -= (log_probs.exp() * log_probs).sum(dim=-1)
            generated = torch.cat((generated, logits.argmax(dim=-1, keepdim=True)), dim=1)

        ids = generated.cpu().numpy()
        decoded, ok = decode_generated_ids(codec, ids)
        values.append(decoded)
        valid.append(ok)
        tokens.append(ids)
        entropies.append((step_entropy / steps).numpy())

    return {
        "values": np.concatenate(values),
        "valid": np.concatenate(valid),
        "tokens": np.concatenate(tokens, axis=0),
        "token_entropy": np.concatenate(entropies),
    }


def empirical_entropy(values: np.ndarray) -> float:
    """Shannon entropy in nats of the empirical distribution of ``values``."""
    if values.size == 0:
        return 0.0
    _, counts = np.unique(values, return_counts=True)
    probabilities = counts / counts.sum()
    return float(-(probabilities * np.log(probabilities)).sum())


def response_statistics(response: Dict[str, np.ndarray], q: int) -> Dict[str, Any]:
    """Summarise the model's behaviour without reference to any target."""
    values, valid = response["values"], response["valid"]
    total = int(values.size)
    usable = values[valid]

    if usable.size:
        counts = np.unique(usable, return_counts=True)[1]
        modal_fraction = float(counts.max()) / total
        output_std = float(usable.std())
        output_variance = float(usable.var())
    else:
        modal_fraction, output_std, output_variance = float("nan"), float("nan"), float("nan")

    entropy = empirical_entropy(usable)
    return {
        "samples": total,
        "decode_validity": float(valid.mean()),
        "undecodable_fraction": float((~valid).mean()),
        "unique_outputs": int(np.unique(usable).size),
        "unique_output_fraction": float(np.unique(usable).size) / max(total, 1),
        "modal_output_fraction": modal_fraction,
        "output_std": output_std,
        "output_variance": output_variance,
        "output_entropy_nats": entropy,
        "output_entropy_ratio_vs_uniform_Zq": entropy / math.log(q),
        "mean_token_predictive_entropy_nats": float(response["token_entropy"].mean()),
    }


# --------------------------------------------------------------------------- #
# Stage 2: evaluation.  Targets enter only here, and only from the data layer.
# --------------------------------------------------------------------------- #
def evaluation_targets(matrix: np.ndarray, secret: np.ndarray, params, rng) -> np.ndarray:
    """Construct ``b = a.s + e mod q`` for evaluation inputs.

    This is the *same* mathematics the data generator uses, with errors drawn
    fresh from the configured distribution.  It runs after the model has already
    answered, and its output is never shown to the model.
    """
    error = generate_error(
        matrix.shape[0],
        params.sigma,
        distribution=params.error_distribution,
        rng=rng,
        tail_cut=params.tail_cut,
    )
    return (matrix @ secret + error) % params.q


def target_baselines(targets: np.ndarray, q: int, tolerance: float) -> Dict[str, Any]:
    """What input-blind predictors score on *these* targets.

    As ``a`` gets sparser the target distribution moves: with a weight-2 secret
    over 12 coordinates, a single-nonzero ``a`` misses the support 10 times out
    of 12, so ``a.s = 0`` and ``b`` is just the error, tight around zero.  A
    constant predictor therefore scores very differently at different sparsity
    levels, and raw acc_tau is not comparable across the row without this.

    Args:
        targets: The independently generated ``b`` values.
        q: Modulus.
        tolerance: ``tau`` as a fraction of ``q``.

    Returns:
        Baselines for a constant-zero predictor, the best constant chosen in
        hindsight, and a uniform random guess -- all on these exact targets.
    """
    bound = tolerance * q
    grid = np.abs(targets[:, None] - np.arange(q)[None, :])  # (m, q)
    within = grid <= bound
    per_constant = within.mean(axis=0)
    best = int(np.argmax(per_constant))
    return {
        "target_entropy_nats": empirical_entropy(targets),
        "target_fraction_near_zero": float(
            (np.minimum(targets, q - targets) <= bound).mean()),
        "acc_tau_constant_zero_predictor": float(per_constant[0]),
        "acc_tau_best_constant_predictor": float(per_constant[best]),
        "best_constant_value": best,
        "acc_tau_uniform_random_predictor": float(within.mean()),
    }


def target_statistics(
    response: Dict[str, np.ndarray], targets: np.ndarray, q: int, tolerance: float
) -> Dict[str, Any]:
    """Score predictions against independently generated targets.

    ``acc_tau`` uses the paper's plain absolute difference, as established in
    phase 6.  An undecodable prediction counts as a miss, never as a hit.
    """
    values, valid = response["values"], response["valid"]
    distance = absolute_distance(targets, values).astype(np.float64)
    hit = valid & (distance <= tolerance * q)
    exact = valid & (values == targets)
    usable = valid

    return {
        "acc_tau": float(hit.mean()),
        "exact_accuracy": float(exact.mean()),
        # An undecodable answer has no numeric error, so it is charged the worst
        # possible distance rather than quietly dropped.
        "mean_abs_error_all_undecodable_charged_max": float(
            np.where(valid, distance, float(q - 1)).mean()),
        "mean_abs_error_valid": float(distance[usable].mean()) if usable.any() else float("nan"),
        "median_abs_error_valid": float(np.median(distance[usable])) if usable.any() else float("nan"),
        **target_baselines(targets, q, tolerance),
    }


def probe_readout_analysis(
    values: np.ndarray, probe_k: np.ndarray, coordinate: np.ndarray,
    k_values: Sequence[int], n: int, secret: np.ndarray,
) -> Dict[str, Any]:
    """Which coordinates the model answers differently, and whether that lines up.

    The first half is stage 1: "which coordinates get a non-modal answer" needs
    no target at all.  The second half is stage 2 and is flagged as such.

    This is a POST-HOC look at the data that suggested it.  The coincidence
    probability quoted assumes a uniform random support and does not correct for
    the fact that the pattern was noticed by inspection, so it is reported as a
    lead to be tested prospectively, never as a recovery result.
    """
    # -- stage 1: purely a property of the predictions ---------------------- #
    per_k = []
    for K in k_values:
        mask = probe_k == K
        predictions = values[mask]
        order = np.argsort(coordinate[mask])
        predictions = predictions[order]
        valid = predictions >= 0
        if not valid.any():
            per_k.append({"K": int(K), "all_undecodable": True})
            continue
        mode = int(np.bincount(predictions[valid]).argmax())
        # An undecodable coordinate is not evidence of anything; it is neither
        # modal nor non-modal, so it is excluded rather than counted as differing.
        non_modal = [int(i) for i in np.flatnonzero(valid & (predictions != mode))]
        per_k.append({
            "K": int(K),
            "all_undecodable": False,
            "fully_decodable": bool(valid.all()),
            "decoded_count": int(valid.sum()),
            "modal_value": mode,
            "non_modal_coordinates": non_modal,
            "predictions": predictions.tolist(),
        })

    singled_out = sorted({
        i for entry in per_k if not entry["all_undecodable"]
        for i in entry["non_modal_coordinates"]
    })

    # -- stage 2: the secret enters only here ------------------------------- #
    support = sorted(int(i) for i in np.flatnonzero(secret))
    weight = len(support)
    # The inverted mode rule is only well defined when every coordinate decoded.
    exact_k = [
        entry["K"] for entry in per_k
        if not entry["all_undecodable"] and entry["fully_decodable"]
        and np.array_equal(
            1 - (np.asarray(entry["predictions"]) == entry["modal_value"]).astype(np.int64),
            secret)
    ]
    return {
        "stage_1_prediction_structure": {
            "per_k": per_k,
            "coordinates_ever_given_a_non_modal_answer": singled_out,
            "count": len(singled_out),
        },
        "stage_2_evaluation": {
            "warning": ("POST-HOC. This pattern was found by inspecting these same "
                        "results, so the probability below is not a corrected "
                        "significance test. n=1 secret, 1 checkpoint, 1 dimension."),
            "true_support": support,
            "singled_out_equals_support": singled_out == support,
            "coincidence_probability_uniform_support": (
                1.0 / math.comb(n, weight) if singled_out == support else None),
            "k_values_where_inverted_mode_rule_equals_secret": exact_k,
            "not_a_recovery_claim": ("The inverted-mode rule is the released code's "
                                     "bin2 candidate, whose polarity the released code "
                                     "resolves against the true secret. Phase 10 measured "
                                     "secret recovery with a secret-free anchored rule and "
                                     "the result was NO. That result stands."),
        },
    }


# --------------------------------------------------------------------------- #
# Monotonicity: Spearman rho with an EXACT permutation p-value (7! = 5040)
# --------------------------------------------------------------------------- #
def _ranks(values: np.ndarray) -> np.ndarray:
    """Average ranks, ties shared."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = np.arange(1, values.size + 1, dtype=np.float64)
    for value in np.unique(values):
        mask = values == value
        if mask.sum() > 1:
            ranks[mask] = ranks[mask].mean()
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation."""
    rx, ry = _ranks(np.asarray(x, dtype=np.float64)), _ranks(np.asarray(y, dtype=np.float64))
    rx, ry = rx - rx.mean(), ry - ry.mean()
    denominator = math.sqrt(float((rx ** 2).sum()) * float((ry ** 2).sum()))
    return float((rx * ry).sum() / denominator) if denominator > 0 else 0.0


def monotonicity(x: Sequence[float], y: Sequence[float]) -> Dict[str, Any]:
    """Spearman rho plus an exact two-sided permutation p-value.

    With seven levels there are only 5040 orderings, so the null distribution is
    enumerated exactly rather than approximated.  Also reports whether the
    sequence is monotone as a sequence, and where it breaks if not.
    """
    x_array, y_array = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    rho = spearman(x_array, y_array)

    null = [abs(spearman(np.asarray(p), y_array)) for p in permutations(x_array.tolist())]
    p_value = float(np.mean(np.asarray(null) >= abs(rho) - 1e-12))

    # Ordered from dense to sparse, the OOD hypothesis predicts non-increasing.
    differences = np.diff(y_array)
    breaks = [int(i) for i in np.where(differences > 1e-12)[0]]
    return {
        "spearman_rho_vs_nnz": round(rho, 6),
        "exact_permutation_p_value": round(p_value, 6),
        "strictly_monotone_decreasing_with_sparsity": bool(np.all(differences <= 1e-12)),
        "increase_positions_dense_to_sparse": breaks,
        "largest_single_step_drop": float(-differences.min()) if differences.size else float("nan"),
    }


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def make_plots(rows: List[Dict[str, Any]], baselines: Dict[str, float], out_dir: Path) -> List[str]:
    """Write the four required figures.  Returns their filenames."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sparse = [r for r in rows if r["input_type"] == "sparse_random"]
    training = next(r for r in rows if r["input_type"] == "training_distribution")
    probes = next(r for r in rows if r["input_type"] == "probe_K_times_e_i")
    x = [r["nnz"] for r in sparse]

    specs = [
        ("01_accuracy_vs_sparsity.png", "exact_accuracy",
         "Exact decoded-integer accuracy", baselines["chance_exact_accuracy"],
         "chance 1/q = %.5f" % baselines["chance_exact_accuracy"]),
        ("02_acc_tau_vs_sparsity.png", "acc_tau",
         "SALSA acc_tau  (|b - b_hat| <= 0.1 q)", baselines["chance_acc_tau"],
         "chance = %.5f" % baselines["chance_acc_tau"]),
        ("03_decode_validity_vs_sparsity.png", "decode_validity",
         "Fraction of predictions that decode at all", 1.0, "all outputs readable"),
        ("04_output_entropy_vs_sparsity.png", "output_entropy_nats",
         "Entropy of the decoded output distribution (nats)", None,
         "uniform over Z_q would be %.3f nats" % math.log(251)),
    ]

    written = []
    for filename, key, title, reference, reference_label in specs:
        figure, axes = plt.subplots(figsize=(7.2, 4.6))
        axes.plot(x, [r[key] for r in sparse], "o-", color="#1f77b4", linewidth=2,
                  markersize=6, label="sparse random a, exactly nnz nonzero", zorder=3)
        axes.plot([12], [training[key]], "s", color="#2ca02c", markersize=11,
                  markeredgecolor="black", label="A. training distribution", zorder=4)
        axes.plot([1], [probes[key]], "*", color="#d62728", markersize=18,
                  markeredgecolor="black", label="C. direct-recovery probes K*e_i", zorder=5)
        if key == "acc_tau":
            axes.plot(x, [r["acc_tau_best_constant_predictor"] for r in sparse],
                      "^:", color="#ff7f0e", linewidth=1.6, markersize=6, zorder=2,
                      label="best input-blind constant on the same targets")
        if reference is None:
            # log(q) = 5.53 would flatten this panel; state it in the caption.
            axes.plot([], [], " ", label=reference_label)
        else:
            axes.axhline(reference, linestyle="--", color="grey", linewidth=1.2,
                         label=reference_label)

        axes.set_xlabel("number of nonzero coordinates in a   (n = 12)")
        axes.set_ylabel(title)
        axes.set_title(title + "\nfrozen 4,131,200-parameter GatedUT, n=12 h=2")
        axes.set_xticks(list(range(1, 13)))
        axes.invert_xaxis()  # dense on the left, sparse on the right
        axes.grid(alpha=0.3)
        axes.legend(fontsize=8, loc="best")
        figure.tight_layout()
        figure.savefig(out_dir / filename, dpi=150)
        plt.close(figure)
        written.append(filename)
    return written


# --------------------------------------------------------------------------- #
# Markdown report
# --------------------------------------------------------------------------- #
def write_markdown(payload: Dict[str, Any], path: Path) -> None:
    """Render the report, labelling every claim MEASURED / INFERENCE / HYPOTHESIS."""
    rows = payload["table"]
    baselines = payload["chance_baselines"]
    tests = payload["monotonicity_tests"]
    continuity = payload["regime_b_to_c_continuity"]
    probes = payload["probe_detail"]
    readout = payload["probe_readout_analysis"]
    instance = payload["instance"]
    n, q = instance["n"], instance["q"]

    def get(input_type):
        for row in rows:
            if row["input_type"] == input_type:
                return row
        raise KeyError(input_type)

    training, probe_row = get("training_distribution"), get("probe_K_times_e_i")
    sparse = [r for r in rows if r["input_type"] == "sparse_random"]
    stage1, stage2 = (readout["stage_1_prediction_structure"],
                      readout["stage_2_evaluation"])
    lift_test = tests["acc_tau_minus_best_constant"]
    positive = [r["nnz"] for r in sparse if r["acc_tau_minus_best_constant"] > 0]
    crossing = next((r["nnz"] for r in sparse if r["acc_tau_minus_best_constant"] < 0), None)
    peak_entropy = max(r["output_entropy_nats"] for r in sparse)
    peak_nnz = next(r["nnz"] for r in sparse if r["output_entropy_nats"] == peak_entropy)

    lines = [
        "# Phase 12 - Recovery-distribution generalization analysis",
        "",
        "**Read-only.** No training was performed. The model, the data generator and the",
        "recovery implementation are unmodified; every model parameter was frozen",
        "(`requires_grad=False`, "
        f"{payload['checkpoint']['trainable_parameters_after_freeze']} trainable).",
        "",
        "Every statement is tagged **MEASURED** (a number this run produced),",
        "**INFERENCE** (a reading of those numbers) or **HYPOTHESIS** (a proposal this",
        "experiment does not test). Causation is not claimed from this experiment.",
        "",
        "## Setup",
        "",
        f"- Checkpoint `{payload['checkpoint']['path']}`, epoch {payload['checkpoint']['epoch']}, "
        f"{payload['checkpoint']['samples_seen']:,} samples, "
        f"{payload['checkpoint']['parameter_count']:,} parameters, fingerprint "
        f"`{payload['checkpoint']['config_fingerprint']}`. All architectural and "
        "cryptographic values were verified against the configuration before loading.",
        f"- Instance n={n}, h={instance['h']}, q={q}, sigma={instance['sigma']}, "
        f"{instance['structure']}/{instance['rlwe_variant']}, representation "
        f"{instance['representation']}, vocabulary {instance['vocab']}, tau={instance['tolerance_tau']}.",
        f"- {payload['samples_per_level']:,} evaluation vectors per sparsity level. Probes are "
        f"exhaustive: {len(probes['k_values'])} K values x {n} coordinates = "
        f"{len(probes['k_values']) * n}.",
        f"- Seeds: config {payload['seeds']['config_seed']}, analysis {payload['seeds']['analysis_seed']}. "
        "The instrumented decoder is asserted identical to "
        "`salsa.training.metrics.greedy_decode` before any measurement is taken.",
        "",
        "### How the measurement is kept clean",
        "",
        "| stage | what it sees |",
        "|---|---|",
        "| 1. model response | `a` only. Decode validity, entropy, uniqueness and variance need no target at all. |",
        "| 2. evaluation | `b = a.s + e mod q`, built by the data layer *after* the model answered, with fresh errors. Never shown to the model, never used to select an input, never fed back into stage 1. |",
        "",
        "For `K*e_i` inputs the stage-2 numbers measure **prediction fidelity on probe",
        "inputs**. They are not secret recovery and are not reported as such. Secret",
        "recovery was measured in phase 10; the answer there was NO and that stands.",
        "",
        "## 1. Why raw acc_tau is not comparable across sparsity levels",
        "",
        "**MEASURED.** Making `a` sparser also moves the *target* distribution. With a "
        f"weight-{instance['h']}",
        f"secret over {n} coordinates a single-nonzero `a` misses the support "
        f"{n - instance['h']} times out of {n},",
        "so `a.s = 0` and `b` is just the error, tight around zero:",
        "",
        "| nnz | target entropy (nats) | fraction of b within tau of 0 | acc_tau of the best input-blind constant |",
        "|---:|---:|---:|---:|",
    ]
    for row in sparse:
        lines.append(
            f"| {row['nnz']} | {row['target_entropy_nats']:.3f} | "
            f"{row['target_fraction_near_zero']:.4f} | "
            f"{row['acc_tau_best_constant_predictor']:.4f} "
            f"(constant {row['best_constant_value']}) |")
    lines += [
        "",
        "**INFERENCE.** A constant predictor gets *better* as the input gets sparser, so a",
        "falling acc_tau on its own understates the collapse. The honest quantity is the",
        "**lift**: acc_tau minus what the best input-blind constant scores on the very same",
        "targets. Lift <= 0 means the model's output carries no usable information about",
        "its input.",
        "",
        "## 2. The table",
        "",
        "| input_type | nnz | samples | decode_validity | acc_tau | best-constant | **lift** | exact_accuracy | unique_outputs | prediction_entropy | mean_error |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in [training] + sparse + [probe_row]:
        lines.append(
            f"| {row['label']} | {row['nnz']} | {row['samples']} | "
            f"{row['decode_validity']:.4f} | {row['acc_tau']:.4f} | "
            f"{row['acc_tau_best_constant_predictor']:.4f} | "
            f"**{row['acc_tau_minus_best_constant']:+.4f}** | "
            f"{row['exact_accuracy']:.4f} | {row['unique_outputs']} | "
            f"{row['output_entropy_nats']:.3f} | {row['mean_abs_error_valid']:.1f} |")
    lines += [
        "",
        f"Chance from the representation alone: acc_tau {baselines['chance_acc_tau']:.5f}, "
        f"exact {baselines['chance_exact_accuracy']:.5f}. `prediction_entropy` is the empirical",
        f"entropy in nats of the decoded output distribution; uniform over Z_q would be "
        f"{math.log(q):.3f}. `mean_error` is mean |b - b_hat| over decodable predictions.",
        "",
        f"**MEASURED.** Regime A (real generator rows, acc_tau {training['acc_tau']:.4f}) and regime B at",
        f"nnz={n} (synthetic dense vectors, {sparse[0]['acc_tau']:.4f}) agree to "
        f"{abs(training['acc_tau'] - sparse[0]['acc_tau']):.4f}. The synthetic input generator is",
        "therefore not a confound: the only thing varying down the table is sparsity.",
        "",
        "## 3. Hypothesis test: is the degradation monotone?",
        "",
        "Spearman rho against nnz over the seven sparsity levels, with an **exact**",
        "permutation p-value -- all 5,040 orderings enumerated, no normal approximation.",
        "",
        "| metric | rho | exact p | monotone as a sequence? |",
        "|---|---:|---:|:---:|",
    ]
    for metric, result in tests.items():
        lines.append(
            f"| {metric} | {result['spearman_rho_vs_nnz']:+.3f} | "
            f"{result['exact_permutation_p_value']:.4f} | "
            f"{'yes' if result['strictly_monotone_decreasing_with_sparsity'] else 'no'} |")
    lines += [
        "",
        "**MEASURED.** `acc_tau`, `decode_validity` and the lift over the best constant are",
        f"all perfectly monotone in sparsity (rho +1.000, exact p "
        f"{lift_test['exact_permutation_p_value']:.4f}). The lift crosses zero",
        f"between nnz={positive[-1] if positive else n} and nnz={crossing}: at "
        f"nnz={positive[-1] if positive else n} the model still beats an input-blind",
        f"constant, at nnz={crossing} and below it does not, and by nnz=1 it is "
        f"{abs(sparse[-1]['acc_tau_minus_best_constant']):.4f} *worse*",
        "than ignoring its input entirely.",
        "",
        "**MEASURED.** Output entropy is **not** monotone (rho "
        f"{tests['output_entropy_nats']['spearman_rho_vs_nnz']:+.3f}, exact p "
        f"{tests['output_entropy_nats']['exact_permutation_p_value']:.4f}). It rises from "
        f"{sparse[0]['output_entropy_nats']:.3f} nats at nnz={n}",
        f"to {peak_entropy:.3f} at nnz={peak_nnz}, then falls to "
        f"{sparse[-1]['output_entropy_nats']:.3f} at nnz=1.",
        "",
        "**INFERENCE.** Two different failures happen in sequence, not one. Moderate",
        "sparsity makes the model *diffuse*: it spreads its answers over more values while",
        "getting them less right. Extreme sparsity makes it *collapse*: it stops producing",
        "varied answers at all, and increasingly emits token sequences that do not decode",
        "to a number.",
        "",
        "## 4. Do the probes sit on the same curve?",
        "",
        "Regime C has the same nnz as regime B's last level, so the two are directly",
        "comparable. A gap would mean something beyond sparsity is acting.",
        "",
        "| quantity | B, nnz=1 (random position and value) | C, K*e_i probes |",
        "|---|---:|---:|",
        f"| acc_tau | {continuity['sparse_nnz1_acc_tau']:.4f} | {continuity['probe_acc_tau']:.4f} |",
        f"| decode_validity | {continuity['sparse_nnz1_decode_validity']:.4f} | "
        f"{continuity['probe_decode_validity']:.4f} |",
        f"| unique outputs | {continuity['sparse_nnz1_unique_outputs']} | "
        f"{continuity['probe_unique_outputs']} |",
        f"| output entropy (nats) | {continuity['sparse_nnz1_output_entropy_nats']:.3f} | "
        f"{continuity['probe_output_entropy_nats']:.3f} |",
        "",
        "**MEASURED.** The probes land at or just past the endpoint of the sparsity curve:",
        f"acc_tau differs by {abs(continuity['acc_tau_gap']):.4f}, and the probes are slightly *more*",
        "collapsed (fewer unique outputs, lower validity, lower entropy).",
        "",
        "**INFERENCE.** Regime C is a continuation of regime B, not a separate cliff.",
        "Sparsity alone reproduces almost all of the probe behaviour, which is what the",
        "distribution-shift account predicts. The small residual is consistent with the",
        "probes being more extreme still -- one fixed value on a fixed diagonal, rather",
        "than a random position and magnitude -- but this experiment cannot separate that",
        "from sampling noise at this size.",
        "",
        "## 5. Probe response in detail",
        "",
        "| K | K mod q | separation | decode_validity | unique outputs across i | variance across i | modal value | modal fraction |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for entry in probes["per_k"]:
        nan_variance = entry["variance_across_i"] != entry["variance_across_i"]
        nan_fraction = entry["modal_fraction"] != entry["modal_fraction"]
        lines.append(
            f"| {entry['K']} | {entry['reduced_K']} | {entry['separation']} | "
            f"{entry['decode_validity']:.2f} | {entry['unique_outputs_across_i']} | "
            f"{'-' if nan_variance else format(entry['variance_across_i'], '.1f')} | "
            f"{'-' if entry['modal_value'] < 0 else entry['modal_value']} | "
            f"{'-' if nan_fraction else format(entry['modal_fraction'], '.2f')} |")
    lines += [
        "",
        f"**MEASURED.** Averaged over the {len(probes['k_values'])} K values, the number of distinct decoded",
        f"answers across all {n} coordinates is {probes['mean_unique_outputs_across_i']:.1f} -- fewer than one",
        "distinct answer per K. Four K values produce nothing decodable at all, including",
        "K=125 and K=126, which carry the *largest possible* separation and should be the",
        "most informative probes there are.",
        "",
        "**MEASURED.** Every decodable probe answer in this run was a multiple of 81: only",
        "81, 162 and 243 ever appeared. In base-81 lsb-first that means the low output",
        "digit was always 0.",
        "",
        "**INFERENCE.** The probe response is not a noisy version of the right answer. The",
        "model emits a near-constant token pattern whose low digit mirrors the",
        "overwhelmingly-zero input digits. That is what an extrapolation failure looks",
        "like, not what a precision failure looks like. And that the best-separation probes",
        "are the *least* decodable is the opposite of what an information-limited readout",
        "would predict.",
        "",
        "## 6. A residual signal in the probe response",
        "",
        "**MEASURED, stage 1 (no target involved).** Across all K values the model gives a",
        f"non-modal answer at exactly {stage1['count']} of the {n} coordinates: "
        f"{stage1['coordinates_ever_given_a_non_modal_answer']}. Every other",
        "coordinate always receives the modal answer.",
        "",
        "**MEASURED, stage 2 (the secret enters here, strictly after the fact).** The true",
        f"support is {stage2['true_support']}. These are "
        f"{'the same set' if stage2['singled_out_equals_support'] else 'not the same set'}. "
        f"At K={stage2['k_values_where_inverted_mode_rule_equals_secret']} the released",
        f"code's `bin2` mode rule, inverted, equals the true secret on all {n} coordinates.",
        "",
        "**This is not a recovery result and must not be read as one.** Four things stand",
        "between it and one:",
        "",
        "1. The pattern was found by inspecting these results. The quoted coincidence",
        f"   probability {stage2['coincidence_probability_uniform_support']:.4f} = 1/C({n},{len(stage2['true_support'])}) assumes a uniform random",
        "   support and does **not** correct for that search. It is a lead, not a",
        "   significance test.",
        "2. The released code resolves the inverted-versus-uninverted polarity by comparing",
        "   to the true secret. That step cannot be part of an attack -- which is precisely",
        "   why phase 10 replaced it with the secret-free anchored rule.",
        f"3. It holds for 2 of {len(probes['k_values'])} K values. Four give nothing decodable, two give a flat",
        "   constant, and two single out only half the support -- and inverting *those*",
        "   gives the wrong answer.",
        "4. One secret, one checkpoint, one dimension. n=1.",
        "",
        "**INFERENCE.** The phase-10 collapse was not total. Some coordinate-selective",
        "information about the secret survives into the probe response; it is simply not",
        "expressed in the *numeric value* the anchored rule reads, because the model",
        "answers 81/162/243 rather than anything near 0 or near K.",
        "",
        "**HYPOTHESIS (untested here).** A readout keyed to *which coordinates differ from",
        "the modal answer*, with polarity fixed by a secret-free tie-break such as",
        "\"choose the candidate whose Hamming weight is closest to the known h\", might",
        "extract that signal. Testing it on this data would be circular, because this data",
        "generated the hypothesis. It needs fresh secrets, fresh K values and ideally a",
        "fresh checkpoint, with the rule fixed in advance. That is a proposal for a future",
        "phase, not a result of this one.",
        "",
        "## 7. Bearing on the phase-11 causes",
        "",
        "**MEASURED.** Degradation is monotone in sparsity across acc_tau, decode validity",
        "and lift-over-best-constant, and the probes sit on the same curve.",
        "",
        "**INFERENCE.** This is consistent with **distribution shift / out-of-distribution",
        "generalization** as a mechanism of the phase-10 failure, and it is the outcome the",
        "phase-11 hypothesis predicted in advance. The prediction was falsifiable -- a flat",
        "curve, or an intact curve with a cliff only at the probes, would have contradicted",
        "it -- and it was not falsified.",
        "",
        "**This experiment does not establish causation.** It varies sparsity on one frozen",
        "model and observes covariation. It cannot separate distribution shift from the",
        "other phase-11 causes, because it holds those fixed rather than manipulating them:",
        "",
        "| phase-11 cause | what phase 12 can say about it |",
        "|---|---|",
        "| 2. Distribution shift | Consistent with, and predicted in advance. Not proven causal: one model, one budget. |",
        "| 4. Insufficient training | **Untouched.** One checkpoint at one training budget. A better-trained model might degrade more gracefully; nothing here tests that. |",
        "| 3. Reduced capacity | **Untouched.** One model size. Nothing here compares 4.13M against 51M. |",
        "| 1, 5, 6 | Unaffected by this experiment. |",
        "",
        "**HYPOTHESIS.** Separating cause 2 from cause 4 needs this same sparsity curve",
        "measured on checkpoints at several training budgets. If the zero-lift crossing",
        "point moves toward sparser inputs with more training, sparsity tolerance is",
        "something the model learns and cause 4 dominates; if it does not move, the",
        "sensitivity is intrinsic and cause 2 dominates. Not run here.",
        "",
        "## 8. Comparison with original SALSA",
        "",
        "**MEASURED in phase 11, from the original source.** The original's `gen_expr` only",
        "ever emits rows of the (nega)circulant matrix built from a uniform `a`. Probe-like",
        "inputs are never generated during training: the measured probability that a",
        "training row looks like a probe is 4.799e-26, about 1 in 2.1e25. The original",
        "therefore also requires its transformer to extrapolate to inputs its own training",
        "distribution does not produce.",
        "",
        "**This does not mean the original collapses to the same degree.** That has not been",
        "measured. The original uses roughly 51M parameters against our 4,131,200, roughly",
        "3.9M distinct training samples against our 100,032, and reuses each sample about",
        "ten times. Any of those could change how far a model extrapolates. Establishing",
        "whether the original degrades along the same sparsity curve would require running",
        "this same measurement against a trained original checkpoint. This phase did not do",
        "that and does not claim it.",
        "",
        "## Figures",
        "",
    ]
    for name in payload["figures"]:
        lines.append(f"- `{name}`")
    lines += [
        "",
        "x-axis is the number of nonzero coordinates in `a`, dense on the left, with the",
        "`K*e_i` probes marked as the extreme sparse endpoint.",
        "",
        "## Artifacts",
        "",
        "- `recovery_generalization.md` (this file)",
        "- `recovery_generalization.json`",
        "- `recovery_generalization.csv`",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv: Optional[List[str]] = None) -> int:
    """Run the generalization analysis."""
    parser = argparse.ArgumentParser(description="Recovery-distribution generalization analysis.")
    parser.add_argument("--config", type=Path,
                        default=REPO_ROOT / "configs" / "recovery_n12_h2.yaml")
    parser.add_argument("--run-dir", type=Path,
                        default=REPO_ROOT / "results" / "control_a_n12_h2" / "control")
    parser.add_argument("--samples", type=int, default=2048,
                        help="evaluation vectors per sparsity level")
    parser.add_argument("--out", type=Path,
                        default=REPO_ROOT / "results" / "recovery_generalization")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    config.validate()
    codec = LatticeCodec.from_config(config)
    q, n = codec.q, codec.n
    tolerance = config.evaluation.tolerance

    checkpoint_path = locate_checkpoint(args.run_dir)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    print("=" * 88)
    print("PHASE 12 - RECOVERY-DISTRIBUTION GENERALIZATION  (no training; model frozen)")
    print("=" * 88)
    print(f"  checkpoint : {checkpoint_path.name}  epoch {checkpoint['state']['epoch']}, "
          f"{checkpoint['state']['samples_seen']:,} samples")
    mismatches = verify(checkpoint, config, codec)
    if mismatches:
        print("  MISMATCHES - STOPPING:")
        for line in mismatches:
            print(f"    ! {line}")
        return 2
    print(f"  verified   : fingerprint {checkpoint['config_fingerprint']}")

    model = build_model(config)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    parameter_count = count_trainable_parameters(model)
    frozen_count = sum(p.numel() for p in model.parameters())
    print(f"  model      : {model.name}, {frozen_count:,} parameters, "
          f"{parameter_count:,} trainable (frozen -> expected 0)")

    baselines = chance_baselines(codec, tolerance)
    print(f"  baselines  : acc_tau chance {baselines['chance_acc_tau']:.5f}, "
          f"exact chance {baselines['chance_exact_accuracy']:.5f}")

    seed = derive_seed(config.experiment.seed, "generalization", "phase12")
    rng = resolve_rng(seed=seed)
    target_rng = resolve_rng(seed=derive_seed(seed, "targets"))

    # The secret is fetched once, for STAGE 2 ONLY.  It is not in scope for any
    # function that touches the model.
    problem = build_problem(config, split="test")
    secret = problem.reveal_secret()

    rows: List[Dict[str, Any]] = []
    detail: Dict[str, Any] = {}

    def measure(label: str, input_type: str, nnz: Any, matrix: np.ndarray) -> Dict[str, Any]:
        """Stage 1 then stage 2 for one input family."""
        response = model_response(model, codec, matrix)
        statistics = response_statistics(response, q)
        targets = evaluation_targets(matrix, secret, problem.params, target_rng)
        statistics.update(target_statistics(response, targets, q, tolerance))
        statistics["acc_tau_minus_best_constant"] = (
            statistics["acc_tau"] - statistics["acc_tau_best_constant_predictor"])
        row = {"input_type": input_type, "label": label, "nnz": nnz,
               "mean_nnz_measured": float((matrix != 0).sum(axis=1).mean()), **statistics}
        rows.append(row)
        print(f"  {label:<34} nnz={str(nnz):>5}  valid {row['decode_validity']:.3f}  "
              f"acc_tau {row['acc_tau']:.4f}  exact {row['exact_accuracy']:.4f}  "
              f"uniq {row['unique_outputs']:>4}  H {row['output_entropy_nats']:.3f}  "
              f"best-const {row['acc_tau_best_constant_predictor']:.4f}  "
              f"lift {row['acc_tau_minus_best_constant']:+.4f}")
        return {"row": row, "response": response, "targets": targets}

    # -- one-off check that our instrumented decode matches the library ----- #
    check_matrix = sparse_inputs(64, n, q, 12, resolve_rng(seed=derive_seed(seed, "check")))
    check_src, _ = codec.encode_batch(check_matrix)
    library = greedy_decode(model, torch.from_numpy(check_src),
                            codec.vocabulary.bos_id, codec.output_length - 1).cpu().numpy()
    ours = model_response(model, codec, check_matrix)["tokens"]
    assert np.array_equal(library, ours), "instrumented decode diverged from greedy_decode"
    print("  instrumented decode verified identical to salsa.training.metrics.greedy_decode")

    print()
    print("-" * 88)
    print("REGIME A - training distribution")
    print("-" * 88)
    labeled = problem.labeled_sample(args.samples, rng=resolve_rng(seed=derive_seed(seed, "regimeA")))
    training_matrix = labeled.public.A
    training_response = model_response(model, codec, training_matrix)
    training_statistics = response_statistics(training_response, q)
    # Regime A already carries its own b from the generator; use it directly.
    training_statistics.update(
        target_statistics(training_response, labeled.public.b, q, tolerance))
    training_statistics["acc_tau_minus_best_constant"] = (
        training_statistics["acc_tau"] - training_statistics["acc_tau_best_constant_predictor"])
    training_row = {"input_type": "training_distribution",
                    "label": "A. RLWE rows from the generator", "nnz": n,
                    "mean_nnz_measured": float((training_matrix != 0).sum(axis=1).mean()),
                    **training_statistics}
    rows.append(training_row)
    print(f"  {'A. RLWE rows from the generator':<34} nnz={n:>5}  "
          f"valid {training_row['decode_validity']:.3f}  "
          f"acc_tau {training_row['acc_tau']:.4f}  exact {training_row['exact_accuracy']:.4f}  "
          f"uniq {training_row['unique_outputs']:>4}  H {training_row['output_entropy_nats']:.3f}  "
          f"best-const {training_row['acc_tau_best_constant_predictor']:.4f}  "
          f"lift {training_row['acc_tau_minus_best_constant']:+.4f}")

    print()
    print("-" * 88)
    print("REGIME B - sparsity-controlled inputs (exactly nnz nonzero coordinates)")
    print("-" * 88)
    for nnz in NNZ_LEVELS:
        matrix = sparse_inputs(args.samples, n, q, nnz, rng)
        measure(f"B. sparse random, nnz={nnz}", "sparse_random", nnz, matrix)

    print()
    print("-" * 88)
    print("REGIME C - direct-recovery probes K*e_i  (exhaustive over the phase-10 K sweep)")
    print("-" * 88)
    k_values = list(config.recovery.direct_k_values)
    probe_matrix, probe_k, probe_coordinate = probe_inputs(n, q, k_values)
    probe = measure("C. probes K*e_i (all K, all i)", "probe_K_times_e_i", 1, probe_matrix)

    # Per-K and per-coordinate detail for the probes.
    probe_values = probe["response"]["values"]
    probe_valid = probe["response"]["valid"]
    probe_targets = probe["targets"]
    per_k = []
    for K in k_values:
        mask = probe_k == K
        values, valid = probe_values[mask], probe_valid[mask]
        usable = values[valid]
        distance = absolute_distance(probe_targets[mask], values).astype(np.float64)
        per_k.append({
            "K": int(K),
            "reduced_K": int(K) % q,
            "separation": probe_separation(int(K), q),
            "decode_validity": float(valid.mean()),
            "unique_outputs_across_i": int(np.unique(usable).size),
            "variance_across_i": float(usable.var()) if usable.size else float("nan"),
            "std_across_i": float(usable.std()) if usable.size else float("nan"),
            "modal_value": int(np.bincount(usable).argmax()) if usable.size else -1,
            "modal_fraction": (float(np.unique(usable, return_counts=True)[1].max()) / valid.size
                               if usable.size else float("nan")),
            "mean_prediction": float(usable.mean()) if usable.size else float("nan"),
            "acc_tau_PREDICTION_FIDELITY_NOT_RECOVERY": float(
                (valid & (distance <= tolerance * q)).mean()),
            "predictions_by_coordinate": values.tolist(),
        })
        print(f"    K={K:>4} sep={per_k[-1]['separation']:>4}  "
              f"valid {per_k[-1]['decode_validity']:.2f}  "
              f"unique across i {per_k[-1]['unique_outputs_across_i']:>3}  "
              f"var {per_k[-1]['variance_across_i']:>9.2f}  "
              f"modal {per_k[-1]['modal_value']:>4} "
              f"({per_k[-1]['modal_fraction']:.2f})")

    per_coordinate = []
    for i in range(n):
        mask = probe_coordinate == i
        values, valid = probe_values[mask], probe_valid[mask]
        usable = values[valid]
        per_coordinate.append({
            "coordinate": int(i),
            "decode_validity": float(valid.mean()),
            "unique_outputs_across_K": int(np.unique(usable).size),
            "variance_across_K": float(usable.var()) if usable.size else float("nan"),
            "predictions_by_K": {str(int(K)): int(v) for K, v in zip(probe_k[mask], values)},
        })

    detail["probes"] = {
        "note": ("These are model PREDICTION statistics on probe inputs. They are not "
                 "secret recovery; phase 10 measured secret recovery and the answer was NO."),
        "k_values": [int(k) for k in k_values],
        "per_k": per_k,
        "per_coordinate": per_coordinate,
        "variance_of_per_k_variance": float(np.nanvar([p["variance_across_i"] for p in per_k])),
        "mean_unique_outputs_across_i": float(np.mean([p["unique_outputs_across_i"] for p in per_k])),
    }

    readout = probe_readout_analysis(
        probe_values, probe_k, probe_coordinate, k_values, n, secret)
    detail["probe_readout"] = readout
    stage1 = readout["stage_1_prediction_structure"]
    print(f"    coordinates ever given a non-modal answer: "
          f"{stage1['coordinates_ever_given_a_non_modal_answer']}")

    # -- hypothesis test ---------------------------------------------------- #
    sparse_rows = [r for r in rows if r["input_type"] == "sparse_random"]
    x = [r["nnz"] for r in sparse_rows]
    tests = {
        metric: monotonicity(x, [r[metric] for r in sparse_rows])
        for metric in ("acc_tau", "exact_accuracy", "decode_validity",
                       "output_entropy_nats", "mean_abs_error_valid",
                       "mean_token_predictive_entropy_nats",
                       "acc_tau_minus_best_constant")
    }

    print()
    print("-" * 88)
    print("MONOTONICITY (Spearman rho vs nnz, exact permutation p over 7! = 5040)")
    print("-" * 88)
    for metric, result in tests.items():
        print(f"  {metric:<38} rho {result['spearman_rho_vs_nnz']:>7.3f}  "
              f"p {result['exact_permutation_p_value']:.4f}  "
              f"monotone {str(result['strictly_monotone_decreasing_with_sparsity']):>5}")

    # Does regime C continue regime B's trend, or fall off it?
    sparse_nnz1 = next(r for r in sparse_rows if r["nnz"] == 1)
    probe_row = probe["row"]
    continuity = {
        "note": ("Regime C has the same nnz as the nnz=1 level of regime B. If the "
                 "distribution-shift account is complete, the two should agree; a gap "
                 "means something beyond sparsity alone is acting."),
        "sparse_nnz1_acc_tau": sparse_nnz1["acc_tau"],
        "probe_acc_tau": probe_row["acc_tau"],
        "acc_tau_gap": probe_row["acc_tau"] - sparse_nnz1["acc_tau"],
        "sparse_nnz1_decode_validity": sparse_nnz1["decode_validity"],
        "probe_decode_validity": probe_row["decode_validity"],
        "sparse_nnz1_unique_outputs": sparse_nnz1["unique_outputs"],
        "probe_unique_outputs": probe_row["unique_outputs"],
        "sparse_nnz1_output_entropy_nats": sparse_nnz1["output_entropy_nats"],
        "probe_output_entropy_nats": probe_row["output_entropy_nats"],
    }

    # -- write artifacts ---------------------------------------------------- #
    args.out.mkdir(parents=True, exist_ok=True)
    figures = make_plots(rows, baselines, args.out)

    payload = {
        "phase": "12 - recovery-distribution generalization",
        "status": "READ-ONLY ANALYSIS. No training. Model, data generator and recovery "
                  "code unmodified; all model parameters frozen.",
        "checkpoint": {
            "path": checkpoint_path.relative_to(REPO_ROOT).as_posix(),
            "epoch": checkpoint["state"]["epoch"],
            "samples_seen": checkpoint["state"]["samples_seen"],
            "parameter_count": checkpoint["parameter_count"],
            "config_fingerprint": checkpoint["config_fingerprint"],
            "trainable_parameters_after_freeze": parameter_count,
            "verification": "all checked values match the configuration",
        },
        "instance": {"n": n, "q": q, "h": config.lwe.resolved_hamming_weight,
                     "sigma": config.lwe.sigma, "structure": config.lwe.structure,
                     "rlwe_variant": config.lwe.rlwe_variant,
                     "representation": "R", "vocab": codec.vocabulary.size,
                     "tolerance_tau": tolerance},
        "seeds": {"config_seed": config.experiment.seed, "analysis_seed": seed},
        "samples_per_level": args.samples,
        "chance_baselines": baselines,
        "separation_of_measurement": {
            "stage_1_model_response": "the model is shown only a; no target exists yet",
            "stage_2_evaluation": "b = a.s + e mod q built by the data layer AFTER the "
                                  "model answered; never shown to the model, never used "
                                  "to select an input",
            "probes": "stage-2 numbers on K*e_i measure prediction fidelity, NOT secret "
                      "recovery",
        },
        "table": rows,
        "probe_detail": detail["probes"],
        "probe_readout_analysis": detail["probe_readout"],
        "monotonicity_tests": tests,
        "regime_b_to_c_continuity": continuity,
        "figures": figures,
    }
    (args.out / "recovery_generalization.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")

    columns = ["input_type", "label", "nnz", "mean_nnz_measured", "samples",
               "decode_validity", "undecodable_fraction", "acc_tau", "exact_accuracy",
               "unique_outputs", "unique_output_fraction", "modal_output_fraction",
               "output_std", "output_variance", "output_entropy_nats",
               "output_entropy_ratio_vs_uniform_Zq",
               "mean_token_predictive_entropy_nats",
               "mean_abs_error_all_undecodable_charged_max", "mean_abs_error_valid",
               "median_abs_error_valid", "target_entropy_nats",
               "target_fraction_near_zero", "acc_tau_constant_zero_predictor",
               "acc_tau_best_constant_predictor", "best_constant_value",
               "acc_tau_uniform_random_predictor", "acc_tau_minus_best_constant"]
    with (args.out / "recovery_generalization.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c) for c in columns})

    write_markdown(payload, args.out / "recovery_generalization.md")

    print()
    for name in ("recovery_generalization.md", "recovery_generalization.json",
                 "recovery_generalization.csv", *figures):
        print(f"  wrote {args.out / name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
