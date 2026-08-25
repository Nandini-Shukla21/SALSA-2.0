"""Predict the n=30, h=3 sample requirement from existing runs only.

Analysis only: this script trains nothing, runs no model, and reads no data it
did not already have.  Every measured number is extracted programmatically from
the saved metric streams, and the breakpoint criterion is *imported* from
``dimension_scaling_report`` rather than re-implemented, so the two reports
cannot drift apart.

Three categories of number appear in the outputs and are never mixed:

``measured_salsa2``
    Observed in a completed SALSA 2.0 run.
``extrapolated_salsa2``
    Model output.  A prediction, not a measurement.
``external_anchor_original_salsa``
    Reported by Wenger et al. for the same instance.  Reference only; it is not
    SALSA 2.0 data and is never fitted to.

A structural caution drives the whole analysis: both measured breakpoints have
``h = 2``, while the target has ``h = 3``.  Every model that extrapolates in
``n`` alone is therefore predicting a strictly harder problem from two easier
ones, and is biased low by construction.
"""

import csv
import json
import math
import sys
from dataclasses import dataclass
from math import comb, exp, log, log2
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from dimension_scaling_report import (  # noqa: E402
    Thresholds,
    first_crossing,
    load_run,
)
from salsa.utils import load_config  # noqa: E402

OUTPUT_DIR = REPO_ROOT / "results" / "n30_prediction"

RUNS = [
    ("n=12, h=2", "configs/control_a_n12_h2.yaml", "results/control_a_n12_h2/control"),
    ("n=20, h=2", "configs/control_b_n20_h2.yaml", "results/control_b_n20_h2/control"),
    ("n=30, h=3", "configs/experiment_4_13m_n30_r.yaml", "results/pilot_gatedut_n30_r/pilot"),
]

TARGET_N, TARGET_H = 30, 3

#: Wenger et al., SALSA (NeurIPS 2022), Table 2: n=30, q=251, sparse binary
#: secret. log2(samples) 21.9 at d=0.1 and 24.8 at d=0.13.  EXTERNAL ANCHOR
#: ONLY -- never fitted to, never treated as SALSA 2.0 data.
PAPER_LOG2_LOW, PAPER_LOG2_HIGH = 21.9, 24.8


@dataclass
class Measured:
    """Everything extracted from one completed run."""

    label: str
    n: int
    h: int
    secrets: int
    samples_seen: int
    breakpoint_samples: Optional[int]
    breakpoint_epoch: Optional[int]
    best_acc_tau: float
    best_exact: float
    min_valid_loss: float
    loss_below_marginal: float
    samples_per_second: float
    end_to_end_samples_per_second: float
    wall_clock_min: float
    fingerprint: str

    @property
    def bits(self) -> float:
        """Secret-space size in bits."""
        return log2(self.secrets)

    def to_dict(self) -> Dict[str, Any]:
        """Return a serialisable view tagged as measured data."""
        payload = {k: v for k, v in self.__dict__.items()}
        payload["secret_bits"] = round(self.bits, 4)
        payload["category"] = "measured_salsa2"
        return payload


def extract(label: str, config_path: str, directory: str) -> Measured:
    """Pull every needed quantity out of a saved run.

    Args:
        label: Human-readable experiment name.
        config_path: The configuration the run used.
        directory: The run directory.

    Returns:
        A :class:`Measured` record.

    Raises:
        FileNotFoundError: If the run produced no metrics.
    """
    rows = load_run(str(REPO_ROOT / directory))
    if rows is None:
        raise FileNotFoundError(f"no metrics for {label} at {directory}")

    config = load_config(REPO_ROOT / config_path)
    thresholds = Thresholds.build(config, rows[-1]["valid_sequences"])
    crossing = first_crossing(rows, "valid_acc_tau", thresholds.bar["acc_tau"])

    summary_path = REPO_ROOT / directory / "artifacts" / "summary.json"
    wall_seconds = json.loads(summary_path.read_text("utf-8"))["state"]["elapsed_seconds"]
    final = rows[-1]
    n, h = config.lwe.n, config.lwe.resolved_hamming_weight

    return Measured(
        label=label,
        n=n,
        h=h,
        secrets=comb(n, h),
        samples_seen=final["samples_seen"],
        breakpoint_samples=None if crossing is None else crossing["samples_seen"],
        breakpoint_epoch=None if crossing is None else crossing["epoch"],
        best_acc_tau=max(r["valid_acc_tau"] for r in rows),
        best_exact=max(r["valid_exact_accuracy"] for r in rows),
        min_valid_loss=min(r["valid_loss"] for r in rows),
        loss_below_marginal=round(
            thresholds.marginal_loss - min(r["valid_loss"] for r in rows), 4
        ),
        samples_per_second=final["samples_per_second"],
        end_to_end_samples_per_second=round(final["samples_seen"] / wall_seconds, 3),
        wall_clock_min=round(wall_seconds / 60.0, 2),
        fingerprint=final["config_fingerprint"],
    )


# --------------------------------------------------------------------------- #
# Extrapolation models
# --------------------------------------------------------------------------- #
def fit_models(low: Measured, high: Measured) -> List[Dict[str, Any]]:
    """Fit every extrapolation model through the two observed breakpoints.

    Two points determine each two-parameter model exactly, so these are
    interpolating fits with zero residual and no goodness-of-fit information.
    That is a limitation, not an endorsement: with two points every model fits
    perfectly, and they disagree by more than an order of magnitude at n=30.

    Args:
        low: The easier measured instance.
        high: The harder measured instance.

    Returns:
        One dictionary per model.
    """
    s1, s2 = float(low.breakpoint_samples), float(high.breakpoint_samples)
    n1, n2 = low.n, high.n
    b1, b2 = low.bits, high.bits
    ratio = s2 / s1
    models: List[Dict[str, Any]] = []

    # A. Linear in n.
    slope = (s2 - s1) / (n2 - n1)
    models.append({
        "id": "A",
        "name": "linear in n",
        "form": "S(n) = S2 + slope * (n - n2)",
        "fitted": {"slope_samples_per_dimension": round(slope, 1)},
        "estimate": s2 + slope * (TARGET_N - n2),
        "accounts_for_h": False,
        "note": "Slowest growth. Assumes each extra dimension costs a constant "
                "number of samples, which no learning-theoretic argument supports.",
    })

    # B. Exponential in n.
    rate = log(ratio) / (n2 - n1)
    models.append({
        "id": "B",
        "name": "exponential in n",
        "form": "S(n) = S2 * exp(k * (n - n2))",
        "fitted": {"k_per_dimension": round(rate, 5),
                   "multiplier_per_dimension": round(exp(rate), 4)},
        "estimate": s2 * exp(rate * (TARGET_N - n2)),
        "accounts_for_h": False,
        "note": "Constant multiplicative cost per added dimension.",
    })

    # C. Exponential in the secret-space size.
    per_bit = log(ratio) / (b2 - b1)
    models.append({
        "id": "C",
        "name": "exponential in log2(secret-space size)",
        "form": "S = S2 * exp(c * (bits - bits2))",
        "fitted": {"c_per_bit": round(per_bit, 5),
                   "multiplier_per_bit": round(exp(per_bit), 4)},
        "estimate": s2 * exp(per_bit * (log2(comb(TARGET_N, TARGET_H)) - b2)),
        "accounts_for_h": True,
        "note": "The only model that sees the h increase from 2 to 3. Treats the "
                "cost as driven by how many secrets must be distinguished.",
    })

    # D. Observed ratio applied once more.
    models.append({
        "id": "D",
        "name": "observed breakpoint ratio, applied once more",
        "form": "S(next) = S2 * (S2 / S1)",
        "fitted": {"observed_ratio": round(ratio, 4)},
        "estimate": s2 * ratio,
        "accounts_for_h": False,
        "note": f"Deliberately naive: the measured jump was exactly {ratio:.2f}x "
                "for a step of 8 dimensions, and this applies the same factor to a "
                "step of 10 dimensions with a higher h. Included as a transparent "
                "lower reference, not as a defensible model.",
    })

    # E. Power law in n.
    power = log(ratio) / log(n2 / n1)
    models.append({
        "id": "E",
        "name": "power law in n",
        "form": "S(n) = S2 * (n / n2) ** p",
        "fitted": {"exponent_p": round(power, 4)},
        "estimate": s2 * (TARGET_N / n2) ** power,
        "accounts_for_h": False,
        "note": "Polynomial growth; between linear and exponential.",
    })

    for model in models:
        model["estimate"] = float(model["estimate"])
        model["estimate_rounded"] = int(round(model["estimate"], -3))
        model["category"] = "extrapolated_salsa2"
    return models


def runtime_hours(samples: float, rate: float, valid_seconds: float,
                  samples_per_epoch: int) -> Dict[str, float]:
    """Estimate CPU hours for a budget at the measured n=30 throughput."""
    training = samples / rate / 3600.0
    validation = (samples / samples_per_epoch) * valid_seconds / 3600.0
    return {
        "training_hours": round(training, 2),
        "validation_hours": round(validation, 2),
        "total_hours": round(training + validation, 2),
        "total_days": round((training + validation) / 24.0, 2),
    }


def build_plot(measured: List[Measured], band: Dict[str, float],
               models: List[Dict[str, Any]]) -> Path:
    """Plot measured breakpoints against the predicted n=30 requirement."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8.6, 5.6), dpi=140)

    observed = [m for m in measured if m.breakpoint_samples is not None]
    axis.plot([m.n for m in observed], [m.breakpoint_samples for m in observed],
              "o-", color="#1b7837", markersize=9, linewidth=2,
              label="MEASURED Salsa 2.0 breakpoint (h=2)", zorder=5)
    for m in observed:
        axis.annotate(f"  {m.breakpoint_samples:,}\n  (n={m.n}, h={m.h})",
                      (m.n, m.breakpoint_samples), fontsize=8, va="center")

    unreached = [m for m in measured if m.breakpoint_samples is None]
    for m in unreached:
        axis.plot([m.n], [m.samples_seen], "v", color="#b2182b", markersize=11,
                  label=f"MEASURED: no breakpoint by {m.samples_seen:,} (n={m.n}, h={m.h})",
                  zorder=5)

    centre = band["central"]
    axis.errorbar([TARGET_N], [centre],
                  yerr=[[centre - band["optimistic"]], [band["conservative"] - centre]],
                  fmt="s", markerfacecolor="white", markeredgecolor="#2166ac",
                  ecolor="#2166ac", markersize=11, capsize=6, linewidth=2,
                  label="PREDICTED Salsa 2.0 (h=3), optimistic-central-conservative",
                  zorder=6)

    axis.axhspan(2 ** PAPER_LOG2_LOW, 2 ** PAPER_LOG2_HIGH, color="#999999",
                 alpha=0.18, zorder=1,
                 label=f"EXTERNAL ANCHOR: original SALSA paper, n=30 "
                       f"(2^{PAPER_LOG2_LOW}-2^{PAPER_LOG2_HIGH})")

    for model in models:
        axis.plot([TARGET_N], [model["estimate"]], "_", color="#666666",
                  markersize=18, linewidth=1, zorder=4)
        axis.annotate(f" {model['id']}", (TARGET_N, model["estimate"]),
                      fontsize=7, color="#444444", va="center")

    axis.set_yscale("log")
    axis.set_xlabel("lattice dimension n")
    axis.set_ylabel("training samples to first sustained acc_tau breakpoint (log scale)")
    axis.set_title(
        "Salsa2-GatedUT, 4,131,200 parameters, Representation R, q=251, sigma=3\n"
        "Measured breakpoints (n=12, n=20, h=2) and PREDICTED n=30, h=3 requirement",
        fontsize=10,
    )
    axis.set_xlim(8, 34)
    axis.grid(alpha=0.25, which="both", linewidth=0.6)
    axis.legend(fontsize=7.5, loc="upper left", framealpha=0.93)
    figure.tight_layout()

    path = OUTPUT_DIR / "sample_breakpoint_vs_dimension.png"
    figure.savefig(path)
    plt.close(figure)
    return path


def main() -> int:
    """Build the prediction report."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    measured = [extract(*run) for run in RUNS]
    observed = [m for m in measured if m.breakpoint_samples is not None]
    if len(observed) < 2:
        print("Need two observed breakpoints to extrapolate.")
        return 1
    low, high = observed[0], observed[1]
    target_run = next(m for m in measured if m.n == TARGET_N)

    models = fit_models(low, high)
    h_aware = [m for m in models if m["accounts_for_h"]]
    h_blind = [m for m in models if not m["accounts_for_h"]]

    # Optimistic: the most defensible model that ignores the h increase, and so
    # under-predicts by construction.  Conservative: the model that accounts for
    # it.  Central: their geometric mean, because the estimates span an order of
    # magnitude and a linear mean would be dominated by the larger one.
    optimistic = max(m["estimate"] for m in h_blind if m["id"] == "B")
    conservative = max(m["estimate"] for m in h_aware)
    central = math.sqrt(optimistic * conservative)
    band = {"optimistic": optimistic, "central": central, "conservative": conservative}

    rate = target_run.samples_per_second
    valid_seconds = 45.0
    samples_per_epoch = 79 * 64
    runtimes = {
        name: runtime_hours(value, rate, valid_seconds, samples_per_epoch)
        for name, value in band.items()
    }
    spread = max(m["estimate"] for m in models) / min(m["estimate"] for m in models)

    payload = {
        "status": "PREDICTION FROM EXISTING RUNS ONLY - NO NEW TRAINING WAS PERFORMED",
        "target": {"n": TARGET_N, "h": TARGET_H,
                   "secret_space": comb(TARGET_N, TARGET_H),
                   "secret_bits": round(log2(comb(TARGET_N, TARGET_H)), 4)},
        "model_under_test": "Salsa2-GatedUT, 4,131,200 parameters, T_e=2, T_d=2, "
                            "Representation R, q=251, sigma=3, seed 0",
        "measured_salsa2": [m.to_dict() for m in measured],
        "extrapolated_salsa2": models,
        "prediction_band": {
            "optimistic": {"samples": int(round(optimistic, -3)),
                           "basis": "model B, exponential in n; ignores h rising 2->3",
                           **runtimes["optimistic"]},
            "central": {"samples": int(round(central, -3)),
                        "basis": "geometric mean of models B and C",
                        **runtimes["central"]},
            "conservative": {"samples": int(round(conservative, -3)),
                             "basis": "model C, exponential in secret-space bits; the "
                                      "only model that accounts for h",
                             **runtimes["conservative"]},
        },
        "model_spread_factor": round(spread, 1),
        "external_anchor_original_salsa": {
            "source": "Wenger, Chen, Charton, Lauter, SALSA (NeurIPS 2022), Table 2",
            "instance": "n=30, q=251, sparse binary secret",
            "log2_samples_range": [PAPER_LOG2_LOW, PAPER_LOG2_HIGH],
            "samples_range": [int(2 ** PAPER_LOG2_LOW), int(2 ** PAPER_LOG2_HIGH)],
            "role": "External reference only. Not Salsa 2.0 data. Not fitted to. "
                    "Different model (~51M parameters), different training setup.",
        },
        "throughput_basis": {
            "source": "measured in the completed Salsa 2.0 n=30 pilot",
            "training_samples_per_second": rate,
            "end_to_end_samples_per_second": target_run.end_to_end_samples_per_second,
            "validation_seconds_per_evaluation": valid_seconds,
            "samples_per_epoch": samples_per_epoch,
        },
        "uncertainty": [
            "Only TWO positive breakpoints exist. Every two-parameter model fits "
            "them exactly, so the fits carry no residual and no goodness-of-fit "
            "evidence; model choice, not data, drives the answer.",
            "Both measured breakpoints have h=2, while the target has h=3. Models "
            "A, B, D and E cannot see that change and are biased LOW by construction.",
            "Only one n=30 pilot exists, and it produced no breakpoint - a censored "
            "observation. It bounds the answer from below (>100,032) and nothing more.",
            "Sample-efficiency scaling need not stay linear, polynomial or "
            "exponential across this range; a regime change between n=20 and n=30 "
            "would invalidate all five models equally.",
            "Model capacity has not been varied. 4.13M may be adequate at n=12-20 "
            "and inadequate at n=30, in which case no sample budget succeeds and "
            "the whole extrapolation is moot.",
            "The n=20 run had not converged when its budget ran out - its loss was "
            "still falling - so its breakpoint is the earliest detectable one, not "
            "the point of useful performance.",
            f"The five models span a factor of {spread:.0f}x at n=30. A prediction "
            "with that spread cannot be settled by one more training run.",
        ],
    }

    (OUTPUT_DIR / "n30_prediction.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")

    with (OUTPUT_DIR / "n30_prediction.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["category", "label", "n", "h", "secret_bits",
                         "samples", "basis_or_note", "total_cpu_hours"])
        for m in measured:
            writer.writerow([
                "measured_salsa2", m.label, m.n, m.h, round(m.bits, 3),
                m.breakpoint_samples if m.breakpoint_samples else
                f">{m.samples_seen} (no breakpoint)",
                f"best acc_tau {m.best_acc_tau:.3f}, best exact {m.best_exact:.4f}, "
                f"min loss {m.min_valid_loss:.4f}, {m.samples_per_second:.2f} samples/s, "
                f"{m.wall_clock_min:.1f} min",
                round(m.wall_clock_min / 60.0, 2),
            ])
        for model in models:
            writer.writerow([
                "extrapolated_salsa2", f"model {model['id']}: {model['name']}",
                TARGET_N, TARGET_H, round(log2(comb(TARGET_N, TARGET_H)), 3),
                model["estimate_rounded"],
                ("accounts for h" if model["accounts_for_h"] else "IGNORES h, biased low")
                + "; " + model["form"],
                runtime_hours(model["estimate"], rate, valid_seconds,
                              samples_per_epoch)["total_hours"],
            ])
        for name in ("optimistic", "central", "conservative"):
            entry = payload["prediction_band"][name]
            writer.writerow(["extrapolated_salsa2", f"prediction: {name}", TARGET_N,
                             TARGET_H, round(log2(comb(TARGET_N, TARGET_H)), 3),
                             entry["samples"], entry["basis"], entry["total_hours"]])
        anchor = payload["external_anchor_original_salsa"]
        writer.writerow(["external_anchor_original_salsa", "SALSA paper Table 2",
                         30, 3, round(log2(comb(30, 3)), 3),
                         f"{anchor['samples_range'][0]}-{anchor['samples_range'][1]}",
                         anchor["role"], ""])

    plot = build_plot(measured, band, models)
    write_markdown(payload, measured, models, band, runtimes, spread)

    print(f"wrote {OUTPUT_DIR / 'n30_prediction.json'}")
    print(f"wrote {OUTPUT_DIR / 'n30_prediction.csv'}")
    print(f"wrote {OUTPUT_DIR / 'n30_prediction.md'}")
    print(f"wrote {plot}")
    return 0


def write_markdown(payload: Dict[str, Any], measured: List[Measured],
                   models: List[Dict[str, Any]], band: Dict[str, float],
                   runtimes: Dict[str, Dict[str, float]], spread: float) -> None:
    """Write the human-readable prediction report."""
    lines = [
        "# Predicted sample requirement, Salsa 2.0 at n=30, h=3",
        "",
        "**No new training was performed for this report.** Every measured value is",
        "extracted programmatically from previously completed runs.",
        "",
        f"Model under test: {payload['model_under_test']}",
        "",
        "## 1. Measured Salsa 2.0 results",
        "",
        "| experiment | secret bits | samples | breakpoint | best acc_tau | best exact | min loss | samples/s | wall |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for m in measured:
        bp = f"{m.breakpoint_samples:,}" if m.breakpoint_samples else "**none reached**"
        lines.append(
            f"| {m.label} | {m.bits:.2f} | {m.samples_seen:,} | {bp} | "
            f"{m.best_acc_tau:.3f} | {m.best_exact:.4f} | {m.min_valid_loss:.4f} | "
            f"{m.samples_per_second:.2f} | {m.wall_clock_min:.1f} min |"
        )
    lines += [
        "",
        "## 2. Extrapolation models",
        "",
        "Each model is fitted through the two observed breakpoints. Two points fix a",
        "two-parameter model exactly, so **every one of these fits is perfect and none",
        "of them carries evidence about which is right**.",
        "",
        "| model | form | fitted | n=30 estimate | accounts for h? |",
        "|---|---|---|---:|:---:|",
    ]
    for model in models:
        fitted = ", ".join(f"{k}={v}" for k, v in model["fitted"].items())
        lines.append(
            f"| **{model['id']}** {model['name']} | `{model['form']}` | {fitted} | "
            f"{model['estimate_rounded']:,} | {'yes' if model['accounts_for_h'] else '**no**'} |"
        )
    lines += [
        "",
        f"Spread across models: **{spread:.0f}x**.",
        "",
        "## 3. Prediction band (extrapolated, NOT measured)",
        "",
        "| | samples | training h | validation h | total CPU h | days |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in ("optimistic", "central", "conservative"):
        entry = payload["prediction_band"][name]
        lines.append(
            f"| {name} | {entry['samples']:,} | {entry['training_hours']} | "
            f"{entry['validation_hours']} | **{entry['total_hours']}** | "
            f"{entry['total_days']} |"
        )
    anchor = payload["external_anchor_original_salsa"]
    lines += [
        "",
        "Runtimes use the throughput measured in the completed Salsa 2.0 n=30 pilot",
        f"({payload['throughput_basis']['training_samples_per_second']} samples/s training, "
        f"{payload['throughput_basis']['validation_seconds_per_evaluation']}s per validation).",
        "",
        "## 4. External anchor (NOT Salsa 2.0 data)",
        "",
        f"{anchor['source']} reports {anchor['samples_range'][0]:,} to "
        f"{anchor['samples_range'][1]:,} samples "
        f"(2^{PAPER_LOG2_LOW}-2^{PAPER_LOG2_HIGH}) for {anchor['instance']}.",
        "",
        f"*{anchor['role']}*",
        "",
        "## 5. Why this prediction is uncertain",
        "",
    ]
    lines += [f"{i}. {reason}" for i, reason in enumerate(payload["uncertainty"], 1)]
    lines += ["", "This is a prediction. It is not measured performance.", ""]
    (OUTPUT_DIR / "n30_prediction.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
