"""Scalability forecast plots and table for Modified NACT.

READ-ONLY and cheap.  This script trains nothing, evaluates nothing and loads no
checkpoint.  Every measured value is read from a recorded artifact under
``results/``; every forecast is computed from an explicitly stated model and is
labelled as a forecast in both the figures and the CSV.

Run::

    python scripts/scalability_forecast.py

Outputs, under ``results/scalability_forecast/``:

    plot_a_valid_loss.png          validation loss vs dimension
    plot_b_acc_tau.png             acc_tau vs dimension
    plot_c_exact_recovery.png      secrets recovered, as counts
    plot_d_probes.png              informative-K success and decode validity
    plot_e_compute.png             relative computational burden
    scalability_forecast.csv       every value, with its evidence class
    forecast_table.md              the summary table

Design decisions that matter for honesty:

* The forecast envelopes in plots A and B are **not** confidence intervals and
  are not fitted.  Each is bounded below by the best value this project has
  measured and above by the chance baseline recorded in the same artifacts.
  The envelope therefore spans "nothing degrades" to "complete failure"; its
  width is the finding, not a defect of the drawing.
* Exact recovery is plotted as **counts of independent secrets**, never as a
  rate or a probability curve.  One success is one success.
* Measurements from other architectures (Salsa2-GatedUT at n=30, full NACT at
  n=12) are drawn in a separate, recessive style and never joined to the
  Modified NACT series.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "scalability_forecast"

SRC = {
    "n12": RESULTS / "nact_ablation_F" / "nact_n12_h2_te2_F" / "seed_0" / "artifacts" / "summary.json",
    "n20": RESULTS / "n20_nact_f_pilot" / "nact_n20_h2_te2_F" / "seed_0" / "artifacts" / "summary.json",
    "final": RESULTS / "final_pipeline" / "final_summary.json",
    "robust": RESULTS / "v2_recovery_robustness" / "recovery_robustness.json",
    "v1_n12": RESULTS / "control_a_n12_h2" / "control" / "artifacts" / "summary.json",
    "v1_n20": RESULTS / "control_b_n20_h2" / "control" / "artifacts" / "summary.json",
    "v1_n30": RESULTS / "pilot_gatedut_n30_r" / "pilot" / "artifacts" / "summary.json",
    "ct_n30": RESULTS / "archive" / "pilot_control_n30_r" / "pilot" / "artifacts" / "summary.json",
}

# --------------------------------------------------------------------------- #
# Design tokens (validated categorical palette, light surface)
# --------------------------------------------------------------------------- #

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUTED = "#8a8982"
GRID = "#e4e3de"

OBSERVED = "#2a78d6"   # slot 1 - Modified NACT, measured
FORECAST = "#eb6834"   # slot 2 - forecast / extrapolation
OTHER = "#1baf7a"      # slot 3 - other architectures, measured

DIMENSIONS = [12, 20, 30, 50, 70, 90, 110, 128]
FORECAST_FROM = 20     # forecast region starts beyond the last measured point

ENCODER_DIM = 512      # NactSpec.encoder_dim
MAX_COORDINATES = 128  # NactSpec.max_coordinates - the hard architectural ceiling


def encoder_length(n: int) -> int:
    """Modified NACT encoder sequence length: one token per coordinate, plus bos/eos."""
    return n + 2


# --------------------------------------------------------------------------- #
# Load measurements
# --------------------------------------------------------------------------- #


def load(path: Path) -> Dict[str, Any]:
    """Read one recorded artifact."""
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def gather() -> Dict[str, Any]:
    """Collect every measured quantity the figures need, from artifacts only."""
    n12, n20 = load(SRC["n12"]), load(SRC["n20"])
    final = load(SRC["final"])
    robust = load(SRC["robust"])
    v1_n12, v1_n20 = load(SRC["v1_n12"]), load(SRC["v1_n20"])
    v1_n30, ct_n30 = load(SRC["v1_n30"]), load(SRC["ct_n30"])

    chance = n12["chance_baselines"]
    return {
        "chance_loss": chance["marginal_loss_nats"],
        "chance_acc_tau": chance["chance_acc_tau"],
        "nact": {
            12: {
                "loss": n12["final_validation"]["valid_loss"],
                "acc_tau": n12["final_validation"]["valid_acc_tau"],
                "samples_per_second": n12["throughput"]["samples_per_second"],
                "peak_rss_mb": n12["cpu_memory_mb"]["peak_rss_mb"],
                "samples": n12["state"]["samples_seen"],
            },
            20: {
                "loss": n20["final_validation"]["valid_loss"],
                "acc_tau": n20["final_validation"]["valid_acc_tau"],
                "samples_per_second": n20["throughput"]["samples_per_second"],
                "peak_rss_mb": n20["cpu_memory_mb"]["peak_rss_mb"],
                "samples": n20["state"]["samples_seen"],
            },
        },
        "recovery": final["recovery_results"],
        "verification": final["verification_results"],
        # Multi-secret robustness used the FULL NACT checkpoints (4,241,288),
        # not Modified NACT.  Kept separate for exactly that reason.
        "full_nact_multi_secret": {
            "n": 12,
            "secrets": robust["summary"]["secrets_attacked"],
            "exact": robust["summary"]["exact_recoveries"],
            "mean_successful_k": robust["summary"]["mean_successful_K"],
            "informative_k": robust["summary"]["informative_K_count"],
            "decode_validity": robust["summary"]["mean_probe_decode_validity"],
        },
        "v1": {
            12: {"loss": v1_n12["final_validation"]["valid_loss"],
                 "acc_tau": v1_n12["final_validation"]["valid_acc_tau"],
                 "samples_per_second": v1_n12["throughput"]["samples_per_second"]},
            20: {"loss": v1_n20["final_validation"]["valid_loss"],
                 "acc_tau": v1_n20["final_validation"]["valid_acc_tau"],
                 "samples_per_second": v1_n20["throughput"]["samples_per_second"]},
            30: {"loss": v1_n30["final_validation"]["valid_loss"],
                 "acc_tau": v1_n30["final_validation"]["valid_acc_tau"],
                 "samples_per_second": v1_n30["throughput"]["samples_per_second"],
                 "params": v1_n30["parameter_count"]},
        },
        "ct_n30": {
            "loss": ct_n30["final_validation"]["valid_loss"],
            "acc_tau": ct_n30["final_validation"]["valid_acc_tau"],
            "params": ct_n30["parameter_count"],
        },
    }


# --------------------------------------------------------------------------- #
# Forecast models - each one stated, none fitted to more points than it has
# --------------------------------------------------------------------------- #


def empirical_cost_model(data: Dict[str, Any]):
    """Straight line through the two measured Modified NACT throughput points.

    Returns ``(intercept_ms, slope_ms_per_token)`` for ``ms/sample = a + b * L``.

    Two points determine a two-parameter model exactly, so this fit has no
    residual and carries no evidence about its own correctness.  It is used
    because it is the only model the measured data can support at all, and it
    includes the large dimension-INDEPENDENT overhead (decoder, optimiser, data
    pipeline, Python) that a pure FLOP count omits.
    """
    ms12 = 1000.0 / data["nact"][12]["samples_per_second"]
    ms20 = 1000.0 / data["nact"][20]["samples_per_second"]
    slope = (ms20 - ms12) / (encoder_length(20) - encoder_length(12))
    intercept = ms12 - slope * encoder_length(12)
    return intercept, slope


def theoretical_encoder_cost(n: int) -> float:
    """Encoder FLOP proxy: ``L * d^2`` (feed-forward and projections) + ``L^2 * d`` (attention).

    Excluded on purpose: the decoder, cross-attention, the optimiser step, data
    generation, validation passes, memory bandwidth and interpreter overhead.
    Those are the terms that make the measured curve differ from this one, and
    at the measured dimensions they are roughly half of wall-clock.
    """
    length = encoder_length(n)
    return length * ENCODER_DIM ** 2 + length ** 2 * ENCODER_DIM


# --------------------------------------------------------------------------- #
# Shared figure furniture
# --------------------------------------------------------------------------- #


def new_axes(title: str, subtitle: str, ylabel: str, figsize=(9.0, 5.4)):
    """Create a figure styled once, so every plot reads as one system."""
    fig, ax = plt.subplots(figsize=figsize, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    fig.suptitle(title, x=0.012, ha="left", fontsize=14.5, color=INK,
                 fontweight="semibold", y=1.045)
    ax.set_title(subtitle, loc="left", fontsize=9.6, color=INK_2, pad=12)
    ax.set_xlabel("lattice dimension  n", fontsize=10, color=INK_2)
    ax.set_ylabel(ylabel, fontsize=10, color=INK_2)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9.2, length=0)
    ax.set_xticks(DIMENSIONS)
    ax.set_xlim(6, 136)
    return fig, ax


def mark_forecast_region(ax, label_y: float) -> None:
    """Shade and label the region in which nothing has been measured."""
    ax.axvspan(FORECAST_FROM + 0.001, 136, color=FORECAST, alpha=0.040, zorder=0)
    ax.axvline(FORECAST_FROM, color=INK_MUTED, linewidth=1.0,
               linestyle=(0, (4, 3)), zorder=1)
    ax.text(21.5, label_y, "no Modified NACT model has been trained beyond here",
            fontsize=8.6, color=INK_MUTED, style="italic", va="center")


def mark_ceiling(ax) -> None:
    """Draw the architectural coordinate-embedding ceiling."""
    ax.axvline(MAX_COORDINATES, color=INK_MUTED, linewidth=1.0,
               linestyle=(0, (1, 2.2)), zorder=1)
    ax.text(MAX_COORDINATES - 1.6, ax.get_ylim()[1], "max_coordinates = 128 ",
            fontsize=8.2, color=INK_MUTED, rotation=90, ha="right", va="top")


def save(fig, name: str) -> Path:
    """Write one figure."""
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    fig.savefig(path, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# Plot A - validation loss
# --------------------------------------------------------------------------- #


def plot_a(data: Dict[str, Any]) -> Path:
    """Validation loss: two measured points, one envelope, no fitted trend."""
    chance = data["chance_loss"]
    best = min(data["nact"][12]["loss"], data["nact"][20]["loss"])

    fig, ax = new_axes(
        "Validation loss vs lattice dimension",
        "Measured Modified NACT (n=12, 20) against the envelope of possible outcomes beyond it.\n"
        "The envelope is bounded by two MEASURED constants - it is not a fitted trend and not a confidence interval.",
        "validation loss  (nats / token)")

    ax.fill_between([FORECAST_FROM, 136], [best, best], [chance, chance],
                    color=FORECAST, alpha=0.13, zorder=2,
                    hatch="///", edgecolor=FORECAST, linewidth=0.0)
    ax.axhline(chance, color=INK_MUTED, linewidth=1.4, linestyle=(0, (5, 3)), zorder=3)
    ax.text(50, chance + 0.032, "chance: marginals-only baseline  1.86498",
            fontsize=8.8, color=INK_2, ha="left")

    # Other architectures, measured at n=30 - never joined to the NACT series.
    ax.scatter([30, 30], [data["v1"][30]["loss"], data["ct_n30"]["loss"]],
               s=86, facecolor="none", edgecolor=OTHER, linewidth=2.0,
               marker="s", zorder=5)
    ax.annotate("GatedUT 4.13M and CompactTransformer 4.25M\n"
                "at n=30, h=3, same 100,032-sample budget:\n"
                "both AT chance  (1.8426, 1.8431)",
                xy=(30, data["v1"][30]["loss"]), xytext=(42, 1.42),
                fontsize=8.8, color=INK_2,
                arrowprops=dict(arrowstyle="-", color=OTHER, linewidth=1.2))

    xs = [12, 20]
    ys = [data["nact"][12]["loss"], data["nact"][20]["loss"]]
    ax.plot(xs, ys, color=OBSERVED, linewidth=2.0, zorder=6)
    ax.scatter(xs, ys, s=74, color=OBSERVED, zorder=7, marker="o",
               edgecolor=SURFACE, linewidth=2.0)
    for x, y, off, ha, va in ((xs[0], ys[0], (0, 13), "center", "bottom"),
                              (xs[1], ys[1], (11, 0), "left", "center")):
        ax.annotate(f"{y:.4f}", xy=(x, y), xytext=off,
                    textcoords="offset points", ha=ha, va=va,
                    fontsize=9.4, color=INK, fontweight="semibold")

    ax.set_ylim(0.80, 2.02)
    mark_forecast_region(ax, 0.86)
    mark_ceiling(ax)

    ax.legend(handles=[
        Line2D([], [], color=OBSERVED, marker="o", linewidth=2.0, markersize=8,
               label="Modified NACT, 4,238,208 params - MEASURED"),
        Patch(facecolor=FORECAST, alpha=0.13, hatch="///", edgecolor=FORECAST,
              label="FORECAST envelope: unchanged (floor) to chance (ceiling)"),
        Line2D([], [], color=OTHER, marker="s", linewidth=0, markersize=9,
               markerfacecolor="none", markeredgewidth=2.0,
               label="other architectures at n=30, h=3 - MEASURED"),
    ], loc="upper left", frameon=False, fontsize=9.0, labelcolor=INK_2,
        bbox_to_anchor=(0.0, -0.14), ncol=1)
    return save(fig, "plot_a_valid_loss.png")


# --------------------------------------------------------------------------- #
# Plot B - acc_tau
# --------------------------------------------------------------------------- #


def plot_b(data: Dict[str, Any]) -> Path:
    """acc_tau, with the same envelope construction as plot A."""
    chance = data["chance_acc_tau"]
    best = max(data["nact"][12]["acc_tau"], data["nact"][20]["acc_tau"])

    fig, ax = new_axes(
        "acc_tau vs lattice dimension",
        "acc_tau = fraction of predictions within tau*q = 25.1 of the true b.  tau = 0.1, q = 251.\n"
        "Envelope bounded below by the chance baseline and above by the best measured value. Not a fitted trend.",
        "acc_tau")

    ax.fill_between([FORECAST_FROM, 136], [chance, chance], [best, best],
                    color=FORECAST, alpha=0.13, zorder=2,
                    hatch="///", edgecolor=FORECAST, linewidth=0.0)
    ax.axhline(chance, color=INK_MUTED, linewidth=1.4, linestyle=(0, (5, 3)), zorder=3)
    ax.text(50, chance + 0.030, "chance  0.19287", fontsize=8.8, color=INK_2, ha="left")

    ax.scatter([30, 30], [data["v1"][30]["acc_tau"], data["ct_n30"]["acc_tau"]],
               s=86, facecolor="none", edgecolor=OTHER, linewidth=2.0,
               marker="s", zorder=5)
    ax.annotate("other architectures at n=30, h=3:\n0.1343 and 0.1479 - BELOW chance",
                xy=(30, data["ct_n30"]["acc_tau"]), xytext=(42, 0.44),
                fontsize=8.8, color=INK_2,
                arrowprops=dict(arrowstyle="-", color=OTHER, linewidth=1.2))

    xs = [12, 20]
    ys = [data["nact"][12]["acc_tau"], data["nact"][20]["acc_tau"]]
    ax.plot(xs, ys, color=OBSERVED, linewidth=2.0, zorder=6)
    ax.scatter(xs, ys, s=74, color=OBSERVED, zorder=7, marker="o",
               edgecolor=SURFACE, linewidth=2.0)
    for x, y, off, ha, va in ((xs[0], ys[0], (0, 13), "center", "bottom"),
                              (xs[1], ys[1], (11, 0), "left", "center")):
        ax.annotate(f"{y:.4f}", xy=(x, y), xytext=off,
                    textcoords="offset points", ha=ha, va=va,
                    fontsize=9.4, color=INK, fontweight="semibold")

    ax.set_ylim(0.0, 1.10)
    mark_forecast_region(ax, 0.075)
    mark_ceiling(ax)

    ax.legend(handles=[
        Line2D([], [], color=OBSERVED, marker="o", linewidth=2.0, markersize=8,
               label="Modified NACT - MEASURED"),
        Patch(facecolor=FORECAST, alpha=0.13, hatch="///", edgecolor=FORECAST,
              label="FORECAST envelope: chance to best-measured"),
        Line2D([], [], color=OTHER, marker="s", linewidth=0, markersize=9,
               markerfacecolor="none", markeredgewidth=2.0,
               label="other architectures at n=30, h=3 - MEASURED"),
    ], loc="upper left", frameon=False, fontsize=9.0, labelcolor=INK_2,
        bbox_to_anchor=(0.0, -0.14), ncol=1)
    return save(fig, "plot_b_acc_tau.png")


# --------------------------------------------------------------------------- #
# Plot C - exact secret recovery, as counts
# --------------------------------------------------------------------------- #


def plot_c(data: Dict[str, Any]) -> Path:
    """Counts of independent secrets recovered.  Never a rate, never a curve."""
    multi = data["full_nact_multi_secret"]

    fig, ax = new_axes(
        "Exact secret recovery - counts of independent secrets",
        "Each marker is a COUNT of distinct secrets attacked, not a success rate.\n"
        "One success is one success: no recovery-probability curve is drawn, because none can be estimated from these counts.",
        "independent secrets recovered exactly")

    ax.scatter([12, 20], [1, 1], s=190, color=OBSERVED, marker="o",
               zorder=7, edgecolor=SURFACE, linewidth=2.0)
    for x, off, ha, va in ((12, (0, -22), "center", "center"), (20, (14, 0), "left", "center")):
        ax.annotate("1 / 1", xy=(x, 1), xytext=off, textcoords="offset points",
                    ha=ha, va=va, fontsize=10.5, color=INK, fontweight="semibold")

    ax.scatter([12], [multi["exact"]], s=190, facecolor="none", edgecolor=OTHER,
               linewidth=2.4, marker="D", zorder=6)
    ax.annotate(f"{multi['exact']} / {multi['secrets']}", xy=(12, multi["exact"]),
                xytext=(0, 15), textcoords="offset points", ha="center",
                fontsize=10.5, color=INK, fontweight="semibold")
    ax.annotate("full NACT (4,241,288 params) - a DIFFERENT model.\n"
                "Three training seeds, three distinct secrets, all recovered.\n"
                "Modified NACT itself has one secret per dimension.",
                xy=(12.9, multi["exact"]), xytext=(27, 2.72),
                fontsize=8.8, color=INK_2,
                arrowprops=dict(arrowstyle="-", color=OTHER, linewidth=1.2))

    ax.axvspan(FORECAST_FROM + 0.001, 136, color=INK_MUTED, alpha=0.065, zorder=1)
    ax.text(74, 1.62, "NOT RELIABLY ESTIMABLE", fontsize=13.0, color=INK_2,
            ha="center", fontweight="semibold")
    ax.text(74, 1.40,
            "Two single-seed successes cannot be turned into a rate,\n"
            "and no forecast curve would be anything but invented.",
            fontsize=9.0, color=INK_MUTED, ha="center")

    ax.set_ylim(0, 3.5)
    ax.set_yticks([0, 1, 2, 3])
    ax.axvline(FORECAST_FROM, color=INK_MUTED, linewidth=1.0,
               linestyle=(0, (4, 3)), zorder=2)
    mark_ceiling(ax)

    ax.legend(handles=[
        Line2D([], [], color=OBSERVED, marker="o", linewidth=0, markersize=11,
               label="Modified NACT - MEASURED (verified by residual test)"),
        Line2D([], [], color=OTHER, marker="D", linewidth=0, markersize=10,
               markerfacecolor="none", markeredgewidth=2.2,
               label="full NACT multi-secret control - MEASURED"),
        Patch(facecolor=INK_MUTED, alpha=0.065, label="no evidence of any kind"),
    ], loc="upper left", frameon=False, fontsize=9.0, labelcolor=INK_2,
        bbox_to_anchor=(0.0, -0.14), ncol=1)
    return save(fig, "plot_c_exact_recovery.png")


# --------------------------------------------------------------------------- #
# Plot D - probe success and decode validity (small multiples, never twin axes)
# --------------------------------------------------------------------------- #


def plot_d(data: Dict[str, Any]) -> Path:
    """Two stacked panels: K-probe success, then decode validity."""
    rec = data["recovery"]
    multi = data["full_nact_multi_secret"]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(9.0, 6.8), facecolor=SURFACE, sharex=True,
        gridspec_kw={"height_ratios": [1.0, 0.78], "hspace": 0.18})
    fig.suptitle("Probe success and decode validity", x=0.012, ha="left",
                 fontsize=14.5, color=INK, fontweight="semibold", y=1.02)
    ax1.set_title(
        "10 multipliers K are probed; K=1 and K=250 are NEGATIVE CONTROLS (ring separation 1) and are\n"
        "excluded by the secret-free min_separation guard, leaving 8 informative multipliers.",
        loc="left", fontsize=9.4, color=INK_2, pad=10)

    for ax in (ax1, ax2):
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=INK_2, labelsize=9.2, length=0)
        ax.set_xlim(6, 136)
        ax.axvspan(FORECAST_FROM + 0.001, 136, color=INK_MUTED, alpha=0.065, zorder=1)
        ax.axvline(FORECAST_FROM, color=INK_MUTED, linewidth=1.0,
                   linestyle=(0, (4, 3)), zorder=2)

    ax1.set_ylabel("informative multipliers\nthat recovered the secret", fontsize=9.6, color=INK_2)
    ax1.axhline(8, color=INK_MUTED, linewidth=1.2, linestyle=(0, (5, 3)), zorder=3)
    ax1.text(52, 8.22, "8 informative multipliers available", fontsize=8.6,
             color=INK_2, ha="left")
    successes = [rec["n12"]["successful_k"], rec["n20"]["successful_k"]]
    ax1.scatter([12, 20], successes, s=150, color=OBSERVED, marker="o",
                zorder=7, edgecolor=SURFACE, linewidth=2.0)
    for (x, y), off, ha, va in zip(zip((12, 20), successes),
                                   ((0, 15), (14, 0)), ("center", "left"),
                                   ("bottom", "center")):
        ax1.annotate(f"{y} / 8", xy=(x, y), xytext=off,
                     textcoords="offset points", ha=ha, va=va,
                     fontsize=10.0, color=INK, fontweight="semibold")
    ax1.scatter([12], [multi["mean_successful_k"]], s=150, facecolor="none",
                edgecolor=OTHER, linewidth=2.2, marker="D", zorder=6)
    ax1.annotate(f"full NACT, mean over 3 secrets: {multi['mean_successful_k']:.2f} / 8",
                 xy=(13.2, multi["mean_successful_k"] - 0.1), xytext=(30, 5.3),
                 fontsize=8.8, color=INK_2,
                 arrowprops=dict(arrowstyle="-", color=OTHER, linewidth=1.2))
    ax1.set_ylim(0, 10.2)
    ax1.set_yticks([0, 2, 4, 6, 8])
    ax1.text(74, 4.4, "NOT RELIABLY ESTIMABLE", fontsize=12.0, color=INK_2,
             ha="center", fontweight="semibold")

    ax2.set_ylabel("probe decode validity", fontsize=9.6, color=INK_2)
    ax2.set_xlabel("lattice dimension  n", fontsize=10, color=INK_2)
    validity = [rec["n12"]["probe_decode_validity"], rec["n20"]["probe_decode_validity"]]
    ax2.scatter([12, 20], validity, s=150, color=OBSERVED, marker="o",
                zorder=7, edgecolor=SURFACE, linewidth=2.0)
    for (x, y), off, ha, va in zip(zip((12, 20), validity),
                                   ((0, 15), (14, 0)), ("center", "left"),
                                   ("bottom", "center")):
        ax2.annotate(f"{y:.3f}", xy=(x, y), xytext=off,
                     textcoords="offset points", ha=ha, va=va,
                     fontsize=10.0, color=INK, fontweight="semibold")
    ax2.axhline(0.417, color=OTHER, linewidth=1.4, linestyle=(0, (2, 2)), zorder=3)
    ax2.text(52, 0.455, "V1 GatedUT on the same probes at n=12:  0.417",
             fontsize=8.6, color=INK_2, ha="left")
    ax2.set_ylim(0.0, 1.22)
    ax2.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax2.set_xticks(DIMENSIONS)
    ax2.text(74, 0.62, "NOT RELIABLY ESTIMABLE", fontsize=12.0, color=INK_2,
             ha="center", fontweight="semibold")

    ax2.legend(handles=[
        Line2D([], [], color=OBSERVED, marker="o", linewidth=0, markersize=10,
               label="Modified NACT - MEASURED"),
        Line2D([], [], color=OTHER, marker="D", linewidth=0, markersize=9,
               markerfacecolor="none", markeredgewidth=2.2,
               label="full NACT, 3-secret mean - MEASURED"),
        Patch(facecolor=INK_MUTED, alpha=0.065, label="no evidence of any kind"),
    ], loc="upper left", frameon=False, fontsize=9.0, labelcolor=INK_2,
        bbox_to_anchor=(0.0, -0.24), ncol=2)
    return save(fig, "plot_d_probes.png")


# --------------------------------------------------------------------------- #
# Plot E - relative computational burden
# --------------------------------------------------------------------------- #


def plot_e(data: Dict[str, Any]) -> Path:
    """Relative cost per 100k samples, n=20 as the reference point."""
    intercept, slope = empirical_cost_model(data)
    ms20 = intercept + slope * encoder_length(20)

    empirical = [(intercept + slope * encoder_length(n)) / ms20 for n in DIMENSIONS]
    theoretical = [theoretical_encoder_cost(n) / theoretical_encoder_cost(20)
                   for n in DIMENSIONS]

    measured_x = [12, 20]
    measured_y = [(1000.0 / data["nact"][n]["samples_per_second"]) / ms20
                  for n in measured_x]

    # V1 GatedUT, measured at three dimensions, normalised to its OWN n=20 run.
    v1_x = [12, 20, 30]
    v1_ref = 1000.0 / data["v1"][20]["samples_per_second"]
    v1_y = [(1000.0 / data["v1"][n]["samples_per_second"]) / v1_ref for n in v1_x]

    fig, ax = new_axes(
        "Relative computational burden per 100,000 training samples",
        "Reference = the measured Modified NACT run at n=20 (87.43 samples/s, 1,144 s of compute).\n"
        "Assumed encoder sequence length L = n+2 (one token per coordinate); encoder width d = 512.",
        "cost relative to the measured n=20 run", figsize=(9.0, 5.8))

    ax.plot(DIMENSIONS, theoretical, color=OTHER, linewidth=2.0,
            linestyle=(0, (1, 2)), zorder=5)
    ax.scatter(DIMENSIONS, theoretical, s=44, color=OTHER, marker="^", zorder=6)
    ax.annotate(f"THEORETICAL encoder FLOP proxy\nL*d^2 + L^2*d   ->  {theoretical[-1]:.2f}x at n=128",
                xy=(110, theoretical[-2]), xytext=(66, 6.2), fontsize=8.8,
                color=INK_2, arrowprops=dict(arrowstyle="-", color=OTHER, linewidth=1.2))

    ax.plot(DIMENSIONS, empirical, color=FORECAST, linewidth=2.0,
            linestyle=(0, (5, 3)), zorder=5)
    ax.scatter(DIMENSIONS[2:], empirical[2:], s=58, facecolor="none",
               edgecolor=FORECAST, linewidth=2.0, marker="o", zorder=6)
    ax.annotate(f"EXTRAPOLATED wall-clock (2-point linear fit)\n->  {empirical[-1]:.2f}x at n=128",
                xy=(118, empirical[-2] + 0.08), xytext=(60, 1.78), fontsize=8.8,
                color=INK_2, arrowprops=dict(arrowstyle="-", color=FORECAST, linewidth=1.2))

    ax.plot(v1_x, v1_y, color=INK_MUTED, linewidth=1.4, zorder=4)
    ax.scatter(v1_x, v1_y, s=42, color=INK_MUTED, marker="x", linewidth=1.8, zorder=5)
    ax.annotate("V1 GatedUT, MEASURED at three dimensions\n"
                "(L = 2n+2, normalised to its own n=20 run):\n"
                "real wall-clock grows FASTER than linear in L",
                xy=(30, v1_y[-1]), xytext=(33, 0.30), fontsize=8.8, color=INK_2,
                arrowprops=dict(arrowstyle="-", color=INK_MUTED, linewidth=1.2))

    ax.scatter(measured_x, measured_y, s=96, color=OBSERVED, marker="o",
               zorder=8, edgecolor=SURFACE, linewidth=2.0)
    for x, y, off, ha in zip(measured_x, measured_y, ((0, 14), (0, 14)), ("center", "center")):
        ax.annotate(f"{y:.2f}x", xy=(x, y), xytext=off,
                    textcoords="offset points", ha=ha,
                    fontsize=9.6, color=INK, fontweight="semibold")

    ax.set_ylim(0, 8.2)
    mark_ceiling(ax)
    ax.axvline(FORECAST_FROM, color=INK_MUTED, linewidth=1.0,
               linestyle=(0, (4, 3)), zorder=1)

    ax.legend(handles=[
        Line2D([], [], color=OBSERVED, marker="o", linewidth=0, markersize=9,
               label="Modified NACT wall-clock - MEASURED"),
        Line2D([], [], color=FORECAST, marker="o", linewidth=2.0, markersize=8,
               linestyle=(0, (5, 3)), markerfacecolor="none", markeredgewidth=2.0,
               label="EXTRAPOLATED wall-clock, linear in L"),
        Line2D([], [], color=OTHER, marker="^", linewidth=2.0, markersize=8,
               linestyle=(0, (1, 2)), label="THEORETICAL encoder FLOP proxy"),
        Line2D([], [], color=INK_MUTED, marker="x", linewidth=1.4, markersize=8,
               label="V1 GatedUT - MEASURED, different model"),
    ], loc="upper left", frameon=False, fontsize=9.0, labelcolor=INK_2,
        bbox_to_anchor=(0.0, -0.14), ncol=2)

    note = ("EXCLUDED from both forecast curves: the number of samples required "
            "(unknown, and the dominant term in total cost), the decoder and "
            "cross-attention, the optimiser step,\ndata generation, validation "
            "passes, memory bandwidth and interpreter overhead. About half of the "
            "measured per-sample time at n=20 is dimension-independent,\nwhich is "
            "why the FLOP proxy rises faster than the wall-clock extrapolation.")
    fig.text(0.012, -0.225, note, fontsize=8.2, color=INK_MUTED, ha="left", va="top")
    return save(fig, "plot_e_compute.png")


# --------------------------------------------------------------------------- #
# CSV and table
# --------------------------------------------------------------------------- #

CONFIDENCE = {12: "high (measured)", 20: "high (measured)", 30: "medium",
              50: "low", 70: "low", 90: "low", 110: "low", 128: "low"}

OUTLOOK = {
    12: "demonstrated", 20: "demonstrated",
    30: "uncertain", 50: "uncertain to unlikely at a 100k-sample budget",
    70: "unlikely at this budget", 90: "unlikely at this budget",
    110: "unlikely at this budget", 128: "unlikely at this budget",
}


def build_rows(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """One row per (dimension, metric), each carrying its evidence class."""
    intercept, slope = empirical_cost_model(data)
    ms20 = intercept + slope * encoder_length(20)
    rows: List[Dict[str, Any]] = []

    def add(n, metric, value, lower, upper, cls, conf, source):
        rows.append({"n": n, "metric": metric, "value": value, "lower": lower,
                     "upper": upper, "evidence_class": cls, "confidence": conf,
                     "source": source})

    for n in (12, 20):
        src = str(SRC["n12" if n == 12 else "n20"].relative_to(ROOT)).replace("\\", "/")
        add(n, "valid_loss", round(data["nact"][n]["loss"], 6), "", "", "observed", "high", src)
        add(n, "acc_tau", round(data["nact"][n]["acc_tau"], 6), "", "", "observed", "high", src)
        add(n, "samples_per_second", round(data["nact"][n]["samples_per_second"], 3),
            "", "", "observed", "high", src)
        add(n, "peak_rss_mb", round(data["nact"][n]["peak_rss_mb"], 1), "", "", "observed", "high", src)
        key = f"n{n}"
        add(n, "exact_recovery_secrets", f"{1}/{1}", "", "", "observed", "high",
            "results/final_pipeline/final_summary.json")
        add(n, "successful_informative_K", f"{data['recovery'][key]['successful_k']}/8",
            "", "", "observed", "high", "results/final_pipeline/final_summary.json")
        add(n, "probe_decode_validity", data["recovery"][key]["probe_decode_validity"],
            "", "", "observed", "high", "results/final_pipeline/final_summary.json")
        add(n, "verification_residual_std",
            round(data["verification"][key]["residual_std"], 4), "", "", "observed",
            "high", "results/final_pipeline/final_summary.json")

    add(12, "exact_recovery_secrets_full_NACT",
        f"{data['full_nact_multi_secret']['exact']}/{data['full_nact_multi_secret']['secrets']}",
        "", "", "observed", "high",
        "results/v2_recovery_robustness/recovery_robustness.json (DIFFERENT model: full NACT, 4,241,288)")

    for n, key, label in ((30, "v1", "GatedUT_4131200_h3"), (30, "ct", "CompactTransformer_4251520_h3")):
        src = data["v1"][30] if key == "v1" else data["ct_n30"]
        add(n, f"valid_loss__{label}", round(src["loss"], 6), "", "", "observed",
            "high", "results/pilot_gatedut_n30_r/ or results/archive/pilot_control_n30_r/")
        add(n, f"acc_tau__{label}", round(src["acc_tau"], 6), "", "", "observed",
            "high", "results/pilot_gatedut_n30_r/ or results/archive/pilot_control_n30_r/")

    for n in DIMENSIONS:
        add(n, "encoder_sequence_length", encoder_length(n), "", "", "theoretical",
            "high", "salsa/models/nact.py - NactFrontEnd, one token per coordinate")
        add(n, "parameter_count", 4238208 if n <= MAX_COORDINATES else "",
            "", "", "theoretical", "high",
            "salsa/models/parameter_count.py:211 - invariant in n up to max_coordinates")
        rel = (intercept + slope * encoder_length(n)) / ms20
        add(n, "relative_compute_per_100k_wallclock", round(rel, 3), "", "",
            "observed" if n in (12, 20) else "extrapolated",
            CONFIDENCE[n], "2-point linear fit to measured throughput; EXCLUDES samples required")
        add(n, "relative_compute_encoder_flop_proxy",
            round(theoretical_encoder_cost(n) / theoretical_encoder_cost(20), 3),
            "", "", "theoretical", "medium", "L*d^2 + L^2*d, d=512, L=n+2")

    for n in (30, 50, 70, 90, 110, 128):
        add(n, "valid_loss", "not reliably estimable",
            round(min(data["nact"][12]["loss"], data["nact"][20]["loss"]), 6),
            round(data["chance_loss"], 6), "extrapolated", CONFIDENCE[n],
            "envelope bounded by measured best and measured chance baseline")
        add(n, "acc_tau", "not reliably estimable",
            round(data["chance_acc_tau"], 6),
            round(max(data["nact"][12]["acc_tau"], data["nact"][20]["acc_tau"]), 6),
            "extrapolated", CONFIDENCE[n],
            "envelope bounded by measured chance baseline and measured best")
        add(n, "exact_recovery_secrets", "not reliably estimable", "", "",
            "not_estimable", CONFIDENCE[n], "no model trained; two single-seed successes give no rate")
        add(n, "successful_informative_K", "not reliably estimable", "", "",
            "not_estimable", CONFIDENCE[n], "no model trained")
        add(n, "probe_decode_validity", "not reliably estimable", "", "",
            "not_estimable", CONFIDENCE[n], "no model trained")
        add(n, "recovery_outlook", OUTLOOK[n], "", "", "extrapolated", CONFIDENCE[n],
            "judgement from the evidence in SCALABILITY_ANALYSIS.md")

    add(30, "samples_required__literature", "3913424 to 29210829", 3913424, 29210829,
        "literature", "n/a - different model",
        "SALSA (NeurIPS 2022) Table 2 as recorded in results/archive/n30_prediction/; "
        "~51M params, sample reuse 10x, negacyclic. NOT a Modified NACT measurement.")
    add(128, "secret_search_space_C_n_2", 8128, "", "", "theoretical", "high",
        "C(n,2) at h=2 - the whole space is enumerable; see SCALABILITY_ANALYSIS.md section A.1")
    return rows


def write_csv(rows: List[Dict[str, Any]]) -> Path:
    """Write the machine-readable table."""
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "scalability_forecast.csv"
    fields = ["n", "metric", "value", "lower", "upper", "evidence_class",
              "confidence", "source"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_table(data: Dict[str, Any]) -> Path:
    """Write the human-readable summary table."""
    intercept, slope = empirical_cost_model(data)
    ms20 = intercept + slope * encoder_length(20)
    lines = [
        "# Modified NACT - scalability summary table",
        "",
        "Generated by `scripts/scalability_forecast.py`. No training, no evaluation,",
        "no checkpoint loaded. Measured values are read from artifacts under `results/`.",
        "",
        "| n | L=n+2 | evidence | valid loss | acc_tau | exact recovery | informative K | "
        "decode validity | compute vs n=20 | confidence |",
        "|---:|---:|---|---|---|---|---|---|---:|---|",
    ]
    for n in DIMENSIONS:
        rel = f"{(intercept + slope * encoder_length(n)) / ms20:.2f}x"
        if n in (12, 20):
            key = f"n{n}"
            lines.append(
                f"| {n} | {encoder_length(n)} | **observed** | {data['nact'][n]['loss']:.4f} | "
                f"{data['nact'][n]['acc_tau']:.4f} | **YES** (1/1 secret) | "
                f"{data['recovery'][key]['successful_k']}/8 | "
                f"{data['recovery'][key]['probe_decode_validity']:.3f} | {rel} | "
                f"{CONFIDENCE[n]} |")
        else:
            extra = " + 2 other-architecture runs at chance (h=3)" if n == 30 else ""
            lines.append(
                f"| {n} | {encoder_length(n)} | forecast{extra} | not estimable "
                f"(envelope {min(data['nact'][12]['loss'], data['nact'][20]['loss']):.4f}"
                f"-{data['chance_loss']:.4f}) | not estimable (envelope "
                f"{data['chance_acc_tau']:.4f}-{max(data['nact'][12]['acc_tau'], data['nact'][20]['acc_tau']):.4f}) | "
                f"{OUTLOOK[n]} | not estimable | not estimable | {rel} | {CONFIDENCE[n]} |")
    lines += [
        "",
        "`compute vs n=20` is per 100,000 training samples and **excludes the number of",
        "samples required**, which is unknown and is the dominant term in total cost.",
        "",
        "At h=2 the secret space is C(n,2) - 66 at n=12, 8,128 at n=128 - so recovery at",
        "any of these dimensions is reachable by exhaustive enumeration plus the project's",
        "own residual verifier, with no model at all. Raising n at fixed h=2 does not",
        "raise cryptographic difficulty.",
        "",
    ]
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "forecast_table.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #


def main() -> None:
    """Build every figure and table."""
    data = gather()
    written = [plot_a(data), plot_b(data), plot_c(data), plot_d(data), plot_e(data),
               write_csv(build_rows(data)), write_table(data)]
    for path in written:
        print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
