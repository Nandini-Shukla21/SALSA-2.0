"""Isolate the Hamming-weight effect and refit the n=30 prediction.

Analysis only: trains nothing, reads only completed runs.  The breakpoint
criterion is imported from ``dimension_scaling_report`` so it cannot drift from
the earlier reports: a crossing counts only if the *next* evaluation is also
above the bar, which rejects the one-epoch spikes a near-constant predictor
produces.

The decisive comparison is ``n=20, h=2`` against ``n=20, h=3``.  Dimension,
sequence length, model, representation, optimiser, schedule, seed and
evaluation protocol are all held fixed, so any difference between them is
attributable to the secret density alone.

With a third breakpoint the two-variable model

    ln S = a + b*n + c*h

becomes determined, which separates dimension cost from density cost for the
first time.  When the h=3 run produces no breakpoint the observation is
*censored*, and it still yields a strict lower bound on the density cost -- that
case is handled explicitly rather than being treated as a missing value.
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

from dimension_scaling_report import Thresholds, first_crossing, load_run  # noqa: E402
from salsa.utils import load_config  # noqa: E402

OUTPUT_DIR = REPO_ROOT / "results" / "hamming_weight_scaling"

EXPERIMENTS = [
    ("n=12, h=2", "configs/control_a_n12_h2.yaml", "results/control_a_n12_h2/control"),
    ("n=20, h=2", "configs/control_b_n20_h2.yaml", "results/control_b_n20_h2/control"),
    ("n=20, h=3", "configs/control_c_n20_h3.yaml", "results/control_c_n20_h3/control"),
    ("n=30, h=3", "configs/experiment_4_13m_n30_r.yaml", "results/pilot_gatedut_n30_r/pilot"),
]

#: Wenger et al., SALSA (NeurIPS 2022), Table 2. EXTERNAL ANCHOR ONLY.
PAPER_LOG2_LOW, PAPER_LOG2_HIGH = 21.9, 24.8


@dataclass
class Run:
    """One completed experiment, with its breakpoint resolved."""

    label: str
    n: int
    h: int
    samples_seen: int
    breakpoint: Optional[int]
    censored: bool
    best_acc_tau: float
    final_acc_tau: float
    best_exact: float
    final_exact: float
    min_valid_loss: float
    final_token_accuracy: float
    final_greedy_token_accuracy: float
    final_decode_failure_rate: float
    mean_distance: float
    median_distance: float
    marginal_loss: float
    acc_tau_bar: float
    exact_bar: float
    token_chance: float
    acc_tau_chance: float
    exact_chance: float
    samples_per_second: float
    wall_clock_min: float
    cpu_rss_mb: float
    parameter_count: int
    fingerprint: str
    curve: Dict[str, List[float]]

    @property
    def secrets(self) -> int:
        """Number of possible secrets."""
        return comb(self.n, self.h)

    @property
    def bits(self) -> float:
        """Secret-space size in bits."""
        return log2(self.secrets)

    @property
    def loss_below_marginal(self) -> float:
        """How far the best loss fell below the marginals-only baseline."""
        return round(self.marginal_loss - self.min_valid_loss, 4)

    def to_dict(self) -> Dict[str, Any]:
        """Return a serialisable view without the full curve."""
        payload = {k: v for k, v in self.__dict__.items() if k != "curve"}
        payload["secrets"] = self.secrets
        payload["secret_bits"] = round(self.bits, 4)
        payload["loss_below_marginal"] = self.loss_below_marginal
        payload["category"] = "measured_salsa2"
        return payload


def extract(label: str, config_path: str, directory: str) -> Optional[Run]:
    """Read one run and resolve its breakpoint against the shared thresholds."""
    rows = load_run(str(REPO_ROOT / directory))
    if rows is None:
        print(f"  ! {label}: no metrics at {directory}")
        return None

    config = load_config(REPO_ROOT / config_path)
    thresholds = Thresholds.build(config, rows[-1]["valid_sequences"])
    crossing = first_crossing(rows, "valid_acc_tau", thresholds.bar["acc_tau"])
    final = rows[-1]

    summary_path = REPO_ROOT / directory / "artifacts" / "summary.json"
    wall_seconds = (
        json.loads(summary_path.read_text("utf-8"))["state"]["elapsed_seconds"]
        if summary_path.is_file()
        else final.get("elapsed_seconds", 0.0)
    )

    return Run(
        label=label,
        n=config.lwe.n,
        h=config.lwe.resolved_hamming_weight,
        samples_seen=final["samples_seen"],
        breakpoint=None if crossing is None else crossing["samples_seen"],
        censored=crossing is None,
        best_acc_tau=max(r["valid_acc_tau"] for r in rows),
        final_acc_tau=final["valid_acc_tau"],
        best_exact=max(r["valid_exact_accuracy"] for r in rows),
        final_exact=final["valid_exact_accuracy"],
        min_valid_loss=min(r["valid_loss"] for r in rows),
        final_token_accuracy=final["valid_token_accuracy"],
        final_greedy_token_accuracy=final["valid_greedy_token_accuracy"],
        final_decode_failure_rate=final["valid_decode_failure_rate"],
        mean_distance=final["valid_mean_distance"],
        median_distance=final["valid_median_distance"],
        marginal_loss=thresholds.marginal_loss,
        acc_tau_bar=round(thresholds.bar["acc_tau"], 6),
        exact_bar=round(thresholds.bar["exact"], 6),
        token_chance=round(thresholds.chance["token"], 6),
        acc_tau_chance=round(thresholds.chance["acc_tau"], 6),
        exact_chance=round(thresholds.chance["exact"], 6),
        samples_per_second=final["samples_per_second"],
        wall_clock_min=round(wall_seconds / 60.0, 2),
        cpu_rss_mb=final["cpu_rss_mb"],
        parameter_count=final["parameter_count"],
        fingerprint=final["config_fingerprint"],
        curve={
            "samples": [r["samples_seen"] for r in rows],
            "valid_loss": [r["valid_loss"] for r in rows],
            "acc_tau": [r["valid_acc_tau"] for r in rows],
            "exact_accuracy": [r["valid_exact_accuracy"] for r in rows],
        },
    )


def analyse_hamming(h2: Run, h3: Run) -> Dict[str, Any]:
    """Quantify the isolated effect of raising h at fixed n."""
    delta_bits = h3.bits - h2.bits
    result: Dict[str, Any] = {
        "dimension": h2.n,
        "h_low": h2.h,
        "h_high": h3.h,
        "secrets_low": h2.secrets,
        "secrets_high": h3.secrets,
        "secret_ratio": round(h3.secrets / h2.secrets, 3),
        "delta_bits": round(delta_bits, 4),
        "h2_breakpoint": h2.breakpoint,
        "h3_breakpoint": h3.breakpoint,
        "h3_censored": h3.censored,
        "h3_budget": h3.samples_seen,
    }

    if h3.breakpoint is not None and h2.breakpoint is not None:
        ratio = h3.breakpoint / h2.breakpoint
        result.update({
            "breakpoint_ratio": round(ratio, 3),
            "cost_per_unit_h": round(ratio, 3),
            "cost_per_bit": round(exp(log(ratio) / delta_bits), 4),
            "interpretation": (
                f"Raising h from {h2.h} to {h3.h} at n={h2.n} multiplied the "
                f"sample requirement by {ratio:.2f}x."
            ),
        })
    else:
        lower = h3.samples_seen / h2.breakpoint
        result.update({
            "breakpoint_ratio": None,
            "breakpoint_ratio_lower_bound": round(lower, 3),
            "cost_per_bit_lower_bound": round(exp(log(lower) / delta_bits), 4),
            "interpretation": (
                f"No sustained breakpoint within {h3.samples_seen:,} samples. The "
                f"cost of raising h from {h2.h} to {h3.h} at n={h2.n} is therefore "
                f"MORE than {lower:.2f}x -- a censored observation giving a strict "
                "lower bound, not a point estimate."
            ),
        })
    return result


def refit_n30(runs: Dict[str, Run], hamming: Dict[str, Any]) -> Dict[str, Any]:
    """Refit the n=30, h=3 requirement using the new measurement.

    With breakpoints at (n=12,h=2), (n=20,h=2) and (n=20,h=3) the model
    ``ln S = a + b*n + c*h`` is exactly determined: ``b`` comes from the two
    h=2 points and ``c`` from the two n=20 points.  This is the first estimate
    that separates the two effects instead of confounding them.
    """
    a12, b20 = runs["n=12, h=2"], runs["n=20, h=2"]
    per_dimension = log(b20.breakpoint / a12.breakpoint) / (b20.n - a12.n)
    base = log(b20.breakpoint) + per_dimension * (30 - b20.n)

    payload: Dict[str, Any] = {
        "model": "ln S = a + b*n + c*h, separating dimension cost from density cost",
        "b_per_dimension": round(per_dimension, 5),
        "multiplier_per_dimension": round(exp(per_dimension), 4),
        "n30_h2_implied": int(round(exp(base), -3)),
        "note": "n30_h2_implied is what n=30 would cost at h=2; the density term "
                "is then applied on top to reach h=3.",
    }

    if hamming.get("breakpoint_ratio") is not None:
        factor = hamming["breakpoint_ratio"]
        payload.update({
            "c_multiplier_per_unit_h": factor,
            "estimate_n30_h3": int(round(exp(base) * factor, -3)),
            "censored": False,
        })
    else:
        factor = hamming["breakpoint_ratio_lower_bound"]
        payload.update({
            "c_multiplier_per_unit_h_lower_bound": factor,
            "estimate_n30_h3_lower_bound": int(round(exp(base) * factor, -3)),
            "censored": True,
            "note_censored": (
                "The density term is only bounded below, so the n=30 estimate is a "
                "LOWER BOUND. The true requirement may be far larger."
            ),
        })
    return payload


def build_plots(runs: List[Run]) -> List[Path]:
    """Plot the h=2 vs h=3 comparison at n=20, with the other runs for context."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    style = {
        "n=12, h=2": ("#1b7837", 1.0, "-"),
        "n=20, h=2": ("#2166ac", 2.4, "-"),
        "n=20, h=3": ("#d95f02", 2.4, "-"),
        "n=30, h=3": ("#b2182b", 1.0, "--"),
    }
    focus = {"n=20, h=2", "n=20, h=3"}
    reference = runs[0]

    panels = [
        ("acc_tau", "SALSA acc_tau  (|b - b_hat| <= 0.1 q)", "01_acc_tau_vs_samples.png",
         [("chance", reference.acc_tau_chance, "--"), ("chance + 3 sd", reference.acc_tau_bar, ":")]),
        ("valid_loss", "Validation loss (nats/token)", "02_valid_loss_vs_samples.png",
         [("marginals-only baseline", reference.marginal_loss, "--")]),
        ("exact_accuracy", "Exact integer accuracy", "03_exact_accuracy_vs_samples.png",
         [("chance", reference.exact_chance, "--"), ("chance + 3 sd", reference.exact_bar, ":")]),
    ]

    written: List[Path] = []
    for key, ylabel, filename, guides in panels:
        figure, axis = plt.subplots(figsize=(8.4, 5.4), dpi=140)
        for run in runs:
            colour, width, dash = style.get(run.label, ("#666666", 1.0, "-"))
            emphasised = run.label in focus
            axis.plot(
                [s / 1000.0 for s in run.curve["samples"]],
                run.curve[key],
                label=f"{run.label}" + ("  <- ISOLATION PAIR" if emphasised else "  (context)"),
                color=colour,
                linewidth=width,
                linestyle=dash,
                alpha=1.0 if emphasised else 0.45,
                marker="o" if emphasised else None,
                markersize=2.4,
                zorder=4 if emphasised else 2,
            )
        for name, value, dash in guides:
            axis.axhline(value, linestyle=dash, linewidth=1.2, color="#555555", label=name)

        for run in runs:
            if run.label in focus and run.breakpoint:
                axis.axvline(run.breakpoint / 1000.0, color=style[run.label][0],
                             linestyle=":", linewidth=1.1, alpha=0.7)

        axis.set_xlabel("training samples seen (thousands)")
        axis.set_ylabel(ylabel)
        axis.set_title(
            "Hamming-weight isolation at n=20 (h=2 vs h=3)\n"
            "Salsa2-GatedUT, 4,131,200 parameters, Representation R, q=251, sigma=3, seed 0",
            fontsize=10,
        )
        axis.grid(alpha=0.25, linewidth=0.6)
        axis.legend(fontsize=8, loc="best", framealpha=0.92)
        figure.tight_layout()
        path = OUTPUT_DIR / filename
        figure.savefig(path)
        plt.close(figure)
        written.append(path)
    return written


def write_markdown(payload: Dict[str, Any], runs: List[Run]) -> None:
    """Write the human-readable comparison report."""
    hamming = payload["hamming_weight_effect"]
    refit = payload["refit_n30_prediction"]
    lines = [
        "# Hamming-weight isolation at n=20",
        "",
        "**Diagnostic experiment. Not a cryptographic result.**",
        "",
        f"Model: {payload['model_under_test']}",
        "",
        "## Dimension / density table",
        "",
        "| experiment | secrets | bits | samples | breakpoint | best acc_tau | best exact | min loss | below marginal |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for run in runs:
        bp = f"{run.breakpoint:,}" if run.breakpoint else f"**none by {run.samples_seen:,}**"
        lines.append(
            f"| {run.label} | {run.secrets:,} | {run.bits:.2f} | {run.samples_seen:,} | {bp} | "
            f"{run.best_acc_tau:.3f} | {run.best_exact:.4f} | {run.min_valid_loss:.4f} | "
            f"{run.loss_below_marginal:+.4f} |"
        )
    lines += [
        "",
        f"Shared thresholds: acc_tau chance {runs[0].acc_tau_chance:.4f} "
        f"(3 sd bar {runs[0].acc_tau_bar:.4f}), exact chance {runs[0].exact_chance:.5f} "
        f"(3 sd bar {runs[0].exact_bar:.5f}), marginals-only loss "
        f"{runs[0].marginal_loss:.4f} nats.",
        "",
        "## Isolated Hamming-weight effect",
        "",
        f"- secrets: {hamming['secrets_low']:,} -> {hamming['secrets_high']:,} "
        f"({hamming['secret_ratio']}x, +{hamming['delta_bits']} bits)",
        f"- {hamming['interpretation']}",
        "",
        "## Refitted n=30, h=3 prediction",
        "",
        f"- {refit['model']}",
        f"- dimension cost: x{refit['multiplier_per_dimension']} per added dimension",
        f"- n=30 at h=2 would imply {refit['n30_h2_implied']:,} samples",
    ]
    if refit["censored"]:
        lines += [
            f"- density cost at least x{refit['c_multiplier_per_unit_h_lower_bound']}",
            f"- **n=30, h=3 lower bound: {refit['estimate_n30_h3_lower_bound']:,} samples**",
            f"- {refit['note_censored']}",
        ]
    else:
        lines += [
            f"- density cost x{refit['c_multiplier_per_unit_h']}",
            f"- **n=30, h=3 estimate: {refit['estimate_n30_h3']:,} samples**",
        ]
    lines += [
        "",
        "## External anchor (NOT Salsa 2.0 data)",
        "",
        f"Wenger et al., SALSA (NeurIPS 2022), Table 2: {int(2**PAPER_LOG2_LOW):,} to "
        f"{int(2**PAPER_LOG2_HIGH):,} samples for n=30, q=251. Different model "
        "(~51M parameters), different setup. Reference only; never fitted to.",
        "",
    ]
    (OUTPUT_DIR / "hamming_weight_scaling.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    """Build the Hamming-weight comparison."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    runs = [r for r in (extract(*e) for e in EXPERIMENTS) if r is not None]
    by_label = {r.label: r for r in runs}
    if "n=20, h=3" not in by_label:
        print("The n=20, h=3 run has produced no metrics yet.")
        return 1

    hamming = analyse_hamming(by_label["n=20, h=2"], by_label["n=20, h=3"])
    refit = refit_n30(by_label, hamming)

    payload = {
        "status": "DIAGNOSTIC HAMMING-WEIGHT ISOLATION - NOT A CRYPTOGRAPHIC RESULT",
        "model_under_test": (
            "Salsa2-GatedUT, 4,131,200 parameters, T_e=2, T_d=2, Representation R, "
            "q=251, sigma=3, RLWE circulant, seed 0"
        ),
        "controlled_comparison": (
            "n=20 h=2 vs n=20 h=3: dimension, sequence length, model, representation, "
            "optimiser, schedule, seed and evaluation protocol all held fixed."
        ),
        "experiments": [r.to_dict() for r in runs],
        "hamming_weight_effect": hamming,
        "refit_n30_prediction": refit,
        "external_anchor_original_salsa": {
            "source": "Wenger, Chen, Charton, Lauter, SALSA (NeurIPS 2022), Table 2",
            "samples_range": [int(2 ** PAPER_LOG2_LOW), int(2 ** PAPER_LOG2_HIGH)],
            "role": "External reference only. Not Salsa 2.0 data. Not fitted to.",
        },
    }
    (OUTPUT_DIR / "hamming_weight_scaling.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")

    fields = ["label", "n", "h", "secrets", "secret_bits", "samples_seen", "breakpoint",
              "censored", "best_acc_tau", "final_acc_tau", "acc_tau_chance", "acc_tau_3sd",
              "best_exact", "final_exact", "exact_chance", "exact_3sd",
              "min_valid_loss", "marginal_loss", "loss_below_marginal",
              "final_token_accuracy", "token_chance", "final_greedy_token_accuracy",
              "final_decode_failure_rate", "mean_distance", "median_distance",
              "samples_per_second", "wall_clock_min", "cpu_rss_mb",
              "parameter_count", "fingerprint"]
    with (OUTPUT_DIR / "hamming_weight_scaling.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for run in runs:
            row = run.to_dict()
            row["breakpoint"] = run.breakpoint if run.breakpoint else ""
            writer.writerow(row)

    plots = build_plots(runs)
    write_markdown(payload, runs)

    print(f"wrote {OUTPUT_DIR / 'hamming_weight_scaling.json'}")
    print(f"wrote {OUTPUT_DIR / 'hamming_weight_scaling.csv'}")
    print(f"wrote {OUTPUT_DIR / 'hamming_weight_scaling.md'}")
    for path in plots:
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
