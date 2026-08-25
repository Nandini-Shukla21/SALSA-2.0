"""Build the dimension-scaling comparison from completed runs.

Reads the metric streams of the n=12, n=20 and n=30 experiments, computes the
learning breakpoints against explicit statistical thresholds, writes a table
(JSON + CSV) and three directly comparable plots into
``results/dimension_scaling/``.

This is analysis only: it trains nothing and changes nothing.

The thresholds matter more than the numbers.  Every metric here has a chance
level fixed by the representation, and ``b`` is uniform over ``Z_q`` regardless
of ``n`` and ``h``, so the *same* baselines apply to all three experiments and
the curves really are comparable -- even though the underlying cryptographic
difficulty is not.
"""

import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from salsa.data import LatticeCodec  # noqa: E402
from salsa.training import chance_baselines  # noqa: E402
from salsa.utils import load_config  # noqa: E402

OUTPUT_DIR = Path("results/dimension_scaling")

#: Completed runs, in increasing order of difficulty.
EXPERIMENTS = [
    ("n=12, h=2", "configs/control_a_n12_h2.yaml", "results/control_a_n12_h2/control"),
    ("n=20, h=2", "configs/control_b_n20_h2.yaml", "results/control_b_n20_h2/control"),
    ("n=30, h=3", "configs/experiment_4_13m_n30_r.yaml", "results/pilot_gatedut_n30_r/pilot"),
]


@dataclass
class Thresholds:
    """Chance levels and the 3-sigma bars a metric must clear to count.

    Attributes:
        valid_samples: Size of the validation set the bars were computed for.
        marginal_loss: Loss of a model that learned only the output marginals.
        uniform_loss: Loss of a model that learned nothing at all.
        chance: Chance level per metric.
        bar: 3-sigma threshold per metric.
    """

    valid_samples: int
    marginal_loss: float
    uniform_loss: float
    chance: Dict[str, float]
    bar: Dict[str, float]

    @classmethod
    def build(cls, config, valid_samples: int) -> "Thresholds":
        """Derive thresholds from a configuration's representation."""
        base = chance_baselines(
            LatticeCodec.from_config(config), config.evaluation.tolerance
        )
        chance = {
            "acc_tau": base["chance_acc_tau"],
            "exact": base["chance_exact_accuracy"],
            "token": base["chance_token_accuracy"],
        }
        bar = {
            name: value + 3.0 * math.sqrt(value * (1.0 - value) / valid_samples)
            for name, value in chance.items()
        }
        return cls(
            valid_samples=valid_samples,
            marginal_loss=base["marginal_loss_nats"],
            uniform_loss=base["uniform_loss_nats"],
            chance=chance,
            bar=bar,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return a serialisable view."""
        return {
            "validation_samples": self.valid_samples,
            "marginal_loss_nats": round(self.marginal_loss, 4),
            "uniform_loss_nats": round(self.uniform_loss, 4),
            "chance": {k: round(v, 6) for k, v in self.chance.items()},
            "three_sigma_bar": {k: round(v, 6) for k, v in self.bar.items()},
        }


def load_run(directory: str) -> Optional[List[Dict[str, Any]]]:
    """Load a run's metric rows, or None when the run never produced any."""
    path = Path(directory) / "metrics.jsonl"
    if not path.is_file():
        return None
    rows = [json.loads(line) for line in path.read_text("utf-8").splitlines() if line.strip()]
    return rows or None


def first_crossing(rows: List[Dict[str, Any]], key: str, bar: float) -> Optional[Dict[str, Any]]:
    """Return the first row whose ``key`` clears ``bar`` and stays clear.

    A single lucky epoch is not a breakpoint.  A crossing counts only if the
    metric is also above the bar at the following evaluation, which rules out
    the one-epoch spikes a near-constant predictor produces.
    """
    for index, row in enumerate(rows):
        if row.get(key, 0.0) <= bar:
            continue
        following = rows[index + 1] if index + 1 < len(rows) else None
        if following is None or following.get(key, 0.0) > bar:
            return row
    return None


def summarise(label: str, config_path: str, directory: str) -> Optional[Dict[str, Any]]:
    """Summarise one run against its thresholds."""
    rows = load_run(directory)
    if rows is None:
        print(f"  ! {label}: no metrics found at {directory}")
        return None

    config = load_config(config_path)
    thresholds = Thresholds.build(config, rows[-1]["valid_sequences"])
    final = rows[-1]
    best_loss = min(r["valid_loss"] for r in rows)
    best_tau = max(r["valid_acc_tau"] for r in rows)
    best_exact = max(r["valid_exact_accuracy"] for r in rows)

    tau_break = first_crossing(rows, "valid_acc_tau", thresholds.bar["acc_tau"])
    exact_break = first_crossing(rows, "valid_exact_accuracy", thresholds.bar["exact"])

    # Wall clock comes from summary.json's end-of-run state, which is the only
    # resume-safe total. The metrics rows carry per-epoch compute time instead.
    summary_path = Path(directory) / "artifacts" / "summary.json"
    completed = summary_path.is_file()
    wall_clock_seconds = final.get("elapsed_seconds", 0.0)
    compute_seconds = sum(r.get("epoch_seconds", 0.0) for r in rows)
    if completed:
        wall_clock_seconds = json.loads(summary_path.read_text("utf-8"))["state"][
            "elapsed_seconds"
        ]
    return {
        "label": label,
        "n": config.lwe.n,
        "h": config.lwe.resolved_hamming_weight,
        "config": config_path,
        "run_dir": directory,
        "completed": completed,
        "config_fingerprint": final["config_fingerprint"],
        "seed": final["seed"],
        "parameter_count": final["parameter_count"],
        "model_name": final["model_name"],
        "samples_seen": final["samples_seen"],
        "tokens_seen": final["tokens_seen"],
        "epochs": final["epoch"] + 1,
        "wall_clock_min": round(wall_clock_seconds / 60.0, 2),
        "training_compute_min": round(compute_seconds / 60.0, 2),
        "mean_samples_per_second_end_to_end": round(
            final["samples_seen"] / wall_clock_seconds, 2
        ) if wall_clock_seconds else 0.0,
        "samples_per_second": final["samples_per_second"],
        "tokens_per_second": final["tokens_per_second"],
        "cpu_rss_mb": final["cpu_rss_mb"],
        "final": {
            "valid_loss": final["valid_loss"],
            "token_accuracy": final["valid_token_accuracy"],
            "greedy_token_accuracy": final["valid_greedy_token_accuracy"],
            "acc_tau": final["valid_acc_tau"],
            "exact_accuracy": final["valid_exact_accuracy"],
            "decode_failure_rate": final["valid_decode_failure_rate"],
        },
        "best": {
            "valid_loss": best_loss,
            "acc_tau": best_tau,
            "exact_accuracy": best_exact,
        },
        "loss_below_marginal": round(thresholds.marginal_loss - best_loss, 4),
        "breakpoints": {
            "acc_tau": None if tau_break is None else {
                "epoch": tau_break["epoch"],
                "samples": tau_break["samples_seen"],
                "value": tau_break["valid_acc_tau"],
            },
            "exact_accuracy": None if exact_break is None else {
                "epoch": exact_break["epoch"],
                "samples": exact_break["samples_seen"],
                "value": exact_break["valid_exact_accuracy"],
            },
        },
        "thresholds": thresholds.to_dict(),
        "curve": {
            "samples": [r["samples_seen"] for r in rows],
            "valid_loss": [r["valid_loss"] for r in rows],
            "acc_tau": [r["valid_acc_tau"] for r in rows],
            "exact_accuracy": [r["valid_exact_accuracy"] for r in rows],
            "token_accuracy": [r["valid_token_accuracy"] for r in rows],
        },
    }


def write_plots(summaries: List[Dict[str, Any]], thresholds: Thresholds) -> List[Path]:
    """Write the three comparison plots with shared axes and formatting."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    colours = {"n=12, h=2": "#1b7837", "n=20, h=2": "#2166ac", "n=30, h=3": "#b2182b"}
    written: List[Path] = []

    panels = [
        ("valid_loss", "Validation loss (nats/token)", "01_valid_loss_vs_samples.png",
         [("marginals-only baseline", thresholds.marginal_loss, "--", "#555555"),
          ("perfect-secret floor", 0.8392, ":", "#999999")]),
        ("acc_tau", "SALSA acc_tau  (|b - b_hat| <= 0.1 q)", "02_acc_tau_vs_samples.png",
         [("chance", thresholds.chance["acc_tau"], "--", "#555555"),
          ("chance + 3 sd", thresholds.bar["acc_tau"], ":", "#999999")]),
        ("exact_accuracy", "Exact integer accuracy", "03_exact_accuracy_vs_samples.png",
         [("chance", thresholds.chance["exact"], "--", "#555555"),
          ("chance + 3 sd", thresholds.bar["exact"], ":", "#999999")]),
    ]

    for key, ylabel, filename, guides in panels:
        figure, axis = plt.subplots(figsize=(8.0, 5.0), dpi=140)
        for summary in summaries:
            curve = summary["curve"]
            axis.plot(
                [s / 1000.0 for s in curve["samples"]],
                curve[key],
                label=f"{summary['label']}  ({summary['samples_seen']:,} samples)",
                color=colours.get(summary["label"], "#444444"),
                linewidth=1.9,
                marker="o",
                markersize=2.6,
            )
        for name, value, style, colour in guides:
            axis.axhline(value, linestyle=style, linewidth=1.2, color=colour, label=name)

        axis.set_xlabel("training samples seen (thousands)")
        axis.set_ylabel(ylabel)
        axis.set_title(
            f"Salsa2-GatedUT, 4,131,200 parameters, Representation R, q=251, sigma=3\n{ylabel}",
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


def main() -> int:
    """Build the report."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summaries = [s for s in (summarise(*e) for e in EXPERIMENTS) if s is not None]
    if not summaries:
        print("No completed runs found.")
        return 1

    thresholds = Thresholds.build(
        load_config(EXPERIMENTS[0][1]), summaries[0]["thresholds"]["validation_samples"]
    )

    payload = {
        "status": "DIAGNOSTIC DIMENSION-SCALING STUDY - NOT A CRYPTOGRAPHIC RESULT",
        "caveat": (
            "n=12 and n=20 are easier instances than n=30 and must not be read as "
            "equivalent cryptographic difficulty. The purpose is to determine "
            "whether the pipeline can learn secret-related structure at all, and "
            "how that ability degrades with dimension."
        ),
        "model": "Salsa2-GatedUT, 4,131,200 trainable parameters, T_e=2, T_d=2",
        "shared_thresholds": thresholds.to_dict(),
        "note_on_thresholds": (
            "b is uniform over Z_q for every (n, h) tested, so the chance levels "
            "and the marginal-loss baseline are identical across all three runs."
        ),
        "experiments": summaries,
    }
    (OUTPUT_DIR / "dimension_scaling.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    fields = [
        "label", "n", "h", "completed", "samples_seen", "valid_loss", "best_valid_loss",
        "loss_below_marginal", "token_accuracy", "token_chance", "acc_tau",
        "best_acc_tau", "acc_tau_chance", "acc_tau_3sd", "exact_accuracy",
        "best_exact_accuracy", "exact_chance", "exact_3sd", "acc_tau_breakpoint_samples",
        "exact_breakpoint_samples", "samples_per_second",
        "end_to_end_samples_per_second", "wall_clock_min", "training_compute_min",
        "cpu_rss_mb", "parameter_count", "config_fingerprint",
    ]
    with (OUTPUT_DIR / "dimension_scaling.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for s in summaries:
            tau_bp = s["breakpoints"]["acc_tau"]
            exact_bp = s["breakpoints"]["exact_accuracy"]
            writer.writerow({
                "label": s["label"], "n": s["n"], "h": s["h"],
                "completed": s["completed"], "samples_seen": s["samples_seen"],
                "valid_loss": s["final"]["valid_loss"],
                "best_valid_loss": round(s["best"]["valid_loss"], 6),
                "loss_below_marginal": s["loss_below_marginal"],
                "token_accuracy": s["final"]["token_accuracy"],
                "token_chance": round(thresholds.chance["token"], 4),
                "acc_tau": s["final"]["acc_tau"],
                "best_acc_tau": round(s["best"]["acc_tau"], 6),
                "acc_tau_chance": round(thresholds.chance["acc_tau"], 4),
                "acc_tau_3sd": round(thresholds.bar["acc_tau"], 4),
                "exact_accuracy": s["final"]["exact_accuracy"],
                "best_exact_accuracy": round(s["best"]["exact_accuracy"], 6),
                "exact_chance": round(thresholds.chance["exact"], 5),
                "exact_3sd": round(thresholds.bar["exact"], 5),
                "acc_tau_breakpoint_samples": "" if tau_bp is None else tau_bp["samples"],
                "exact_breakpoint_samples": "" if exact_bp is None else exact_bp["samples"],
                "samples_per_second": s["samples_per_second"],
                "end_to_end_samples_per_second": s["mean_samples_per_second_end_to_end"],
                "wall_clock_min": s["wall_clock_min"],
                "training_compute_min": s["training_compute_min"],
                "cpu_rss_mb": s["cpu_rss_mb"],
                "parameter_count": s["parameter_count"],
                "config_fingerprint": s["config_fingerprint"],
            })

    plots = write_plots(summaries, thresholds)
    print(f"wrote {OUTPUT_DIR / 'dimension_scaling.json'}")
    print(f"wrote {OUTPUT_DIR / 'dimension_scaling.csv'}")
    for path in plots:
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
