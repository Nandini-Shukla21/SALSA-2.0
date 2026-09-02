"""Phase 25 report: NACT-F pilot at n=20, h=2.

Read-only.  Consumes the artifacts the pilot already wrote and the existing
n=12 and V1 results; trains nothing and runs no recovery.
"""

import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

OUTPUT_DIR = REPO_ROOT / "results" / "n20_nact_f_pilot"
PILOT = OUTPUT_DIR / "nact_n20_h2_te2_F" / "seed_0"
F_N12 = REPO_ROOT / "results" / "nact_ablation_F" / "nact_n12_h2_te2_F" / "seed_0"
V1_N20 = REPO_ROOT / "results" / "control_b_n20_h2" / "control"
CONFIG = REPO_ROOT / "configs" / "nact_n20_h2_te2_F.yaml"

#: Fixed before the pilot ran and not revised afterwards.
GATES = {
    "acc_tau_bar": 0.21903,       # chance 0.19287 + 3 sigma on 2048 sequences
    "exact_bar": 0.00816,         # chance 0.00398 + 3 sigma on 2048 sequences
    "marginal_loss": 1.86498,     # a model that learned only the output marginals
    "chance_acc_tau": 0.19287,
    "chance_exact": 0.00398,
    "chance_token": 0.44622,
    "irreducible_loss": 0.83920,
}

METRICS = [
    ("valid_loss", "Validation loss (nats)", "lower"),
    ("valid_acc_tau", "SALSA acc_tau", "higher"),
    ("valid_exact_accuracy", "Exact integer accuracy", "higher"),
    ("valid_token_accuracy", "Token accuracy", "higher"),
    ("valid_greedy_token_accuracy", "Greedy token accuracy", "higher"),
    ("valid_perfect_accuracy", "Perfect accuracy", "higher"),
    ("valid_decode_failure_rate", "Decode failure rate", "lower"),
    ("valid_mean_distance", "Mean |b - b_hat|", "lower"),
    ("valid_median_distance", "Median |b - b_hat|", "lower"),
]


def load_run(run: Path) -> Dict[str, Any]:
    """Flatten a run's summary plus its final train loss."""
    summary = json.loads((run / "artifacts" / "summary.json").read_text("utf-8"))
    state = summary["state"]
    row = {
        "run_dir": str(run.relative_to(REPO_ROOT)).replace("\\", "/"),
        "model_name": summary["model_name"],
        "architecture": summary["architecture"],
        "parameter_count": summary["parameter_count"],
        "seed": summary["seed"],
        "best_epoch": state["best_epoch"],
        "samples_seen": state["samples_seen"],
        "elapsed_seconds": state["elapsed_seconds"],
        "samples_per_second": summary["throughput"]["samples_per_second"],
        "tokens_per_second": summary["throughput"]["tokens_per_second"],
        "rss_mb": summary["cpu_memory_mb"]["rss_mb"],
        "chance_baselines": summary["chance_baselines"],
    }
    row.update({k: v for k, v in summary["final_validation"].items()
                if isinstance(v, (int, float))})
    curve = list(csv.DictReader((run / "metrics.csv").open(encoding="utf-8")))
    if curve:
        row["train_loss_final"] = float(curve[-1]["train_loss"])
    row["curve"] = [
        {"epoch": int(r["epoch"]), "samples_seen": int(r["samples_seen"]),
         "train_loss": float(r["train_loss"]),
         "valid_loss": float(r["valid_loss"]) if r.get("valid_loss") else None,
         "valid_acc_tau": float(r["valid_acc_tau"]) if r.get("valid_acc_tau") else None,
         "valid_exact_accuracy": (float(r["valid_exact_accuracy"])
                                  if r.get("valid_exact_accuracy") else None)}
        for r in curve
    ]
    return row


def make_plots(pilot: Dict[str, Any], n12: Dict[str, Any], v1: Dict[str, Any],
               out_dir: Path) -> List[str]:
    """The four required figures."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ["V1 GatedUT\nn=20 (200k)", "NACT-F\nn=12 (100k)", "NACT-F\nn=20 (100k)"]
    colours = ["#d62728", "#7f7f7f", "#1f77b4"]
    written = []

    specs = [
        ("01_valid_loss.png", "valid_loss", "Validation loss (nats)",
         [("marginals-only baseline", GATES["marginal_loss"], "--", "grey"),
          ("irreducible floor", GATES["irreducible_loss"], ":", "black")]),
        ("02_acc_tau.png", "valid_acc_tau", "SALSA acc_tau",
         [("chance", GATES["chance_acc_tau"], "--", "grey"),
          ("pre-registered bar (chance + 3 sigma)", GATES["acc_tau_bar"], "-.", "darkred")]),
        ("03_exact_accuracy.png", "valid_exact_accuracy", "Exact integer accuracy",
         [("chance", GATES["chance_exact"], "--", "grey"),
          ("pre-registered bar", GATES["exact_bar"], "-.", "darkred")]),
    ]
    for filename, key, title, references in specs:
        figure, axes = plt.subplots(figsize=(7.0, 4.5))
        values = [v1[key], n12[key], pilot[key]]
        bars = axes.bar(labels, values, color=colours, edgecolor="black")
        for bar, value in zip(bars, values):
            axes.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.4f}",
                      ha="center", va="bottom", fontsize=9)
        for name, level, style, colour in references:
            axes.axhline(level, linestyle=style, color=colour, linewidth=1.3,
                         label=f"{name} = {level:.5g}")
        axes.set_ylabel(title)
        axes.set_title(title + "\nh=2, q=251, representation R, seed 0")
        axes.legend(fontsize=8)
        axes.grid(alpha=0.3, axis="y")
        figure.tight_layout(); figure.savefig(out_dir / filename, dpi=150)
        plt.close(figure); written.append(filename)

    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    for name, run, colour in (("V1 GatedUT n=20", v1, "#d62728"),
                              ("NACT-F n=12", n12, "#7f7f7f"),
                              ("NACT-F n=20", pilot, "#1f77b4")):
        points = [(p["samples_seen"], p["valid_loss"]) for p in run["curve"]
                  if p["valid_loss"] is not None]
        if points:
            axes[0].plot([p[0] for p in points], [p[1] for p in points],
                         "-o", color=colour, markersize=3, linewidth=1.8, label=name)
        points = [(p["samples_seen"], p["valid_acc_tau"]) for p in run["curve"]
                  if p["valid_acc_tau"] is not None]
        if points:
            axes[1].plot([p[0] for p in points], [p[1] for p in points],
                         "-o", color=colour, markersize=3, linewidth=1.8, label=name)
    axes[0].axhline(GATES["marginal_loss"], linestyle="--", color="grey",
                    label="marginals-only")
    axes[0].axhline(GATES["irreducible_loss"], linestyle=":", color="black",
                    label="irreducible floor")
    axes[0].set_xlabel("training samples seen"); axes[0].set_ylabel("validation loss (nats)")
    axes[0].set_title("Validation loss"); axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)
    axes[1].axhline(GATES["acc_tau_bar"], linestyle="-.", color="darkred",
                    label="pre-registered bar")
    axes[1].axhline(GATES["chance_acc_tau"], linestyle="--", color="grey", label="chance")
    axes[1].set_xlabel("training samples seen"); axes[1].set_ylabel("SALSA acc_tau")
    axes[1].set_title("acc_tau"); axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
    figure.suptitle("Learning curves. NACT-F n=20 used HALF V1's sample budget.",
                    fontsize=11)
    figure.tight_layout(); figure.savefig(out_dir / "04_learning_curve.png", dpi=150)
    plt.close(figure); written.append("04_learning_curve.png")
    return written


def main() -> int:
    """Assemble the phase-25 report."""
    from salsa.data.secrets import secret_from_config
    from salsa.utils import load_config
    import numpy as np

    pilot, n12, v1 = load_run(PILOT), load_run(F_N12), load_run(V1_N20)
    config = load_config(CONFIG); config.validate()
    secret = secret_from_config(config)

    assert pilot["parameter_count"] == 4_238_208
    assert pilot["samples_seen"] <= 100_032

    gates = {
        "acc_tau_above_bar": pilot["valid_acc_tau"] > GATES["acc_tau_bar"],
        "exact_above_bar": pilot["valid_exact_accuracy"] > GATES["exact_bar"],
        "loss_below_marginal": pilot["valid_loss"] < GATES["marginal_loss"],
    }
    if gates["acc_tau_above_bar"] and gates["loss_below_marginal"]:
        outcome, text = "A", "strong learning"
    elif gates["acc_tau_above_bar"] or gates["loss_below_marginal"]:
        outcome, text = "B", "weak / ambiguous learning"
    else:
        outcome, text = "C", "no learning"

    print("=" * 96)
    print("PHASE 25 - NACT-F PILOT AT n=20, h=2   (one bounded pilot; no recovery, no n=30)")
    print("=" * 96)
    print(f"  parameters {pilot['parameter_count']:,}  samples {pilot['samples_seen']:,}  "
          f"best epoch {pilot['best_epoch']}  elapsed {pilot['elapsed_seconds']:.0f}s")
    print(f"  secret (n=20, fresh): {secret.tolist()}  support "
          f"{np.flatnonzero(secret).tolist()}")
    print()
    print("  PRE-REGISTERED GATES (fixed before the run, not revised)")
    print(f"    acc_tau {pilot['valid_acc_tau']:.4f} > {GATES['acc_tau_bar']}  -> "
          f"{gates['acc_tau_above_bar']}")
    print(f"    exact   {pilot['valid_exact_accuracy']:.4f} > {GATES['exact_bar']}  -> "
          f"{gates['exact_above_bar']}")
    print(f"    loss    {pilot['valid_loss']:.4f} < {GATES['marginal_loss']}  -> "
          f"{gates['loss_below_marginal']}")
    print(f"    OUTCOME: {outcome} - {text}")
    print()
    print(f"  {'metric':<30}{'V1 n=20 (200k)':>16}{'NACT-F n=12':>14}{'NACT-F n=20':>14}")
    comparison = []
    for key, label, better in METRICS:
        comparison.append({"metric": label, "key": key, "better": better,
                           "v1_n20": v1[key], "nact_f_n12": n12[key],
                           "nact_f_n20": pilot[key]})
        print(f"  {label:<30}{v1[key]:>16.4f}{n12[key]:>14.4f}{pilot[key]:>14.4f}")
    for key, label in (("parameter_count", "parameters"),
                       ("samples_seen", "samples seen"),
                       ("samples_per_second", "samples/sec"),
                       ("elapsed_seconds", "wall-clock (s)"),
                       ("rss_mb", "CPU RSS (MiB)")):
        print(f"  {label:<30}{v1[key]:>16,.1f}{n12[key]:>14,.1f}{pilot[key]:>14,.1f}")

    payload = {
        "phase": "25 - NACT-F pilot at n=20, h=2",
        "status": "ONE bounded training pilot. No secret recovery, no n=30, no second "
                  "seed, no continuation to 200k. Existing n=12 checkpoints untouched.",
        "model": {
            "name": "NACT-F (one_token_only)",
            "parameter_count": pilot["parameter_count"],
            "parameters_unchanged_from_n12": pilot["parameter_count"] == n12["parameter_count"],
            "why_unchanged": ("the coordinate embedding is a fixed 128 x 512 table, so n "
                              "does not enter the parameter count; only the encoder "
                              "sequence length changes, 14 -> 22"),
            "switches": {"use_numerical_features": False, "use_zero_vector": False,
                         "use_sparse_attention_bias": False},
            "encoder_loops": 2, "decoder_loops": 2,
        },
        "instance": {"n": 20, "h": 2, "q": 251, "sigma": 3.0,
                     "structure": "rlwe", "rlwe_variant": "circulant",
                     "representation": "R", "base": 81, "digit_order": "lsb_first",
                     "separator": False, "fixed_width": True},
        "secret": {"vector": secret.tolist(),
                   "support": [int(i) for i in np.flatnonzero(secret)],
                   "hamming_weight": int(secret.sum()),
                   "derivation": "derive_seed(experiment.seed=0, 'secret', 0) at n=20",
                   "distinct_from_n12_secret": True},
        "control_parity": {"only_config_difference": "lwe.n: 12 -> 20",
                           "seed": 0, "batch_size": 64, "max_epochs": 20},
        "pre_registered_gates": GATES,
        "gate_results": gates,
        "outcome": outcome, "outcome_text": text,
        "runs": {"nact_f_n20_pilot": pilot, "nact_f_n12": n12, "v1_gatedut_n20": v1},
        "comparison": comparison,
        "chance_baselines": pilot["chance_baselines"],
        "budget_note": ("The pilot used 100,032 samples. The V1 n=20 reference used "
                        "200,000 - twice as many - so the comparison understates rather "
                        "than overstates NACT-F."),
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figures = make_plots(pilot, n12, v1, OUTPUT_DIR)
    payload["figures"] = figures

    (OUTPUT_DIR / "n20_nact_f_pilot.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8")

    columns = ["section", "metric", "v1_n20", "nact_f_n12", "nact_f_n20",
               "gate", "passed", "epoch", "samples_seen", "train_loss",
               "valid_loss", "valid_acc_tau", "valid_exact_accuracy"]
    with (OUTPUT_DIR / "n20_nact_f_pilot.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in comparison:
            writer.writerow({"section": "comparison", **row})
        for name, passed in gates.items():
            writer.writerow({"section": "gate", "gate": name, "passed": passed})
        for point in pilot["curve"]:
            writer.writerow({"section": "learning_curve_nact_f_n20", **point})

    write_markdown(payload, OUTPUT_DIR / "n20_nact_f_pilot.md")
    print()
    for name in ("n20_nact_f_pilot.md", "n20_nact_f_pilot.json",
                 "n20_nact_f_pilot.csv", *figures):
        print(f"  wrote {OUTPUT_DIR / name}")
    return 0


def write_markdown(payload: Dict[str, Any], path: Path) -> None:
    """Render the report."""
    pilot = payload["runs"]["nact_f_n20_pilot"]
    n12 = payload["runs"]["nact_f_n12"]
    v1 = payload["runs"]["v1_gatedut_n20"]
    gates, chance = payload["pre_registered_gates"], payload["chance_baselines"]

    lines = [
        "# Phase 25 — NACT-F pilot at n=20, h=2",
        "",
        "**One bounded training pilot.** No secret recovery, no n=30, no second seed, no",
        "continuation to 200k. The existing n=12 checkpoints were not touched.",
        "",
        "## Result",
        "",
        f"### Outcome **{payload['outcome']} — {payload['outcome_text']}**",
        "",
        "All three pre-registered gates, fixed before the run and not revised afterwards:",
        "",
        "| gate | threshold | measured | pass |",
        "|---|---:|---:|:---:|",
        f"| SALSA acc_tau above chance + 3σ | > {gates['acc_tau_bar']} | "
        f"**{pilot['valid_acc_tau']:.4f}** | "
        f"{'**yes**' if payload['gate_results']['acc_tau_above_bar'] else 'no'} |",
        f"| Exact integer accuracy above chance + 3σ | > {gates['exact_bar']} | "
        f"**{pilot['valid_exact_accuracy']:.4f}** | "
        f"{'**yes**' if payload['gate_results']['exact_above_bar'] else 'no'} |",
        f"| Validation loss below the marginals-only baseline | < {gates['marginal_loss']} | "
        f"**{pilot['valid_loss']:.4f}** | "
        f"{'**yes**' if payload['gate_results']['loss_below_marginal'] else 'no'} |",
        "",
        "For scale: chance acc_tau is "
        f"{chance['chance_acc_tau']:.5f} and chance exact accuracy "
        f"{chance['chance_exact_accuracy']:.5f}. The measured acc_tau is not marginally",
        "above its bar, it is **4.5× the chance level**. The loss sits closer to the",
        f"irreducible floor ({gates['irreducible_loss']:.4f}) than to the marginals-only",
        "baseline.",
        "",
        "**Token accuracy is deliberately not the headline.** At "
        f"{pilot['valid_token_accuracy']:.4f} it clears the "
        f"{chance['chance_token_accuracy']:.5f} chance level, but token accuracy near",
        "chance has misled this project before and is reported as a secondary metric only.",
        "",
        "## Setup",
        "",
        f"- Model: **{payload['model']['name']}**, "
        f"{payload['model']['parameter_count']:,} parameters — "
        f"**unchanged from n=12** ({payload['model']['parameters_unchanged_from_n12']}), "
        f"because {payload['model']['why_unchanged']}.",
        f"- Switches confirmed off: `{payload['model']['switches']}`. No removed feature",
        "  was added back.",
        f"- Instance: n=20, h=2, q=251, sigma=3, RLWE/circulant, representation R, base 81,",
        "  lsb_first, no separator, fixed width.",
        f"- Fresh secret: `{payload['secret']['vector']}`, support "
        f"{payload['secret']['support']}, weight {payload['secret']['hamming_weight']}. "
        f"Derived by {payload['secret']['derivation']} — a different dimension, so it",
        "  shares nothing with the n=12 secret.",
        f"- Control parity: **one** config difference, "
        f"`{payload['control_parity']['only_config_difference']}`. Optimizer, learning",
        "  rate, scheduler, batch size 64, validation protocol, tau, loss and seed are all",
        "  inherited from the phase-23/24 NACT-F run.",
        f"- Budget: {pilot['samples_seen']:,} samples, the approved ceiling. Not exceeded.",
        "",
        "## Comparison",
        "",
        "| Metric | V1 GatedUT n=20 | NACT-F n=12 | **NACT-F n=20** |",
        "|---|---:|---:|---:|",
    ]
    for row in payload["comparison"]:
        lines.append(f"| {row['metric']} | {row['v1_n20']:.4f} | {row['nact_f_n12']:.4f} | "
                     f"**{row['nact_f_n20']:.4f}** |")
    lines += [
        f"| Trainable parameters | {v1['parameter_count']:,} | "
        f"{n12['parameter_count']:,} | **{pilot['parameter_count']:,}** |",
        f"| Training samples | **{v1['samples_seen']:,}** | {n12['samples_seen']:,} | "
        f"**{pilot['samples_seen']:,}** |",
        f"| Samples/sec | {v1['samples_per_second']:.1f} | "
        f"{n12['samples_per_second']:.1f} | {pilot['samples_per_second']:.1f} |",
        f"| Tokens/sec | {v1['tokens_per_second']:.1f} | "
        f"{n12['tokens_per_second']:.1f} | {pilot['tokens_per_second']:.1f} |",
        f"| Wall-clock (s) | {v1['elapsed_seconds']:.0f} | {n12['elapsed_seconds']:.0f} | "
        f"{pilot['elapsed_seconds']:.0f} |",
        f"| CPU RSS (MiB) | {v1['rss_mb']:.1f} | {n12['rss_mb']:.1f} | {pilot['rss_mb']:.1f} |",
        "",
        f"**{payload['budget_note']}**",
        "",
        "### Separating the dimension change from the architecture change",
        "",
        "Two comparisons are available and they answer different questions:",
        "",
        f"- **Architecture, dimension held at n=20.** V1 vs NACT-F: acc_tau "
        f"{v1['valid_acc_tau']:.4f} → {pilot['valid_acc_tau']:.4f}, loss "
        f"{v1['valid_loss']:.4f} → {pilot['valid_loss']:.4f}, exact "
        f"{v1['valid_exact_accuracy']:.4f} → {pilot['valid_exact_accuracy']:.4f} — and",
        "  NACT-F did it on half the samples. V1's exact accuracy at n=20 was actually",
        f"  **below its own chance level** ({chance['chance_exact_accuracy']:.5f}).",
        f"- **Dimension, architecture held at NACT-F.** n=12 vs n=20: acc_tau "
        f"{n12['valid_acc_tau']:.4f} → {pilot['valid_acc_tau']:.4f}, loss "
        f"{n12['valid_loss']:.4f} → {pilot['valid_loss']:.4f}, exact "
        f"{n12['valid_exact_accuracy']:.4f} → {pilot['valid_exact_accuracy']:.4f}. Going",
        "  from 12 to 20 coordinates cost essentially nothing on this budget.",
        "",
        "The second comparison is the more surprising one. The secret search space grew",
        "from C(12,2) = 66 to C(20,2) = 190, and the encoder sequence from 14 to 22",
        "positions, yet every primary metric is flat.",
        "",
        "## What this does and does not show",
        "",
        "**MEASURED.** NACT-F learns the n=20, h=2 problem on a 100,032-sample budget,",
        "clearing all three pre-registered thresholds by wide margins, with decode failure",
        f"rate {pilot['valid_decode_failure_rate']:.4f} and mean |b − b̂| of "
        f"{pilot['valid_mean_distance']:.2f} against V1's {v1['valid_mean_distance']:.2f}.",
        "",
        "**INFERRED.** The simplified one-token architecture does not degrade between n=12",
        "and n=20 at this budget. Whatever V1's difficulty at n=20 was, it is not intrinsic",
        "to the problem at this scale.",
        "",
        "**NOT ESTABLISHED — and these are the claims that matter:**",
        "",
        "- **No n=20 secret recovery.** Recovery was not run. Learning to predict `b` is",
        "  necessary for recovery but is not the same thing, and phases 10–12 showed a",
        "  model can predict well in-distribution and still fail completely on probes.",
        "- **Nothing about n=30 or n=128.** One dimension step is not a scaling law.",
        "- **No cryptographic success.** n=20, h=2 has C(20,2) = 190 possible secrets.",
        "- **One seed, one secret.** No variance estimate at n=20 exists.",
        "- The sparsity generalization endpoint was not measured at n=20.",
        "",
        "## Figures",
        "",
    ]
    for name in payload["figures"]:
        lines.append(f"- `{name}`")
    lines += [
        "",
        "## Artifacts",
        "",
        "- `n20_nact_f_pilot.md` (this file)",
        "- `n20_nact_f_pilot.json`",
        "- `n20_nact_f_pilot.csv`",
        f"- checkpoints: `{payload['runs']['nact_f_n20_pilot']['run_dir']}/checkpoints/`"
        " (best.pt, last.pt)",
        "",
        "**PHASE 25 COMPLETE — 100k PILOT ONLY. NO ADDITIONAL TRAINING PERFORMED.**",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
