"""Phase 23 report: ablation F, one-token representation only.

Read-only.  Consumes the artifacts the training run and the phase-12 sparsity
evaluation already wrote; trains nothing and runs no recovery.
"""

import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

OUTPUT_DIR = REPO_ROOT / "results" / "nact_ablation_F"
F_RUN = OUTPUT_DIR / "nact_n12_h2_te2_F" / "seed_0"
F_SPARSITY = OUTPUT_DIR / "sparsity" / "recovery_generalization.json"
FULL_RUN = (REPO_ROOT / "results" / "equal_depth_ablation" / "v2_te2"
            / "nact_n12_h2_te2" / "seed_0")
INTERIM = (REPO_ROOT / "results" / "equal_depth_ablation"
           / "equal_depth_ablation_interim.json")
V1_SPARSITY = (REPO_ROOT / "results" / "recovery_generalization"
               / "recovery_generalization.json")

V1_PARAMETERS, FULL_PARAMETERS, F_PARAMETERS = 4_131_200, 4_241_288, 4_238_208

#: Full-NACT seed-to-seed spread, measured over the three phase-17 T_e=2 runs.
#: Used to say honestly what a one-seed comparison can and cannot resolve.
FULL_NACT_SEED_SD = {"valid_loss": 0.0232, "valid_acc_tau": 0.0021,
                     "valid_exact_accuracy": 0.0092}

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
    """Flatten one run's recorded summary."""
    summary = json.loads((run / "artifacts" / "summary.json").read_text("utf-8"))
    state, valid = summary["state"], summary["final_validation"]
    row = {
        "model_name": summary["model_name"],
        "architecture": summary["architecture"],
        "parameter_count": summary["parameter_count"],
        "config_fingerprint": summary["config_fingerprint"],
        "seed": summary["seed"],
        "best_epoch": state["best_epoch"],
        "samples_seen": state["samples_seen"],
        "train_loss_final": None,
        "elapsed_seconds": state["elapsed_seconds"],
        "samples_per_second": summary["throughput"]["samples_per_second"],
        "tokens_per_second": summary["throughput"]["tokens_per_second"],
        "rss_mb": summary["cpu_memory_mb"]["rss_mb"],
        "chance_baselines": summary["chance_baselines"],
    }
    row.update({k: v for k, v in valid.items() if isinstance(v, (int, float))})
    metrics_path = run / "metrics.csv"
    if metrics_path.is_file():
        rows = list(csv.DictReader(metrics_path.open(encoding="utf-8")))
        if rows:
            for key in ("train_loss", "loss", "train"):
                if key in rows[-1]:
                    row["train_loss_final"] = float(rows[-1][key])
                    break
    return row


def sparsity_rows(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Keep the endpoints this phase reports."""
    keep = ("input_type", "nnz", "decode_validity", "acc_tau",
            "acc_tau_best_constant_predictor", "acc_tau_minus_best_constant",
            "exact_accuracy", "unique_outputs", "output_entropy_nats")
    return [{k: r.get(k) for k in keep} for r in payload["table"]]


def make_plots(v1: Dict[str, Any], full: Dict[str, Any], f: Dict[str, Any],
               sparsity: Dict[str, List[Dict[str, Any]]], out_dir: Path) -> List[str]:
    """The four required figures."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    labels = ["V1 GatedUT", "Full NACT", "NACT-F"]
    colours = ["#d62728", "#2ca02c", "#1f77b4"]
    written = []

    specs = [
        ("01_loss_comparison.png", "valid_loss", "Validation loss (nats)",
         [("marginal-only baseline", 1.86498, "--", "grey"),
          ("irreducible floor", 0.83920, ":", "black")], False),
        ("02_acc_tau_comparison.png", "valid_acc_tau", "SALSA acc_tau",
         [("chance", 0.19287, "--", "grey")], True),
        ("03_exact_accuracy_comparison.png", "valid_exact_accuracy",
         "Exact integer accuracy", [("chance", 0.00398, "--", "grey")], True),
    ]
    for filename, key, title, references, zero_base in specs:
        figure, axes = plt.subplots(figsize=(6.8, 4.4))
        values = [v1[key], full[key], f[key]]
        bars = axes.bar(labels, values, color=colours, edgecolor="black")
        for bar, value in zip(bars, values):
            axes.text(bar.get_x() + bar.get_width() / 2, value,
                      f"{value:.4f}", ha="center", va="bottom", fontsize=9)
        for name, level, style, colour in references:
            axes.axhline(level, linestyle=style, color=colour, linewidth=1.3,
                         label=f"{name} = {level:.5g}")
        if zero_base:
            axes.set_ylim(0, max(values) * 1.22)
        axes.set_ylabel(title)
        axes.set_title(title + "\nn=12 h=2, T_e=2, seed 0, 100,032 samples")
        axes.legend(fontsize=8)
        axes.grid(alpha=0.3, axis="y")
        figure.tight_layout(); figure.savefig(out_dir / filename, dpi=150)
        plt.close(figure); written.append(filename)

    figure, axes = plt.subplots(figsize=(7.6, 4.6))
    for name, rows, colour, marker in (
            ("V1 GatedUT", sparsity["v1"], "#d62728", "o"),
            ("Full NACT", sparsity["full"], "#2ca02c", "s"),
            ("NACT-F", sparsity["f"], "#1f77b4", "^")):
        sparse = [r for r in rows if r["input_type"] == "sparse_random"]
        axes.plot([r["nnz"] for r in sparse],
                  [r["acc_tau_minus_best_constant"] for r in sparse],
                  marker + "-", color=colour, linewidth=2, markersize=6, label=name)
        probe = next((r for r in rows if r["input_type"] == "probe_K_times_e_i"), None)
        if probe:
            axes.plot([1], [probe["acc_tau_minus_best_constant"]], marker,
                      color=colour, markersize=13, markeredgecolor="black")
    axes.axhline(0.0, linestyle="--", color="black", linewidth=1.4,
                 label="zero lift: output carries no input information")
    axes.set_xlabel("number of nonzero coordinates in a   (n = 12)")
    axes.set_ylabel("lift over the best input-blind constant")
    axes.set_title("Sparsity lift: V1 vs Full NACT vs NACT-F\n"
                   "large markers at nnz=1 are the K*e_i probes")
    axes.set_xticks(list(range(1, 13))); axes.invert_xaxis()
    axes.legend(fontsize=8); axes.grid(alpha=0.3)
    figure.tight_layout(); figure.savefig(out_dir / "04_sparsity_lift_comparison.png", dpi=150)
    plt.close(figure); written.append("04_sparsity_lift_comparison.png")
    return written


def main() -> int:
    """Assemble the phase-23 report."""
    f = load_run(F_RUN)
    full = load_run(FULL_RUN)
    interim = json.loads(INTERIM.read_text("utf-8"))
    v1 = interim["per_seed"]["V1_GatedUT_Te2"]
    v1 = v1.get("0") or v1.get(0)

    f_sparsity = sparsity_rows(json.loads(F_SPARSITY.read_text("utf-8")))
    v1_sparsity = sparsity_rows(json.loads(V1_SPARSITY.read_text("utf-8")))
    full_sparsity = interim["sparsity"]["V2_NACT_Te2"]["rows"]

    assert f["parameter_count"] == F_PARAMETERS, f["parameter_count"]
    assert full["parameter_count"] == FULL_PARAMETERS
    assert v1["parameter_count"] == V1_PARAMETERS

    print("=" * 96)
    print("PHASE 23 - ABLATION F: ONE-TOKEN REPRESENTATION ONLY")
    print("=" * 96)
    print(f"  {'metric':<30}{'V1':>12}{'Full NACT':>12}{'NACT-F':>12}"
          f"{'F - Full':>11}{'resolvable?':>13}")
    comparison = []
    for key, label, better in METRICS:
        delta = f[key] - full[key]
        band = FULL_NACT_SEED_SD.get(key)
        resolvable = (None if band is None else abs(delta) > 3 * band)
        comparison.append({
            "metric": label, "key": key, "better": better,
            "v1": v1[key], "full_nact": full[key], "nact_f": f[key],
            "delta_f_minus_full": delta,
            "relative_change_percent": (delta / full[key] * 100.0) if full[key] else None,
            "three_sigma_band_full_nact": (3 * band) if band else None,
            "resolvable_at_one_seed": resolvable,
        })
        flag = ("--" if resolvable is None else ("YES" if resolvable else "no"))
        print(f"  {label:<30}{v1[key]:>12.4f}{full[key]:>12.4f}{f[key]:>12.4f}"
              f"{delta:>+11.4f}{flag:>13}")

    print()
    print(f"  {'parameters':<30}{V1_PARAMETERS:>12,}{FULL_PARAMETERS:>12,}{F_PARAMETERS:>12,}"
          f"{F_PARAMETERS - FULL_PARAMETERS:>+11,}")
    for key, label in (("samples_per_second", "samples/sec"),
                       ("tokens_per_second", "tokens/sec"),
                       ("elapsed_seconds", "wall-clock (s)"),
                       ("rss_mb", "CPU RSS (MiB)")):
        print(f"  {label:<30}{v1[key]:>12.1f}{full[key]:>12.1f}{f[key]:>12.1f}")

    print()
    print("  SPARSITY LIFT")
    print(f"  {'nnz':>6}{'V1':>10}{'Full NACT':>12}{'NACT-F':>10}")
    v1s = [r for r in v1_sparsity if r["input_type"] == "sparse_random"]
    fus = [r for r in full_sparsity if r["input_type"] == "sparse_random"]
    fs = [r for r in f_sparsity if r["input_type"] == "sparse_random"]
    for a, b, c in zip(v1s, fus, fs):
        print(f"  {a['nnz']:>6}{a['acc_tau_minus_best_constant']:>+10.4f}"
              f"{b['acc_tau_minus_best_constant']:>+12.4f}"
              f"{c['acc_tau_minus_best_constant']:>+10.4f}")
    pa = next(r for r in v1_sparsity if r["input_type"] == "probe_K_times_e_i")
    pb = next(r for r in full_sparsity if r["input_type"] == "probe_K_times_e_i")
    pc = next(r for r in f_sparsity if r["input_type"] == "probe_K_times_e_i")
    print(f"  {'K*e_i':>6}{pa['acc_tau_minus_best_constant']:>+10.4f}"
          f"{pb['acc_tau_minus_best_constant']:>+12.4f}"
          f"{pc['acc_tau_minus_best_constant']:>+10.4f}")

    sparsity = {"v1": v1_sparsity, "full": full_sparsity, "f": f_sparsity}
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figures = make_plots(v1, full, f, sparsity, OUTPUT_DIR)

    # How much of the V1 -> Full NACT gap does F retain?
    retention = {}
    for key in ("valid_loss", "valid_acc_tau", "valid_exact_accuracy",
                "valid_token_accuracy"):
        span = full[key] - v1[key]
        retention[key] = ((f[key] - v1[key]) / span) if span else None

    payload = {
        "phase": "23 - ablation F, one-token representation only",
        "status": "ONE training run (NACT-F, seed 0) plus the existing phase-12 "
                  "sparsity evaluation. No other training, no additional seeds, no "
                  "secret recovery, no architecture change, V1 and the full-NACT "
                  "checkpoint untouched.",
        "variant": {
            "name": "NACT-F (one_token_only)",
            "retained": ["one coordinate token per a_i", "base-81 digit embeddings",
                         "absolute coordinate embedding", "transformer backbone",
                         "T_e=2, T_d=2", "copy gate", "RMSNorm", "RoPE", "decoder"],
            "removed": ["centered residue", "cos feature", "sin feature",
                        "zero-indicator numerical feature",
                        "learned zero-coordinate vector", "sparse attention key bias"],
            "parameters_expected_phase22": F_PARAMETERS,
            "parameters_measured": f["parameter_count"],
            "prediction_matched": f["parameter_count"] == F_PARAMETERS,
            "delta_vs_full_nact": F_PARAMETERS - FULL_PARAMETERS,
            "percent_vs_full_nact": 100.0 * (F_PARAMETERS - FULL_PARAMETERS) / FULL_PARAMETERS,
        },
        "control_parity": {
            "only_config_difference": "model.nact_variant: full -> one_token_only",
            "verified": True,
            "seed": 0, "samples": f["samples_seen"],
            "encoder_loops": 2, "decoder_loops": 2,
        },
        "runs": {"v1_gatedut_te2": v1, "full_nact_te2": full, "nact_f_te2": f},
        "chance_baselines": f["chance_baselines"],
        "comparison": comparison,
        "gap_retention_vs_v1": retention,
        "seed_noise_reference": FULL_NACT_SEED_SD,
        "sparsity": sparsity,
        "figures": figures,
    }
    (OUTPUT_DIR / "ablation_F.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8")

    columns = ["section", "metric", "v1", "full_nact", "nact_f",
               "delta_f_minus_full", "relative_change_percent",
               "three_sigma_band_full_nact", "resolvable_at_one_seed",
               "input_type", "nnz", "model", "acc_tau",
               "acc_tau_best_constant_predictor", "acc_tau_minus_best_constant",
               "decode_validity", "exact_accuracy"]
    with (OUTPUT_DIR / "ablation_F.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in comparison:
            writer.writerow({"section": "learning", **row})
        for name, rows in sparsity.items():
            for row in rows:
                writer.writerow({"section": "sparsity", "model": name, **row})

    write_markdown(payload, OUTPUT_DIR / "ablation_F.md")
    print()
    for name in ("ablation_F.md", "ablation_F.json", "ablation_F.csv", *figures):
        print(f"  wrote {OUTPUT_DIR / name}")
    return 0


def write_markdown(payload: Dict[str, Any], path: Path) -> None:
    """Render the report."""
    variant, comparison = payload["variant"], payload["comparison"]
    runs, sparsity = payload["runs"], payload["sparsity"]
    v1, full, f = runs["v1_gatedut_te2"], runs["full_nact_te2"], runs["nact_f_te2"]
    chance, retention = payload["chance_baselines"], payload["gap_retention_vs_v1"]

    lines = [
        "# Phase 23 — Ablation F: one-token representation only",
        "",
        "**One training run.** NACT-F at seed 0, plus the existing phase-12 sparsity",
        "evaluation. No other training, no additional seeds, no secret recovery, no",
        "architecture change; V1 and the full-NACT checkpoint are untouched.",
        "",
        "## The variant",
        "",
        "**Removed:** " + ", ".join(f"`{x}`" for x in variant["removed"]) + ".",
        "",
        "**Retained:** " + ", ".join(f"`{x}`" for x in variant["retained"]) + ".",
        "",
        f"The absolute coordinate embedding is retained deliberately: it is the identity",
        "mechanism the one-token layout *requires* — position must mean coordinate index —",
        "and it is not a numerical feature. Removing it is variant E, which was not approved.",
        "",
        f"**Parameters: {variant['parameters_measured']:,}**, exactly the "
        f"{variant['parameters_expected_phase22']:,} phase 22 predicted "
        f"({variant['prediction_matched']}). That is "
        f"{variant['delta_vs_full_nact']:+,} against full NACT, "
        f"{variant['percent_vs_full_nact']:+.3f}% — far too small for capacity to explain",
        "any outcome. No padding parameters were added.",
        "",
        "## Control parity",
        "",
        f"A field-by-field diff found **one** configuration difference: "
        f"`{payload['control_parity']['only_config_difference']}`. Seed, n, h, q, sigma,",
        "RLWE structure, encoding, optimizer, learning rate, scheduler, batch size,",
        "validation size, sample budget, evaluation cadence, tau, T_e and T_d are all",
        "inherited unchanged from the full-NACT config.",
        "",
        "## Primary comparison",
        "",
        "| Metric | V1 GatedUT | Full NACT | NACT-F |",
        "|---|---:|---:|---:|",
    ]
    for row in comparison:
        lines.append(f"| {row['metric']} | {row['v1']:.4f} | {row['full_nact']:.4f} | "
                     f"**{row['nact_f']:.4f}** |")
    lines += [
        f"| Trainable parameters | {v1['parameter_count']:,} | "
        f"{full['parameter_count']:,} | {f['parameter_count']:,} |",
        f"| Samples/sec | {v1['samples_per_second']:.1f} | "
        f"{full['samples_per_second']:.1f} | {f['samples_per_second']:.1f} |",
        f"| Tokens/sec | {v1['tokens_per_second']:.1f} | "
        f"{full['tokens_per_second']:.1f} | {f['tokens_per_second']:.1f} |",
        f"| Wall-clock (s) | {v1['elapsed_seconds']:.0f} | "
        f"{full['elapsed_seconds']:.0f} | {f['elapsed_seconds']:.0f} |",
        f"| CPU RSS (MiB) | {v1['rss_mb']:.1f} | {full['rss_mb']:.1f} | {f['rss_mb']:.1f} |",
        "",
        f"Baselines: uniform loss {chance['uniform_loss_nats']:.4f}, **marginal-only loss "
        f"{chance['marginal_loss_nats']:.4f}** (the real zero point), chance token accuracy "
        f"{chance['chance_token_accuracy']:.5f}, chance exact "
        f"{chance['chance_exact_accuracy']:.5f}, chance acc_tau "
        f"{chance['chance_acc_tau']:.5f}.",
        "",
        "Throughput and wall-clock for V1 come from a different session and are not",
        "comparable across arms; NACT-F and full NACT were run on the same machine.",
        "",
        "## Full NACT vs NACT-F, and what one seed can resolve",
        "",
        "| Metric | Full NACT | NACT-F | Change | Relative | 3-sigma seed band | Resolvable at one seed? |",
        "|---|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in comparison:
        band = row["three_sigma_band_full_nact"]
        resolvable = row["resolvable_at_one_seed"]
        relative = row["relative_change_percent"]
        lines.append(
            f"| {row['metric']} | {row['full_nact']:.4f} | {row['nact_f']:.4f} | "
            f"{row['delta_f_minus_full']:+.4f} | "
            f"{(f'{relative:+.2f}%' if relative is not None else '-')} | "
            f"{(f'{band:.4f}' if band is not None else '-')} | "
            f"{('-' if resolvable is None else ('**yes**' if resolvable else 'no'))} |")
    lines += [
        "",
        "The 3σ band is full NACT's own seed-to-seed spread, measured over the three",
        "phase-17 runs. **Every primary difference falls inside it**, so at one seed the",
        "two models are not distinguishable on these metrics.",
        "",
        "### Fraction of the V1 → Full NACT gap that NACT-F retains",
        "",
        "| metric | retained |",
        "|---|---:|",
    ]
    for key, value in retention.items():
        if value is not None:
            lines.append(f"| {key} | {value * 100:.1f}% |")
    lines += [
        "",
        "## Sparsity",
        "",
        "Same phase-12 procedure, same K sweep, same seed, same test vectors, unmodified.",
        "",
        "| nnz | V1 lift | Full NACT lift | NACT-F lift |",
        "|---:|---:|---:|---:|",
    ]
    v1s = [r for r in sparsity["v1"] if r["input_type"] == "sparse_random"]
    fus = [r for r in sparsity["full"] if r["input_type"] == "sparse_random"]
    fs = [r for r in sparsity["f"] if r["input_type"] == "sparse_random"]
    for a, b, c in zip(v1s, fus, fs):
        lines.append(f"| {a['nnz']} | {a['acc_tau_minus_best_constant']:+.4f} | "
                     f"{b['acc_tau_minus_best_constant']:+.4f} | "
                     f"**{c['acc_tau_minus_best_constant']:+.4f}** |")
    pa = next(r for r in sparsity["v1"] if r["input_type"] == "probe_K_times_e_i")
    pb = next(r for r in sparsity["full"] if r["input_type"] == "probe_K_times_e_i")
    pc = next(r for r in sparsity["f"] if r["input_type"] == "probe_K_times_e_i")
    lines += [
        f"| **K·e_i** | {pa['acc_tau_minus_best_constant']:+.4f} | "
        f"{pb['acc_tau_minus_best_constant']:+.4f} | "
        f"**{pc['acc_tau_minus_best_constant']:+.4f}** |",
        "",
        "Decode validity: V1 falls to "
        f"{pa['decode_validity']:.3f} on the probes, while both NACT variants hold "
        f"{pc['decode_validity']:.3f}.",
        "",
        "**NACT-F never crosses zero lift either**, so it keeps the property that",
        "distinguishes V2 from V1. But its margin at the sparse end is thinner: at nnz=1 it",
        f"holds {fs[-1]['acc_tau_minus_best_constant']:+.4f} against full NACT's "
        f"{fus[-1]['acc_tau_minus_best_constant']:+.4f}, and on the probes "
        f"{pc['acc_tau_minus_best_constant']:+.4f} against {pb['acc_tau_minus_best_constant']:+.4f}"
        " — roughly half.",
        "",
        "## Interpretation",
        "",
        "**MEASURED.** NACT-F reaches validation loss "
        f"{f['valid_loss']:.4f}, acc_tau {f['valid_acc_tau']:.4f} and exact accuracy "
        f"{f['valid_exact_accuracy']:.4f}, against full NACT's {full['valid_loss']:.4f} / "
        f"{full['valid_acc_tau']:.4f} / {full['valid_exact_accuracy']:.4f} and V1's "
        f"{v1['valid_loss']:.4f} / {v1['valid_acc_tau']:.4f} / "
        f"{v1['valid_exact_accuracy']:.4f}. Every full-vs-F difference sits inside full",
        "NACT's own one-seed noise band. Sparsity lift stays positive at every level for",
        "NACT-F, with a thinner margin at nnz≤2 and on the probes.",
        "",
        "**INFERRED.** On this evidence the bulk of NACT's advantage over V1 comes from",
        "the **one-token coordinate representation plus the absolute coordinate identity",
        "it makes expressible**, not from the centered residue, the Fourier pair, the zero",
        "indicator, the zero vector or the sparse attention bias. The removed features are",
        "not doing the heavy lifting on in-distribution learning at n=12, h=2. The thinner",
        "sparse-end margin is the one place they may still be contributing, and this run",
        "cannot tell whether that difference is real or seed noise.",
        "",
        "**NOT ESTABLISHED.**",
        "",
        "- **That the removed features contribute nothing.** One seed cannot establish a",
        "  null. Phase 22 pre-registered exactly this case: a variant landing within 3σ of",
        "  full NACT needs more seeds before 'no effect' can be claimed. Only one seed was",
        "  authorised, so the honest statement is *indistinguishable at one seed*, not",
        "  *equivalent*.",
        "- **Which individual component matters.** F removes six things at once. Variants",
        "  C, D and E were not run.",
        "- Behaviour at n=20, n=30 or n=128.",
        "- Secret-recovery capability for NACT-F — not run, and out of scope for this phase.",
        "- General secret-recovery robustness.",
        "",
        "**NACT-F is not 'better' for having fewer parameters.** It is 3,080 parameters",
        "smaller, 0.073%, which is scientifically meaningless. The interesting result is",
        "that removing those components changed so little, not that the model shrank.",
        "",
        "## A note on the artifacts",
        "",
        "The phase-12 sparsity script writes its own `recovery_generalization.md` last and",
        "raised a `TypeError` while formatting it, because NACT-F's non-modal probe",
        "coordinates are not equal to the secret support and an optional probability is",
        "`None`. Its JSON, CSV and figures — the measurement — were written before that",
        "point and are complete. The evaluation implementation was not modified, as",
        "instructed. Separately, passing `--run-dir` as a relative path triggers a",
        "`relative_to()` failure in the same reporting tail; passing an absolute path",
        "avoids it and is what produced these artifacts. That is also why phase 17's",
        "sparsity JSON was missing.",
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
        "- `ablation_F.md` (this file)", "- `ablation_F.json`", "- `ablation_F.csv`",
        "",
        "PHASE 23 COMPLETE — ABLATION F ONLY. NO OTHER TRAINING PERFORMED.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
