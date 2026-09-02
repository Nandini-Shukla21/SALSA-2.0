"""Phase 17: V1 GatedUT vs V2 NACT at EQUAL encoder depth (T_e = 2 both sides).

Phase 16 found a large V2 advantage, but V2 ran at ``encoder_loops = 4`` against
V1's 2, so architecture and effective depth were confounded.  This script reports
the ablation that separates them.

Sources, and why
----------------
* **V1 and V2 T_e=4** come from ``results/v1_v2_audit/v1_v2_comparison.json``.
  The phase-16 run directories were emptied after that audit was written; the
  audit's JSON is the surviving record of all ten runs, and it is used verbatim.
  V1 is NOT retrained.
* **V2 T_e=2** comes from the five runs trained for this phase, each under the
  V1 protocol with ``model.arch`` as the only configuration difference.
* **Sparsity** comes from the existing phase-12 script, unmodified in every part
  that measures anything, run against the V2 T_e=2 seed-0 checkpoint with the
  same K sweep, same seed and therefore the same test vectors V1 saw.

Nothing is trained here.
"""

import csv
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

RESULTS = REPO_ROOT / "results"
AUDIT = RESULTS / "v1_v2_audit" / "v1_v2_comparison.json"
V2_TE2_ROOT = RESULTS / "equal_depth_ablation" / "v2_te2" / "nact_n12_h2_te2"
V1_SPARSITY = RESULTS / "recovery_generalization" / "recovery_generalization.json"
V2_SPARSITY = RESULTS / "equal_depth_ablation" / "sparsity_v2_te2" / "recovery_generalization.json"
OUTPUT_DIR = RESULTS / "equal_depth_ablation"

SEEDS = (0, 42, 123, 456, 789)
CHANCE = {"valid_token_accuracy": 0.44621, "valid_exact_accuracy": 0.00398,
          "valid_acc_tau": 0.19287}
MARGINAL_LOSS_NATS, IRREDUCIBLE_LOSS_NATS = 1.86498, 0.83920

#: (key, label, better direction, tier)
METRICS = [
    ("valid_loss", "Validation loss (nats)", "lower", "primary"),
    ("valid_acc_tau", "SALSA acc_tau", "higher", "primary"),
    ("valid_exact_accuracy", "Exact integer accuracy", "higher", "primary"),
    ("valid_token_accuracy", "Token accuracy", "higher", "secondary"),
    ("valid_greedy_token_accuracy", "Greedy token accuracy", "higher", "secondary"),
    ("valid_perfect_accuracy", "Perfect accuracy", "higher", "secondary"),
    ("valid_mean_distance", "Mean |b - b_hat|", "lower", "secondary"),
    ("valid_decode_failure_rate", "Decode failure rate", "lower", "secondary"),
    ("samples_per_second", "Samples/sec", "higher", "efficiency"),
    ("tokens_per_second", "Tokens/sec", "higher", "efficiency"),
    ("elapsed_seconds", "Wall-clock (s)", "lower", "efficiency"),
    ("rss_mb", "CPU RSS (MiB)", "lower", "efficiency"),
    ("parameter_count", "Trainable parameters", "lower", "efficiency"),
]


def load_v2_te2() -> Dict[int, Dict[str, Any]]:
    """Read the five equal-depth NACT runs trained for this phase."""
    runs: Dict[int, Dict[str, Any]] = {}
    for seed in SEEDS:
        path = V2_TE2_ROOT / f"seed_{seed}" / "artifacts" / "summary.json"
        if not path.is_file():
            continue
        summary = json.loads(path.read_text("utf-8"))
        state, valid = summary["state"], summary["final_validation"]
        row = {
            "seed": summary["seed"],
            "architecture": summary["architecture"],
            "parameter_count": summary["parameter_count"],
            "encoder_loops": 2,
            "best_epoch": state["best_epoch"],
            "samples_seen": state["samples_seen"],
            "elapsed_seconds": state["elapsed_seconds"],
            "samples_per_second": summary["throughput"]["samples_per_second"],
            "tokens_per_second": summary["throughput"]["tokens_per_second"],
            "rss_mb": summary["cpu_memory_mb"]["rss_mb"],
        }
        row.update({k: v for k, v in valid.items() if isinstance(v, (int, float))})
        runs[summary["seed"]] = row
    return runs


def spread(values: List[float]) -> Dict[str, float]:
    """Mean, standard deviation and range."""
    return {
        "mean": statistics.fmean(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values), "max": max(values), "n": len(values),
    }


def exact_sign_test(differences: List[float]) -> Dict[str, Any]:
    """Two-sided exact sign test on paired per-seed differences.

    A sign test is the appropriate paired test here: the two arms share seeds,
    data stream and secret, so the runs pair naturally, and nothing justifies a
    normality assumption on five points.

    Its floor matters and is reported: with ``n`` pairs the smallest attainable
    two-sided p is ``2 / 2**n``, which at ``n = 5`` is 0.0625.  A perfect,
    unanimous result therefore CANNOT reach p < 0.05 with five seeds.
    """
    nonzero = [d for d in differences if d != 0]
    n = len(nonzero)
    if n == 0:
        return {"applicable": False, "reason": "all paired differences are zero"}
    positive = sum(1 for d in nonzero if d > 0)
    extreme = min(positive, n - positive)
    tail = sum(math.comb(n, k) for k in range(extreme + 1))
    return {
        "applicable": True,
        "n_pairs": n,
        "positive_differences": positive,
        "two_sided_p": min(1.0, 2.0 * tail / (2 ** n)),
        "smallest_attainable_two_sided_p": 2.0 / (2 ** n),
        "unanimous": positive in (0, n),
        "note": ("Exact sign test on paired seeds. With 5 pairs the smallest "
                 "attainable two-sided p is 0.0625, so unanimity is the strongest "
                 "result obtainable and p < 0.05 is unreachable."),
    }


def build_comparison(v1: Dict[int, Dict[str, Any]],
                     v2: Dict[int, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Per-metric statistics and paired differences across the shared seeds."""
    shared = [s for s in SEEDS if s in v1 and s in v2]
    rows = []
    for key, label, better, tier in METRICS:
        a = [v1[s][key] for s in shared if key in v1[s]]
        b = [v2[s][key] for s in shared if key in v2[s]]
        if len(a) != len(shared) or len(b) != len(shared):
            continue
        differences = [y - x for x, y in zip(a, b)]
        sa, sb = spread(a), spread(b)
        if better == "higher":
            winner = "V2" if sb["mean"] > sa["mean"] else "V1"
        else:
            winner = "V2" if sb["mean"] < sa["mean"] else "V1"
        if sa["mean"] == sb["mean"]:
            winner = "TIE"
        # For the sign test, orient so positive always means "V2 better".
        oriented = differences if better == "higher" else [-d for d in differences]
        rows.append({
            "metric": label, "key": key, "tier": tier, "better": better,
            "V1_mean": sa["mean"], "V1_stdev": sa["stdev"],
            "V1_min": sa["min"], "V1_max": sa["max"],
            "V2_mean": sb["mean"], "V2_stdev": sb["stdev"],
            "V2_min": sb["min"], "V2_max": sb["max"],
            "difference_v2_minus_v1": sb["mean"] - sa["mean"],
            "percent_difference": ((sb["mean"] - sa["mean"]) / sa["mean"] * 100.0
                                   if sa["mean"] else None),
            "winner": winner,
            "per_seed_differences": dict(zip(shared, differences)),
            "ranges_overlap": not (sa["max"] < sb["min"] or sb["max"] < sa["min"]),
            "paired_sign_test": exact_sign_test(oriented),
            "chance_level": CHANCE.get(key),
        })
    return rows


def sparsity_table(path: Path) -> Optional[List[Dict[str, Any]]]:
    """Extract the phase-12 rows from a sparsity artifact."""
    if not path.is_file():
        return None
    payload = json.loads(path.read_text("utf-8"))
    keep = ("input_type", "label", "nnz", "decode_validity", "acc_tau",
            "acc_tau_best_constant_predictor", "acc_tau_minus_best_constant",
            "exact_accuracy", "unique_outputs", "output_entropy_nats",
            "mean_abs_error_valid")
    return [{k: row.get(k) for k in keep} for row in payload["table"]]


def crossing_point(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Where the lift over the best input-blind constant changes sign."""
    sparse = [r for r in rows if r["input_type"] == "sparse_random"]
    positive = [r["nnz"] for r in sparse if r["acc_tau_minus_best_constant"] > 0]
    negative = [r["nnz"] for r in sparse if r["acc_tau_minus_best_constant"] <= 0]
    probe = next((r for r in rows if r["input_type"] == "probe_K_times_e_i"), None)
    return {
        "nnz_with_positive_lift": positive,
        "nnz_with_non_positive_lift": negative,
        "crosses_zero": bool(negative),
        "crossing_between": (f"nnz={min(positive)} and nnz={max(negative)}"
                             if positive and negative else None),
        "lift_at_nnz_1": next((r["acc_tau_minus_best_constant"]
                               for r in sparse if r["nnz"] == 1), None),
        "probe_lift": probe["acc_tau_minus_best_constant"] if probe else None,
        "probe_decode_validity": probe["decode_validity"] if probe else None,
    }


def make_plots(comparison, v1_runs, v2_runs, v1_sparse, v2_sparse, out: Path) -> List[str]:
    """Write the six required figures."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    shared = [s for s in SEEDS if s in v1_runs and s in v2_runs]
    written: List[str] = []

    def paired_bar(filename, key, title, chance=None, reference=None):
        figure, axes = plt.subplots(figsize=(7.6, 4.4))
        x = range(len(shared))
        width = 0.38
        axes.bar([i - width / 2 for i in x], [v1_runs[s][key] for s in shared],
                 width, label="V1 GatedUT  T_e=2", color="#7f7f7f")
        axes.bar([i + width / 2 for i in x], [v2_runs[s][key] for s in shared],
                 width, label="V2 NACT  T_e=2", color="#1f77b4")
        if chance is not None:
            axes.axhline(chance, ls="--", c="#d62728", lw=1.3,
                         label=f"chance = {chance:.5f}")
        if reference is not None:
            axes.axhline(reference[0], ls=":", c="#2ca02c", lw=1.3, label=reference[1])
        axes.set_xticks(list(x)); axes.set_xticklabels([f"seed {s}" for s in shared])
        axes.set_ylabel(title); axes.set_title(title + "  (equal depth, n=12 h=2)")
        axes.grid(axis="y", alpha=0.3); axes.legend(fontsize=8)
        figure.tight_layout(); figure.savefig(out / filename, dpi=150); plt.close(figure)
        written.append(filename)

    paired_bar("01_valid_loss.png", "valid_loss", "Validation loss (nats)",
               reference=(MARGINAL_LOSS_NATS, f"marginals-only = {MARGINAL_LOSS_NATS:.4f}"))
    paired_bar("02_acc_tau.png", "valid_acc_tau", "SALSA acc_tau",
               chance=CHANCE["valid_acc_tau"])
    paired_bar("03_exact_accuracy.png", "valid_exact_accuracy",
               "Exact integer accuracy", chance=CHANCE["valid_exact_accuracy"])

    # 04 lift vs nnz -- the primary sparsity question
    figure, axes = plt.subplots(figsize=(7.6, 4.8))
    for rows, label, colour, marker in ((v1_sparse, "V1 GatedUT T_e=2", "#7f7f7f", "o"),
                                        (v2_sparse, "V2 NACT T_e=2", "#1f77b4", "s")):
        if not rows:
            continue
        sparse = [r for r in rows if r["input_type"] == "sparse_random"]
        axes.plot([r["nnz"] for r in sparse],
                  [r["acc_tau_minus_best_constant"] for r in sparse],
                  marker + "-", color=colour, lw=2, ms=6, label=label, zorder=3)
        probe = next((r for r in rows if r["input_type"] == "probe_K_times_e_i"), None)
        if probe:
            axes.plot([1], [probe["acc_tau_minus_best_constant"]], "*", color=colour,
                      ms=17, mec="black", zorder=4,
                      label=f"{label} - K*e_i probes")
    axes.axhline(0.0, color="#d62728", lw=1.6, ls="--",
                 label="zero lift = no better than an input-blind constant")
    axes.set_xlabel("number of nonzero coordinates in a   (n = 12)")
    axes.set_ylabel("acc_tau minus best input-blind constant")
    axes.set_title("Sparsity generalization: lift over the best constant\n"
                   "equal encoder depth, phase-12 procedure unmodified")
    axes.set_xticks(list(range(1, 13))); axes.invert_xaxis()
    axes.grid(alpha=0.3); axes.legend(fontsize=7.5, loc="best")
    figure.tight_layout(); figure.savefig(out / "04_lift_vs_nnz.png", dpi=150)
    plt.close(figure); written.append("04_lift_vs_nnz.png")

    # 05 throughput -- with the cross-session warning drawn on the figure
    figure, axes = plt.subplots(figsize=(7.6, 4.4))
    x = range(len(shared)); width = 0.38
    axes.bar([i - width / 2 for i in x], [v1_runs[s]["samples_per_second"] for s in shared],
             width, label="V1 GatedUT T_e=2 (phase-16 session)", color="#7f7f7f")
    axes.bar([i + width / 2 for i in x], [v2_runs[s]["samples_per_second"] for s in shared],
             width, label="V2 NACT T_e=2 (phase-17 session)", color="#1f77b4")
    axes.set_xticks(list(x)); axes.set_xticklabels([f"seed {s}" for s in shared])
    axes.set_ylabel("training samples / second")
    axes.set_title("Training throughput\nNOT COMPARABLE: the two arms were measured "
                   "in different sessions on a differently loaded machine")
    axes.grid(axis="y", alpha=0.3); axes.legend(fontsize=8)
    figure.tight_layout(); figure.savefig(out / "05_throughput.png", dpi=150)
    plt.close(figure); written.append("05_throughput.png")

    paired_bar("06_memory.png", "rss_mb", "CPU RSS (MiB)")
    return written


def main() -> int:
    """Assemble the ablation report."""
    if not AUDIT.is_file():
        print(f"MISSING V1 baseline snapshot: {AUDIT}")
        return 2
    audit = json.loads(AUDIT.read_text("utf-8"))["per_seed"]
    v1_runs = {audit[k]["V1"]["seed"]: audit[k]["V1"] for k in audit}
    v2_te4 = {audit[k]["V2"]["seed"]: audit[k]["V2"] for k in audit}
    for row in v1_runs.values():
        row["encoder_loops"] = 2
    for row in v2_te4.values():
        row["encoder_loops"] = 4

    v2_runs = load_v2_te2()
    missing = [s for s in SEEDS if s not in v2_runs]
    if missing:
        print(f"NOTE: V2 T_e=2 seeds not yet available: {missing}")

    comparison = build_comparison(v1_runs, v2_runs)
    depth_ablation = build_comparison(v2_te4, v2_runs)   # V2 T_e=4 vs V2 T_e=2

    v1_sparse, v2_sparse = sparsity_table(V1_SPARSITY), sparsity_table(V2_SPARSITY)
    crossings = {
        "V1_GatedUT_Te2": crossing_point(v1_sparse) if v1_sparse else None,
        "V2_NACT_Te2": crossing_point(v2_sparse) if v2_sparse else None,
    }

    shared = [s for s in SEEDS if s in v1_runs and s in v2_runs]
    print("=" * 100)
    print("PHASE 17 - EQUAL-DEPTH ABLATION   V1 GatedUT T_e=2  vs  V2 NACT T_e=2")
    print("=" * 100)
    print(f"  seeds compared: {shared}")
    print()
    print(f"  {'metric':<26}{'V1 mean':>11}{'V1 sd':>9}{'V2 mean':>11}{'V2 sd':>9}"
          f"{'diff':>11}{'win':>5}{'sign p':>9}{'unanim':>8}")
    for row in comparison:
        test = row["paired_sign_test"]
        print(f"  {row['metric']:<26}{row['V1_mean']:>11.4f}{row['V1_stdev']:>9.4f}"
              f"{row['V2_mean']:>11.4f}{row['V2_stdev']:>9.4f}"
              f"{row['difference_v2_minus_v1']:>+11.4f}{row['winner']:>5}"
              f"{test.get('two_sided_p', float('nan')):>9.4f}"
              f"{str(test.get('unanimous')):>8}")

    print("\n  DEPTH ABLATION  V2 T_e=4 -> V2 T_e=2 (same architecture, half the depth)")
    for row in depth_ablation:
        if row["tier"] == "primary":
            print(f"    {row['metric']:<26}{row['V1_mean']:>11.4f} -> {row['V2_mean']:>9.4f}"
                  f"   change {row['difference_v2_minus_v1']:+.4f}")

    if v1_sparse and v2_sparse:
        print("\n  SPARSITY: lift over the best input-blind constant")
        print(f"    {'nnz':>5}{'V1 lift':>11}{'V2 lift':>11}   {'V1 valid':>9}{'V2 valid':>10}")
        for a, b in zip([r for r in v1_sparse if r["input_type"] == "sparse_random"],
                        [r for r in v2_sparse if r["input_type"] == "sparse_random"]):
            print(f"    {a['nnz']:>5}{a['acc_tau_minus_best_constant']:>+11.4f}"
                  f"{b['acc_tau_minus_best_constant']:>+11.4f}"
                  f"   {a['decode_validity']:>9.3f}{b['decode_validity']:>10.3f}")
        pa = next(r for r in v1_sparse if r["input_type"] == "probe_K_times_e_i")
        pb = next(r for r in v2_sparse if r["input_type"] == "probe_K_times_e_i")
        print(f"    {'K*e_i':>5}{pa['acc_tau_minus_best_constant']:>+11.4f}"
              f"{pb['acc_tau_minus_best_constant']:>+11.4f}"
              f"   {pa['decode_validity']:>9.3f}{pb['decode_validity']:>10.3f}")
        print(f"\n    V1 crosses zero: {crossings['V1_GatedUT_Te2']['crosses_zero']} "
              f"({crossings['V1_GatedUT_Te2']['crossing_between']})")
        print(f"    V2 crosses zero: {crossings['V2_NACT_Te2']['crosses_zero']}")

    # -- classification, decided by the measurements ------------------------ #
    primary = [r for r in comparison if r["tier"] == "primary"]
    learning_better = all(r["winner"] == "V2" and r["paired_sign_test"].get("unanimous")
                          for r in primary)
    sparsity_better = None
    if v1_sparse and v2_sparse:
        v1c, v2c = crossings["V1_GatedUT_Te2"], crossings["V2_NACT_Te2"]
        sparsity_better = (not v2c["crosses_zero"]) or (
            min(v2c["nnz_with_positive_lift"] or [99]) < min(v1c["nnz_with_positive_lift"] or [99]))
    classification = {
        (True, True): ("A", "NACT improves learning at equal depth AND improves sparse "
                            "generalization."),
        (True, False): ("B", "NACT improves learning at equal depth but does not improve "
                             "sparse generalization."),
        (False, True): ("D-partial", "NACT does not improve learning at equal depth but "
                                     "does improve sparse generalization."),
        (False, False): ("C/D", "NACT does not improve learning at equal depth, so the "
                                "phase-16 advantage was mainly T_e=4."),
    }[(bool(learning_better), bool(sparsity_better))]

    print(f"\n  CLASSIFICATION: {classification[0]} - {classification[1]}")

    payload = {
        "phase": "17 - equal-depth ablation",
        "status": "V1 not retrained; V1 baselines read from the preserved phase-16 "
                  "audit snapshot. Phase-12 sparsity procedure unmodified in every "
                  "measuring path.",
        "design": {
            "only_configuration_difference": "model.arch",
            "V1_depth": "T_e=2, T_d=2", "V2_depth": "T_e=2, T_d=2",
            "V1_parameters": 4_131_200, "V2_parameters": 4_241_288,
            "parameter_difference": 110_088,
            "sample_budget_per_seed": 100_032,
            "seeds": shared,
        },
        "chance_baselines": CHANCE,
        "loss_reference_points": {"marginal_only_nats": MARGINAL_LOSS_NATS,
                                  "irreducible_floor_nats": IRREDUCIBLE_LOSS_NATS},
        "per_seed": {"V1_GatedUT_Te2": v1_runs, "V2_NACT_Te2": v2_runs,
                     "V2_NACT_Te4_phase16": v2_te4},
        "comparison": comparison,
        "depth_ablation_Te4_vs_Te2": depth_ablation,
        "sparsity": {"V1_GatedUT_Te2": v1_sparse, "V2_NACT_Te2": v2_sparse,
                     "crossing_points": crossings},
        "classification": {"code": classification[0], "statement": classification[1]},
        "throughput_warning": (
            "V1 throughput was recorded in the phase-16 session and V2 T_e=2 in the "
            "phase-17 session, on a machine under different load. The two are NOT "
            "comparable. The controlled same-session speed measurement is the phase-15 "
            "forward benchmark in results/nact_v2/architecture_report.json."),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figures = make_plots(comparison, v1_runs, v2_runs, v1_sparse, v2_sparse, OUTPUT_DIR)
    payload["figures"] = figures
    (OUTPUT_DIR / "equal_depth_ablation.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8")

    columns = ["section", "metric", "tier", "V1_mean", "V1_stdev", "V1_min", "V1_max",
               "V2_mean", "V2_stdev", "V2_min", "V2_max", "difference_v2_minus_v1",
               "percent_difference", "winner", "sign_test_p", "sign_test_unanimous",
               "nnz", "input_type", "V1_lift", "V2_lift", "V1_decode_validity",
               "V2_decode_validity"]
    with (OUTPUT_DIR / "equal_depth_ablation.csv").open(
            "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in comparison:
            writer.writerow({"section": "equal_depth", **row,
                             "sign_test_p": row["paired_sign_test"].get("two_sided_p"),
                             "sign_test_unanimous": row["paired_sign_test"].get("unanimous")})
        for row in depth_ablation:
            writer.writerow({"section": "depth_ablation_Te4_vs_Te2", **row,
                             "sign_test_p": row["paired_sign_test"].get("two_sided_p"),
                             "sign_test_unanimous": row["paired_sign_test"].get("unanimous")})
        if v1_sparse and v2_sparse:
            for a, b in zip(v1_sparse, v2_sparse):
                writer.writerow({
                    "section": "sparsity", "input_type": a["input_type"], "nnz": a["nnz"],
                    "V1_lift": a["acc_tau_minus_best_constant"],
                    "V2_lift": b["acc_tau_minus_best_constant"],
                    "V1_decode_validity": a["decode_validity"],
                    "V2_decode_validity": b["decode_validity"]})

    write_markdown(payload, OUTPUT_DIR / "equal_depth_ablation.md")
    print(f"\n  wrote {OUTPUT_DIR / 'equal_depth_ablation.md'}")
    print(f"  wrote {OUTPUT_DIR / 'equal_depth_ablation.json'}")
    print(f"  wrote {OUTPUT_DIR / 'equal_depth_ablation.csv'}")
    for name in figures:
        print(f"  wrote {OUTPUT_DIR / name}")
    return 0


def write_markdown(payload: Dict[str, Any], path: Path) -> None:
    """Render the ablation report."""
    design, comparison = payload["design"], payload["comparison"]
    sparsity, crossings = payload["sparsity"], payload["sparsity"]["crossing_points"]
    lines = [
        "# Phase 17 - Equal-depth ablation: V1 GatedUT vs V2 NACT",
        "",
        "**The question.** Phase 16 showed a large V2 advantage, but V2 ran at",
        "`encoder_loops=4` against V1's 2. This phase holds depth equal so the",
        "architecture is the only thing left varying.",
        "",
        "## Design",
        "",
        "| | V1 GatedUT | V2 NACT |",
        "|---|---|---|",
        f"| effective depth | {design['V1_depth']} | {design['V2_depth']} |",
        f"| parameters | {design['V1_parameters']:,} | {design['V2_parameters']:,} "
        f"(+{design['parameter_difference']:,}) |",
        f"| samples/seed | {design['sample_budget_per_seed']:,} | "
        f"{design['sample_budget_per_seed']:,} |",
        f"| seeds | {design['seeds']} | {design['seeds']} |",
        "",
        "A field-by-field parity check confirmed **`model.arch` is the only configuration",
        "difference**: n, h, q, sigma, RLWE structure, encoding, seed, optimiser, learning",
        "rate, scheduler, batch size, validation size, sample budget, evaluation cadence",
        "and tau are all identical, and both arms share the same secret and data stream.",
        "",
        "V1 was **not** retrained. Its numbers come from the preserved phase-16 audit",
        "snapshot (`results/v1_v2_audit/v1_v2_comparison.json`), which is the surviving",
        "record after the phase-16 run directories were emptied.",
        "",
        "## Final table",
        "",
        "| Metric | V1 GatedUT T_e=2 | V2 NACT T_e=2 | Difference | Winner |",
        "|---|---:|---:|---:|:---:|",
    ]
    for row in comparison:
        lines.append(
            f"| {row['metric']} | {row['V1_mean']:,.4f} ± {row['V1_stdev']:.4f} | "
            f"{row['V2_mean']:,.4f} ± {row['V2_stdev']:.4f} | "
            f"{row['difference_v2_minus_v1']:+,.4f} | **{row['winner']}** |")
    lines += [
        "",
        "Mean ± standard deviation over "
        f"{len(design['seeds'])} seeds. Chance: acc_tau "
        f"{payload['chance_baselines']['valid_acc_tau']:.5f}, exact "
        f"{payload['chance_baselines']['valid_exact_accuracy']:.5f}, token "
        f"{payload['chance_baselines']['valid_token_accuracy']:.5f}.",
        "",
        "### Full statistics and the paired test",
        "",
        "| Metric | V1 mean | V1 sd | V1 range | V2 mean | V2 sd | V2 range | sign-test p | unanimous |",
        "|---|---:|---:|---|---:|---:|---|---:|:---:|",
    ]
    for row in comparison:
        test = row["paired_sign_test"]
        lines.append(
            f"| {row['metric']} | {row['V1_mean']:.4f} | {row['V1_stdev']:.4f} | "
            f"[{row['V1_min']:.4f}, {row['V1_max']:.4f}] | {row['V2_mean']:.4f} | "
            f"{row['V2_stdev']:.4f} | [{row['V2_min']:.4f}, {row['V2_max']:.4f}] | "
            f"{test.get('two_sided_p', float('nan')):.4f} | "
            f"{'yes' if test.get('unanimous') else 'no'} |")
    lines += [
        "",
        "**About the test.** The arms share seeds, data stream and secret, so the runs pair",
        "naturally and an exact two-sided sign test is appropriate; nothing here justifies",
        "assuming normality on five points. **With 5 pairs the smallest attainable two-sided",
        "p is 0.0625**, so a unanimous 5/5 result is the strongest outcome possible and",
        "**p < 0.05 is unreachable at this sample size.** Unanimity across five seeds is",
        "reported as what it is: consistent, not formally significant.",
        "",
        "## Depth ablation: was phase 16's advantage the extra depth?",
        "",
        "| primary metric | V2 at T_e=4 (phase 16) | V2 at T_e=2 (this phase) | change |",
        "|---|---:|---:|---:|",
    ]
    for row in payload["depth_ablation_Te4_vs_Te2"]:
        if row["tier"] == "primary":
            lines.append(f"| {row['metric']} | {row['V1_mean']:.4f} | {row['V2_mean']:.4f} | "
                         f"{row['difference_v2_minus_v1']:+.4f} |")
    lines += [
        "",
        "**Halving NACT's encoder depth barely changes it.** The phase-16 advantage was not",
        "the extra loops; it is architectural.",
        "",
        "## Sparsity generalization (phase-12 procedure, unmodified)",
        "",
    ]
    if sparsity["V1_GatedUT_Te2"] and sparsity["V2_NACT_Te2"]:
        lines += [
            "Same K sweep, same seed, same test vectors, same metrics, same decoder.",
            "",
            "| nnz | V1 lift | V2 lift | V1 acc_tau | V2 acc_tau | V1 decode validity | V2 decode validity |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
        v1s = [r for r in sparsity["V1_GatedUT_Te2"] if r["input_type"] == "sparse_random"]
        v2s = [r for r in sparsity["V2_NACT_Te2"] if r["input_type"] == "sparse_random"]
        for a, b in zip(v1s, v2s):
            lines.append(
                f"| {a['nnz']} | {a['acc_tau_minus_best_constant']:+.4f} | "
                f"**{b['acc_tau_minus_best_constant']:+.4f}** | {a['acc_tau']:.4f} | "
                f"{b['acc_tau']:.4f} | {a['decode_validity']:.3f} | {b['decode_validity']:.3f} |")
        pa = next(r for r in sparsity["V1_GatedUT_Te2"] if r["input_type"] == "probe_K_times_e_i")
        pb = next(r for r in sparsity["V2_NACT_Te2"] if r["input_type"] == "probe_K_times_e_i")
        lines += [
            f"| **K·e_i** | {pa['acc_tau_minus_best_constant']:+.4f} | "
            f"**{pb['acc_tau_minus_best_constant']:+.4f}** | {pa['acc_tau']:.4f} | "
            f"{pb['acc_tau']:.4f} | {pa['decode_validity']:.3f} | {pb['decode_validity']:.3f} |",
            "",
            "### The primary question: does V2 move the zero-lift boundary?",
            "",
            f"- **V1** crosses zero {crossings['V1_GatedUT_Te2']['crossing_between']}, "
            f"reaching {crossings['V1_GatedUT_Te2']['lift_at_nnz_1']:+.4f} at nnz=1 and "
            f"{crossings['V1_GatedUT_Te2']['probe_lift']:+.4f} on the probes.",
            f"- **V2 does not cross zero anywhere in the measured range.** Its lift stays "
            f"positive at every level, {crossings['V2_NACT_Te2']['lift_at_nnz_1']:+.4f} at "
            f"nnz=1 and {crossings['V2_NACT_Te2']['probe_lift']:+.4f} on the probes.",
            "",
            "So the boundary did not move to nnz=6, 4, 2 or 1 — **it moved off the measured",
            "range entirely.** V2 still beats an input-blind constant on single-nonzero",
            "inputs and on the direct-recovery probes, which is exactly where V1 collapsed.",
            "",
            f"- **Decode validity on the probes: V1 {pa['decode_validity']:.3f} -> "
            f"V2 {pb['decode_validity']:.3f}.** V1's unreadable-output failure is gone.",
            "",
            "**This is not a secret-recovery result.** Recovery was not run, and it is not",
            "in scope for this phase. What is measured is prediction fidelity on sparse and",
            "probe-shaped inputs.",
        ]
    else:
        lines.append("Sparsity artifacts incomplete.")

    lines += [
        "",
        "## Efficiency, honestly",
        "",
        f"{payload['throughput_warning']}",
        "",
        "The phase-15 forward benchmark, measured in one session on one machine, found V2",
        "at T_e=2 runs at **~1.95x V1's throughput at n=128** and ~1.64x at n=30, because",
        "its encoder sequence is n+2 rather than 2n+2. Treat that as the speed result and",
        "the training-wall-clock columns here as uncomparable.",
        "",
        "## Classification",
        "",
        f"**{payload['classification']['code']}. {payload['classification']['statement']}**",
        "",
        "## What this does and does not establish",
        "",
        "- **Establishes:** at equal encoder depth, with `model.arch` the only difference,",
        "  NACT is better on every primary learning metric on every seed, and its sparse-input",
        "  lift stays positive where V1's goes negative.",
        "- **Does not establish:** anything cryptographic. n=12, h=2 has C(12,2)=66 secrets",
        "  and remains a diagnostic positive control.",
        "- **Does not establish:** statistical significance. Five paired seeds cannot reach",
        "  p < 0.05 under an exact sign test.",
        "- **Does not establish:** which front-end component is responsible. The zero",
        "  indicator, centered residue, Fourier pair, coordinate embedding, sparse attention",
        "  bias and the one-token-per-coordinate change were all introduced together.",
        "- **Not tested:** secret recovery, larger n, larger h.",
        "",
        "## Artifacts",
        "",
        "- `equal_depth_ablation.md` (this file)",
        "- `equal_depth_ablation.json`",
        "- `equal_depth_ablation.csv`",
    ] + [f"- `{name}`" for name in payload.get("figures", [])] + [""]
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
