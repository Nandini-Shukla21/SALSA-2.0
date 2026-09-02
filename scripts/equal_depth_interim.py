"""Phase 17 INTERIM report: equal-depth V1 vs V2, incomplete seed sweep.

Read-only.  Nothing is trained, no experiment is run, no evaluation code is
touched and no result is deleted.  The seed sweep was stopped by the user after
three of five V2 seeds, so this is a **pilot / interim** report and is labelled
as one throughout.

Data provenance, stated because it is not uniform
-------------------------------------------------
* **V1 GatedUT T_e=2** -- five seeds, read from
  ``results/v1_v2_audit/v1_v2_comparison.json``.  That audit snapshot is the
  surviving record of the phase-16 runs, whose directories were emptied
  afterwards.  V1 was not retrained for this phase.
* **V2 NACT T_e=2** -- the three seeds that finished (0, 42, 123), read from
  each run's own ``artifacts/summary.json``.  Seed 456 was interrupted mid-run
  and is excluded; seed 789 never started.
* **V2 sparsity** -- the phase-12 evaluation completed and printed its full
  table, and its four figures were written, but the run then failed before
  writing its ``.json``/``.csv``/``.md``.  The numbers below are transcribed
  from that completed run's console output and are marked as transcribed.  The
  evaluation was **not** re-run, and the phase-12 script was not modified.
"""

import csv
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

RESULTS = REPO_ROOT / "results"
AUDIT = RESULTS / "v1_v2_audit" / "v1_v2_comparison.json"
V2_ROOT = RESULTS / "equal_depth_ablation" / "v2_te2" / "nact_n12_h2_te2"
V1_SPARSITY = RESULTS / "recovery_generalization" / "recovery_generalization.json"
OUTPUT_DIR = RESULTS / "equal_depth_ablation"

COMPLETED_V2_SEEDS = (0, 42, 123)
INTERRUPTED = {"seed_456": "interrupted mid-run; excluded",
               "seed_789": "never started"}

CHANCE = {"valid_token_accuracy": 0.44621, "valid_exact_accuracy": 0.00398,
          "valid_acc_tau": 0.19287}
MARGINAL_LOSS_NATS, IRREDUCIBLE_LOSS_NATS = 1.86498, 0.83920

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

#: TRANSCRIBED from the completed phase-12 evaluation of the V2 T_e=2 seed-0
#: checkpoint (same K sweep, same seed, same test vectors as V1's phase-12 run).
#: The run's own JSON was not written; its four figures were.  Not re-run.
V2_SPARSITY_TRANSCRIBED = {
    "provenance": ("transcribed from the completed run's console output; the run "
                   "wrote its four figures then failed before writing its JSON/CSV/MD. "
                   "Evaluation not re-run; phase-12 script not modified."),
    "checkpoint": "results/equal_depth_ablation/v2_te2/nact_n12_h2_te2/seed_0/checkpoints/best.pt",
    "config_fingerprint": "be1c85196c6d267c",
    "k_sweep": [1, 31, 63, 94, 125, 126, 157, 188, 220, 250],
    "samples_per_level": 2048,
    "rows": [
        {"input_type": "training_distribution", "nnz": 12, "decode_validity": 1.000,
         "acc_tau": 0.9888, "exact_accuracy": 0.1104, "unique_outputs": 162,
         "output_entropy_nats": 4.671, "acc_tau_best_constant_predictor": 0.2217,
         "acc_tau_minus_best_constant": 0.7671},
        {"input_type": "sparse_random", "nnz": 12, "decode_validity": 1.000,
         "acc_tau": 0.9878, "exact_accuracy": 0.1108, "unique_outputs": 163,
         "output_entropy_nats": 4.680, "acc_tau_best_constant_predictor": 0.2236,
         "acc_tau_minus_best_constant": 0.7642},
        {"input_type": "sparse_random", "nnz": 10, "decode_validity": 1.000,
         "acc_tau": 0.9756, "exact_accuracy": 0.1074, "unique_outputs": 161,
         "output_entropy_nats": 4.637, "acc_tau_best_constant_predictor": 0.2354,
         "acc_tau_minus_best_constant": 0.7402},
        {"input_type": "sparse_random", "nnz": 8, "decode_validity": 1.000,
         "acc_tau": 0.9438, "exact_accuracy": 0.1084, "unique_outputs": 152,
         "output_entropy_nats": 4.556, "acc_tau_best_constant_predictor": 0.2207,
         "acc_tau_minus_best_constant": 0.7231},
        {"input_type": "sparse_random", "nnz": 6, "decode_validity": 1.000,
         "acc_tau": 0.8696, "exact_accuracy": 0.1040, "unique_outputs": 136,
         "output_entropy_nats": 4.181, "acc_tau_best_constant_predictor": 0.2817,
         "acc_tau_minus_best_constant": 0.5879},
        {"input_type": "sparse_random", "nnz": 4, "decode_validity": 1.000,
         "acc_tau": 0.7773, "exact_accuracy": 0.1074, "unique_outputs": 122,
         "output_entropy_nats": 3.617, "acc_tau_best_constant_predictor": 0.3530,
         "acc_tau_minus_best_constant": 0.4243},
        {"input_type": "sparse_random", "nnz": 2, "decode_validity": 1.000,
         "acc_tau": 0.6704, "exact_accuracy": 0.0972, "unique_outputs": 108,
         "output_entropy_nats": 2.537, "acc_tau_best_constant_predictor": 0.4546,
         "acc_tau_minus_best_constant": 0.2158},
        {"input_type": "sparse_random", "nnz": 1, "decode_validity": 1.000,
         "acc_tau": 0.6279, "exact_accuracy": 0.0903, "unique_outputs": 83,
         "output_entropy_nats": 1.576, "acc_tau_best_constant_predictor": 0.5083,
         "acc_tau_minus_best_constant": 0.1196},
        {"input_type": "probe_K_times_e_i", "nnz": 1, "decode_validity": 1.000,
         "acc_tau": 0.6000, "exact_accuracy": 0.0833, "unique_outputs": 15,
         "output_entropy_nats": 1.418, "acc_tau_best_constant_predictor": 0.4833,
         "acc_tau_minus_best_constant": 0.1167},
    ],
}


def load_v1() -> Dict[int, Dict[str, Any]]:
    """V1 GatedUT T_e=2, five seeds, from the preserved audit snapshot."""
    audit = json.loads(AUDIT.read_text("utf-8"))["per_seed"]
    return {audit[k]["V1"]["seed"]: audit[k]["V1"] for k in audit}


def load_v2_te4() -> Dict[int, Dict[str, Any]]:
    """V2 NACT at T_e=4, five seeds, from the same snapshot."""
    audit = json.loads(AUDIT.read_text("utf-8"))["per_seed"]
    return {audit[k]["V2"]["seed"]: audit[k]["V2"] for k in audit}


def load_v2_te2() -> Dict[int, Dict[str, Any]]:
    """The V2 T_e=2 runs that actually completed."""
    runs: Dict[int, Dict[str, Any]] = {}
    for seed in COMPLETED_V2_SEEDS:
        path = V2_ROOT / f"seed_{seed}" / "artifacts" / "summary.json"
        if not path.is_file():
            continue
        summary = json.loads(path.read_text("utf-8"))
        state, valid = summary["state"], summary["final_validation"]
        row = {
            "seed": summary["seed"], "architecture": summary["architecture"],
            "parameter_count": summary["parameter_count"], "encoder_loops": 2,
            "best_epoch": state["best_epoch"], "samples_seen": state["samples_seen"],
            "elapsed_seconds": state["elapsed_seconds"],
            "samples_per_second": summary["throughput"]["samples_per_second"],
            "tokens_per_second": summary["throughput"]["tokens_per_second"],
            "rss_mb": summary["cpu_memory_mb"]["rss_mb"],
            "config_fingerprint": summary["config_fingerprint"],
        }
        row.update({k: v for k, v in valid.items() if isinstance(v, (int, float))})
        runs[summary["seed"]] = row
    return runs


def spread(values: List[float]) -> Dict[str, float]:
    """Mean, standard deviation and range."""
    return {"mean": statistics.fmean(values),
            "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min": min(values), "max": max(values), "n": len(values)}


def exact_sign_test(differences: List[float]) -> Dict[str, Any]:
    """Two-sided exact sign test, with its resolution floor made explicit."""
    nonzero = [d for d in differences if d != 0]
    n = len(nonzero)
    if n == 0:
        return {"applicable": False}
    positive = sum(1 for d in nonzero if d > 0)
    extreme = min(positive, n - positive)
    tail = sum(math.comb(n, k) for k in range(extreme + 1))
    return {
        "applicable": True, "n_pairs": n, "positive_differences": positive,
        "two_sided_p": min(1.0, 2.0 * tail / (2 ** n)),
        "smallest_attainable_two_sided_p": 2.0 / (2 ** n),
        "unanimous": positive in (0, n),
        "note": (f"With {n} pairs the smallest attainable two-sided p is "
                 f"{2.0 / (2 ** n):.4f}. p < 0.05 is UNREACHABLE at this sample size, "
                 "so unanimity is the strongest available outcome and is reported as "
                 "consistency, not significance."),
    }


def compare(v1: Dict[int, Dict[str, Any]], v2: Dict[int, Dict[str, Any]],
            paired_seeds: List[int]) -> List[Dict[str, Any]]:
    """Per-metric statistics; paired stats only over seeds present in both arms."""
    rows = []
    for key, label, better, tier in METRICS:
        a_all = [v1[s][key] for s in sorted(v1) if key in v1[s]]
        b_all = [v2[s][key] for s in sorted(v2) if key in v2[s]]
        if not a_all or not b_all:
            continue
        sa, sb = spread(a_all), spread(b_all)
        paired = [(v1[s][key], v2[s][key]) for s in paired_seeds
                  if key in v1[s] and key in v2[s]]
        differences = [y - x for x, y in paired]
        oriented = differences if better == "higher" else [-d for d in differences]
        if sa["mean"] == sb["mean"]:
            winner = "TIE"
        elif better == "higher":
            winner = "V2" if sb["mean"] > sa["mean"] else "V1"
        else:
            winner = "V2" if sb["mean"] < sa["mean"] else "V1"
        rows.append({
            "metric": label, "key": key, "tier": tier, "better": better,
            "V1_n": sa["n"], "V1_mean": sa["mean"], "V1_stdev": sa["stdev"],
            "V1_min": sa["min"], "V1_max": sa["max"],
            "V2_n": sb["n"], "V2_mean": sb["mean"], "V2_stdev": sb["stdev"],
            "V2_min": sb["min"], "V2_max": sb["max"],
            "difference_v2_minus_v1": sb["mean"] - sa["mean"],
            "percent_difference": ((sb["mean"] - sa["mean"]) / sa["mean"] * 100.0
                                   if sa["mean"] else None),
            "winner": winner,
            "paired_seeds": paired_seeds,
            "per_seed_differences": dict(zip(paired_seeds, differences)),
            "ranges_overlap": not (sa["max"] < sb["min"] or sb["max"] < sa["min"]),
            "paired_sign_test": exact_sign_test(oriented),
            "chance_level": CHANCE.get(key),
        })
    return rows


def sparsity_rows(path: Path) -> Optional[List[Dict[str, Any]]]:
    """V1's phase-12 rows, from its intact artifact."""
    if not path.is_file():
        return None
    payload = json.loads(path.read_text("utf-8"))
    keep = ("input_type", "nnz", "decode_validity", "acc_tau",
            "acc_tau_best_constant_predictor", "acc_tau_minus_best_constant",
            "exact_accuracy", "unique_outputs", "output_entropy_nats",
            "mean_abs_error_valid")
    return [{k: r.get(k) for k in keep} for r in payload["table"]]


def crossing(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Where the lift over the best input-blind constant changes sign."""
    sparse = [r for r in rows if r["input_type"] == "sparse_random"]
    positive = [r["nnz"] for r in sparse if r["acc_tau_minus_best_constant"] > 0]
    negative = [r["nnz"] for r in sparse if r["acc_tau_minus_best_constant"] <= 0]
    probe = next((r for r in rows if r["input_type"] == "probe_K_times_e_i"), None)
    return {
        "nnz_with_positive_lift": positive,
        "nnz_with_non_positive_lift": negative,
        "crosses_zero_in_measured_range": bool(negative),
        "crossing_between": (f"nnz={min(positive)} and nnz={max(negative)}"
                             if positive and negative else None),
        "lift_at_nnz_1": next((r["acc_tau_minus_best_constant"]
                               for r in sparse if r["nnz"] == 1), None),
        "probe_lift": probe["acc_tau_minus_best_constant"] if probe else None,
        "probe_decode_validity": probe["decode_validity"] if probe else None,
    }


def main() -> int:
    """Assemble the interim report."""
    v1, v2_te4, v2 = load_v1(), load_v2_te4(), load_v2_te2()
    paired = [s for s in COMPLETED_V2_SEEDS if s in v1 and s in v2]

    comparison = compare(v1, v2, paired)
    depth = compare(v2_te4, v2, paired)

    v1_sparse = sparsity_rows(V1_SPARSITY)
    v2_sparse = V2_SPARSITY_TRANSCRIBED["rows"]
    crossings = {"V1_GatedUT_Te2": crossing(v1_sparse) if v1_sparse else None,
                 "V2_NACT_Te2": crossing(v2_sparse)}

    print("=" * 100)
    print("PHASE 17 EQUAL-DEPTH ABLATION - PILOT / INTERIM RESULT   (NOT a 5-seed final result)")
    print("=" * 100)
    print(f"  V1 seeds: {sorted(v1)}   V2 T_e=2 seeds COMPLETED: {sorted(v2)}")
    print(f"  paired seeds used for the paired test: {paired}")
    print(f"  not available: {INTERRUPTED}")
    print()
    print(f"  {'metric':<26}{'V1 mean':>11}{'V1 sd':>9}{'V2 mean':>11}{'V2 sd':>9}"
          f"{'diff':>11}{'win':>5}{'sign p':>9}")
    for row in comparison:
        test = row["paired_sign_test"]
        print(f"  {row['metric']:<26}{row['V1_mean']:>11.4f}{row['V1_stdev']:>9.4f}"
              f"{row['V2_mean']:>11.4f}{row['V2_stdev']:>9.4f}"
              f"{row['difference_v2_minus_v1']:>+11.4f}{row['winner']:>5}"
              f"{test.get('two_sided_p', float('nan')):>9.4f}")

    print("\n  SPARSITY - lift over the best input-blind constant")
    print(f"    {'nnz':>6}{'V1 lift':>11}{'V2 lift':>11}{'V1 valid':>11}{'V2 valid':>11}")
    v1s = [r for r in v1_sparse if r["input_type"] == "sparse_random"]
    v2s = [r for r in v2_sparse if r["input_type"] == "sparse_random"]
    for a, b in zip(v1s, v2s):
        print(f"    {a['nnz']:>6}{a['acc_tau_minus_best_constant']:>+11.4f}"
              f"{b['acc_tau_minus_best_constant']:>+11.4f}"
              f"{a['decode_validity']:>11.3f}{b['decode_validity']:>11.3f}")
    pa = next(r for r in v1_sparse if r["input_type"] == "probe_K_times_e_i")
    pb = next(r for r in v2_sparse if r["input_type"] == "probe_K_times_e_i")
    print(f"    {'K*e_i':>6}{pa['acc_tau_minus_best_constant']:>+11.4f}"
          f"{pb['acc_tau_minus_best_constant']:>+11.4f}"
          f"{pa['decode_validity']:>11.3f}{pb['decode_validity']:>11.3f}")
    print(f"\n    V1 crosses zero: {crossings['V1_GatedUT_Te2']['crosses_zero_in_measured_range']}"
          f" ({crossings['V1_GatedUT_Te2']['crossing_between']})")
    print(f"    V2 crosses zero: {crossings['V2_NACT_Te2']['crosses_zero_in_measured_range']}")

    payload = {
        "title": "Phase 17 Equal-Depth Ablation - Pilot / Interim Result",
        "status": "INTERIM. NOT a 5-seed final result. The seed sweep was stopped by "
                  "the user after 3 of 5 V2 seeds. Read-only: nothing trained here, "
                  "no evaluation re-run, no code modified, no result deleted.",
        "design": {
            "only_configuration_difference": "model.arch",
            "V1_depth": "T_e=2, T_d=2", "V2_depth": "T_e=2, T_d=2",
            "V1_parameters": 4_131_200, "V2_parameters": 4_241_288,
            "parameter_difference": 110_088,
            "sample_budget_per_seed": 100_032,
            "parity_check": ("field-by-field: n, h, q, sigma, structure, encoding, seed, "
                             "optimizer, learning rate, scheduler, batch size, validation "
                             "size, sample budget, evaluation cadence and tau all identical"),
        },
        "seed_status": {
            "V1_GatedUT_Te2_seeds": sorted(v1),
            "V2_NACT_Te2_completed_seeds": sorted(v2),
            "V2_NACT_Te2_unavailable": INTERRUPTED,
            "paired_seeds": paired,
        },
        "chance_baselines": CHANCE,
        "loss_reference_points": {"marginal_only_nats": MARGINAL_LOSS_NATS,
                                  "irreducible_floor_nats": IRREDUCIBLE_LOSS_NATS},
        "per_seed": {"V1_GatedUT_Te2": v1, "V2_NACT_Te2": v2,
                     "V2_NACT_Te4_phase16": v2_te4},
        "comparison": comparison,
        "depth_ablation_Te4_vs_Te2": depth,
        "sparsity": {
            "V1_GatedUT_Te2": v1_sparse,
            "V2_NACT_Te2": V2_SPARSITY_TRANSCRIBED,
            "crossing_points": crossings,
        },
        "throughput_warning": (
            "V1 throughput was recorded in the phase-16 session and V2 T_e=2 in the "
            "phase-17 session, on a machine under different load. They are NOT "
            "comparable. The controlled same-session speed measurement is the phase-15 "
            "forward benchmark in results/nact_v2/architecture_report.json."),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "equal_depth_ablation_interim.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8")

    columns = ["section", "metric", "tier", "V1_n", "V1_mean", "V1_stdev", "V1_min",
               "V1_max", "V2_n", "V2_mean", "V2_stdev", "V2_min", "V2_max",
               "difference_v2_minus_v1", "percent_difference", "winner",
               "sign_test_p", "sign_test_unanimous", "seed", "arm", "value",
               "nnz", "input_type", "V1_lift", "V2_lift",
               "V1_decode_validity", "V2_decode_validity"]
    with (OUTPUT_DIR / "equal_depth_ablation_interim.csv").open(
            "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in comparison:
            writer.writerow({"section": "equal_depth", **row,
                             "sign_test_p": row["paired_sign_test"].get("two_sided_p"),
                             "sign_test_unanimous": row["paired_sign_test"].get("unanimous")})
        for row in depth:
            writer.writerow({"section": "depth_ablation_Te4_vs_Te2", **row,
                             "sign_test_p": row["paired_sign_test"].get("two_sided_p")})
        for arm, runs in (("V1_GatedUT_Te2", v1), ("V2_NACT_Te2", v2)):
            for seed, run in sorted(runs.items()):
                for key, _l, _b, _t in METRICS:
                    if key in run:
                        writer.writerow({"section": "per_seed", "arm": arm,
                                         "seed": seed, "metric": key, "value": run[key]})
        for a, b in zip(v1_sparse, v2_sparse):
            writer.writerow({"section": "sparsity", "input_type": a["input_type"],
                             "nnz": a["nnz"],
                             "V1_lift": a["acc_tau_minus_best_constant"],
                             "V2_lift": b["acc_tau_minus_best_constant"],
                             "V1_decode_validity": a["decode_validity"],
                             "V2_decode_validity": b["decode_validity"]})

    write_markdown(payload, OUTPUT_DIR / "equal_depth_ablation_interim.md")
    for name in ("equal_depth_ablation_interim.md", "equal_depth_ablation_interim.json",
                 "equal_depth_ablation_interim.csv"):
        print(f"\n  wrote {OUTPUT_DIR / name}" if name.endswith(".md") else
              f"  wrote {OUTPUT_DIR / name}")
    return 0


def write_markdown(payload: Dict[str, Any], path: Path) -> None:
    """Render the interim report."""
    design, seeds = payload["design"], payload["seed_status"]
    comparison, sparsity = payload["comparison"], payload["sparsity"]
    crossings = sparsity["crossing_points"]
    v1s = [r for r in sparsity["V1_GatedUT_Te2"] if r["input_type"] == "sparse_random"]
    v2s = [r for r in sparsity["V2_NACT_Te2"]["rows"] if r["input_type"] == "sparse_random"]
    pa = next(r for r in sparsity["V1_GatedUT_Te2"] if r["input_type"] == "probe_K_times_e_i")
    pb = next(r for r in sparsity["V2_NACT_Te2"]["rows"] if r["input_type"] == "probe_K_times_e_i")

    lines = [
        "# Phase 17 Equal-Depth Ablation — Pilot / Interim Result",
        "",
        "> **This is NOT a 5-seed final result.** The seed sweep was stopped after three",
        "> of five V2 seeds. Everything below is a pilot reading of partial data.",
        "",
        "Read-only: nothing was trained to produce this report, no evaluation was re-run,",
        "no code was modified and no result was deleted.",
        "",
        "## What was being asked",
        "",
        "Phase 16 showed a large V2 advantage, but V2 ran at `encoder_loops=4` against",
        "V1's 2, confounding architecture with effective depth. This phase holds depth",
        "equal so `model.arch` is the only thing that varies.",
        "",
        "## 1. Design and parity",
        "",
        "| | V1 GatedUT | V2 NACT |",
        "|---|---|---|",
        f"| effective depth | {design['V1_depth']} | {design['V2_depth']} |",
        f"| parameters | {design['V1_parameters']:,} | {design['V2_parameters']:,} "
        f"(+{design['parameter_difference']:,}) |",
        f"| samples/seed | {design['sample_budget_per_seed']:,} | "
        f"{design['sample_budget_per_seed']:,} |",
        "",
        f"A field-by-field parity check confirmed **`model.arch` is the only configuration",
        f"difference** — {design['parity_check']}. Both arms share the same secret and data stream.",
        "",
        "V1 was **not** retrained. Its five seeds come from the preserved phase-16 audit",
        "snapshot (`results/v1_v2_audit/v1_v2_comparison.json`), the surviving record after",
        "the phase-16 run directories were emptied.",
        "",
        "## 2. Seed status — the central limitation",
        "",
        "| arm | seeds available |",
        "|---|---|",
        f"| V1 GatedUT T_e=2 | {seeds['V1_GatedUT_Te2_seeds']} (5) |",
        f"| **V2 NACT T_e=2** | **{seeds['V2_NACT_Te2_completed_seeds']} (3 of 5)** |",
        "",
        "Seed 456 was interrupted mid-run and is excluded; seed 789 never started. Neither",
        "was deleted. Paired statistics use only the "
        f"{len(seeds['paired_seeds'])} seeds present in both arms: {seeds['paired_seeds']}.",
        "",
        "## 3. Learning metrics",
        "",
        "| Metric | V1 GatedUT T_e=2 (5 seeds) | V2 NACT T_e=2 (3 seeds) | Difference | Winner |",
        "|---|---:|---:|---:|:---:|",
    ]
    for row in comparison:
        lines.append(
            f"| {row['metric']} | {row['V1_mean']:,.4f} ± {row['V1_stdev']:.4f} | "
            f"{row['V2_mean']:,.4f} ± {row['V2_stdev']:.4f} | "
            f"{row['difference_v2_minus_v1']:+,.4f} | **{row['winner']}** |")
    lines += [
        "",
        f"Chance: acc_tau {payload['chance_baselines']['valid_acc_tau']:.5f}, exact "
        f"{payload['chance_baselines']['valid_exact_accuracy']:.5f}, token "
        f"{payload['chance_baselines']['valid_token_accuracy']:.5f}. A model that learned "
        f"only the output marginals scores {payload['loss_reference_points']['marginal_only_nats']:.4f} "
        f"nats; the irreducible floor is {payload['loss_reference_points']['irreducible_floor_nats']:.4f}.",
        "",
        "**Reading the loss.** V1's mean sits essentially at the marginals-only baseline —",
        "i.e. at this budget V1 largely has not learned secret-related structure. V2's mean",
        "sits well below it and closer to the error floor.",
        "",
        "**Token accuracy must not be read alone.** V1's mean is near the 0.4462 chance",
        "level; V2 clears it. That is why acc_tau and exact accuracy are the primary metrics.",
        "",
        "### Per-seed detail",
        "",
        "| seed | V1 loss | V2 loss | V1 acc_tau | V2 acc_tau | V1 exact | V2 exact |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    v1r, v2r = payload["per_seed"]["V1_GatedUT_Te2"], payload["per_seed"]["V2_NACT_Te2"]
    for seed in sorted(v1r, key=int):
        a = v1r[seed]
        b = v2r.get(seed) or v2r.get(str(seed))
        if b is None:
            lines.append(f"| {seed} | {a['valid_loss']:.4f} | — | {a['valid_acc_tau']:.4f} "
                         f"| — | {a['valid_exact_accuracy']:.4f} | — |")
        else:
            lines.append(f"| {seed} | {a['valid_loss']:.4f} | {b['valid_loss']:.4f} | "
                         f"{a['valid_acc_tau']:.4f} | {b['valid_acc_tau']:.4f} | "
                         f"{a['valid_exact_accuracy']:.4f} | {b['valid_exact_accuracy']:.4f} |")

    lines += [
        "",
        "### Paired test, and why it cannot conclude much",
        "",
        "| Metric | paired n | sign-test p | unanimous |",
        "|---|---:|---:|:---:|",
    ]
    for row in comparison:
        if row["tier"] in ("primary", "secondary"):
            t = row["paired_sign_test"]
            lines.append(f"| {row['metric']} | {t.get('n_pairs', 0)} | "
                         f"{t.get('two_sided_p', float('nan')):.4f} | "
                         f"{'yes' if t.get('unanimous') else 'no'} |")
    lines += [
        "",
        "The arms share seeds, data stream and secret, so runs pair naturally and an exact",
        "two-sided sign test is the appropriate test; nothing justifies assuming normality.",
        "**With 3 pairs the smallest attainable two-sided p is 0.25.** No statistical",
        "significance is claimed and none is reachable. Unanimity across three seeds is",
        "reported as consistency of direction, nothing more.",
        "",
        "## 4. Was phase 16's advantage just the extra depth?",
        "",
        "| primary metric | V2 at T_e=4 (5 seeds) | V2 at T_e=2 (3 seeds) | change |",
        "|---|---:|---:|---:|",
    ]
    for row in payload["depth_ablation_Te4_vs_Te2"]:
        if row["tier"] == "primary":
            lines.append(f"| {row['metric']} | {row['V1_mean']:.4f} | {row['V2_mean']:.4f} | "
                         f"{row['difference_v2_minus_v1']:+.4f} |")
    lines += [
        "",
        "**Halving NACT's encoder depth barely changes it.** On the seeds available, the",
        "phase-16 advantage does not appear to have come from the extra loops. This is the",
        "single most useful thing the interim data says, and it is what the phase was for.",
        "",
        "## 5. Sparsity generalization",
        "",
        f"*{sparsity['V2_NACT_Te2']['provenance']}*",
        "",
        "Same phase-12 procedure, same K sweep "
        f"{sparsity['V2_NACT_Te2']['k_sweep']}, same seed, same test vectors, same",
        "metrics, same decoder. Evaluated on the **seed-0** V2 T_e=2 checkpoint.",
        "",
        "| nnz | V1 lift | V2 lift | V1 acc_tau | V2 acc_tau | V1 decode validity | V2 decode validity |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for a, b in zip(v1s, v2s):
        lines.append(
            f"| {a['nnz']} | {a['acc_tau_minus_best_constant']:+.4f} | "
            f"**{b['acc_tau_minus_best_constant']:+.4f}** | {a['acc_tau']:.4f} | "
            f"{b['acc_tau']:.4f} | {a['decode_validity']:.3f} | {b['decode_validity']:.3f} |")
    lines += [
        f"| **K·e_i probes** | {pa['acc_tau_minus_best_constant']:+.4f} | "
        f"**{pb['acc_tau_minus_best_constant']:+.4f}** | {pa['acc_tau']:.4f} | "
        f"{pb['acc_tau']:.4f} | {pa['decode_validity']:.3f} | {pb['decode_validity']:.3f} |",
        "",
        "**Lift** is acc_tau minus what the best input-blind constant scores on the same",
        "targets. Lift ≤ 0 means the model's output carries no usable information about its",
        "input. This correction matters because sparser inputs make a constant predictor",
        "*better*, so raw acc_tau alone would understate the collapse.",
        "",
        "### The headline",
        "",
        f"- **V1** crosses zero {crossings['V1_GatedUT_Te2']['crossing_between']}, falling to "
        f"**{crossings['V1_GatedUT_Te2']['lift_at_nnz_1']:+.4f}** at nnz=1 and "
        f"**{crossings['V1_GatedUT_Te2']['probe_lift']:+.4f}** on the probes, with probe",
        f"  decode validity {pa['decode_validity']:.3f} — most probe outputs were unreadable.",
        f"- **V2 never crosses zero in the measured range.** Lift stays positive at every",
        f"  level: **{crossings['V2_NACT_Te2']['lift_at_nnz_1']:+.4f}** at nnz=1 and "
        f"**{crossings['V2_NACT_Te2']['probe_lift']:+.4f}** on the probes, with probe",
        f"  decode validity **{pb['decode_validity']:.3f}** — every output readable.",
        "",
        "**This strongly suggests V2 removes the severe sparse-input collapse seen in V1.**",
        "The boundary did not move to nnz=6, 4, 2 or 1 — on this evidence it moved off the",
        "measured range entirely, and V1's unreadable-output failure on probes is absent.",
        "",
        "**But it is not yet a multi-seed statistical conclusion.** It rests on **one**",
        "checkpoint (V2 seed 0) against **one** V1 checkpoint. The sparsity evaluation was",
        "not repeated across seeds for either arm, so seed-to-seed variability of the lift",
        "curve is entirely unmeasured — and V1's *learning* metrics vary a lot by seed",
        "(acc_tau 0.10–0.36), which is reason to expect its sparsity curve might vary too.",
        "",
        "**It is also not a secret-recovery result.** Recovery was not run and is out of",
        "scope. What is measured is prediction fidelity on sparse and probe-shaped inputs.",
        "",
        "## 6. CPU efficiency",
        "",
        f"{payload['throughput_warning']}",
        "",
        "| | V1 GatedUT T_e=2 | V2 NACT T_e=2 |",
        "|---|---:|---:|",
    ]
    for key, label in (("samples_per_second", "Samples/sec"),
                       ("tokens_per_second", "Tokens/sec"),
                       ("elapsed_seconds", "Wall-clock (s)"),
                       ("rss_mb", "CPU RSS (MiB)")):
        row = next((r for r in comparison if r["key"] == key), None)
        if row:
            lines.append(f"| {label} | {row['V1_mean']:,.1f} | {row['V2_mean']:,.1f} |")
    lines += [
        "",
        "**Do not read the throughput rows as an architecture comparison.** The controlled,",
        "same-session measurement is the phase-15 forward benchmark: V2 at T_e=2 runs at",
        "**~1.95× V1's throughput at n=128** and ~1.64× at n=30, because its encoder",
        "sequence is `n+2` rather than `2n+2`. Memory is measured within each run and is",
        "the more trustworthy column here.",
        "",
        "## 7. Parameter counts",
        "",
        "| | parameters | fp32 |",
        "|---|---:|---:|",
        f"| V1 Salsa2-GatedUT | {design['V1_parameters']:,} | "
        f"{design['V1_parameters'] * 4 / 2 ** 20:.2f} MiB |",
        f"| V2 Salsa2-NACT | {design['V2_parameters']:,} | "
        f"{design['V2_parameters'] * 4 / 2 ** 20:.2f} MiB |",
        f"| difference | +{design['parameter_difference']:,} (+2.7%) | |",
        "",
        "The +110,088 is entirely the input front end (digit embeddings 82,944; coordinate",
        "embedding 65,536; numerical projection 2,560; special embeddings 2,048; zero vector",
        "512; sparse attention bias 8; minus V1's 43,520 token embedding). Verified three",
        "ways — actual, component breakdown, analytical — with nothing unclassified. Both",
        "models sit inside the 4–5M budget with no padding parameters.",
        "",
        "## 8. Limitations",
        "",
        "1. **Three of five V2 seeds.** Not a 5-seed result and must not be cited as one.",
        "2. **No reachable significance.** 3 pairs floor the two-sided sign test at p=0.25.",
        "3. **Sparsity is single-checkpoint on both sides.** One V2 seed vs one V1 seed;",
        "   seed variability of the lift curve is unmeasured.",
        "4. **The V2 sparsity artifact is incomplete.** The evaluation ran and printed its",
        "   full table and wrote its four figures, but crashed before writing its JSON/CSV/MD.",
        "   The numbers here are transcribed from that run's output. Re-running it would",
        "   restore the artifacts; that was not done because no further runs were authorised.",
        "5. **Throughput is not comparable across arms** (different sessions).",
        "6. **Component attribution is impossible.** The zero indicator, centered residue,",
        "   Fourier pair, coordinate embedding, sparse attention bias and one-token-per-",
        "   coordinate change were introduced together.",
        "7. **Nothing cryptographic.** n=12, h=2 has C(12,2)=66 secrets and is a diagnostic",
        "   positive control, as in phases 7 and 16.",
        "",
        "## Interim classification",
        "",
        "On the available evidence the result points to **A — NACT improves learning at",
        "equal depth AND improves sparse generalization** — but with 3 of 5 seeds and a",
        "single-checkpoint sparsity evaluation this is a **provisional** classification, not",
        "a confirmed one. Completing seeds 456 and 789, and evaluating sparsity on more than",
        "one checkpoint per arm, are what would settle it.",
        "",
        "## Artifacts",
        "",
        "- `equal_depth_ablation_interim.md` (this file)",
        "- `equal_depth_ablation_interim.json`",
        "- `equal_depth_ablation_interim.csv`",
        "- `sparsity_v2_te2/0{1,2,3,4}_*.png` — the four figures the V2 sparsity run wrote",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
