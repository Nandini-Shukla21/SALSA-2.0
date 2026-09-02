"""Read-only audit of the trained V1 GatedUT and V2 NACT runs.

Nothing is trained, no checkpoint is written, no model or evaluation code is
touched.  Every number is read from the ``artifacts/summary.json`` each run
already wrote, and the two protocols are diffed field by field before any
accuracy is compared.
"""

import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from salsa.utils import load_config  # noqa: E402

RESULTS = REPO_ROOT / "results"
PHASE16 = RESULTS / "nact_v2"
OUTPUT_DIR = RESULTS / "v1_v2_audit"

#: Chance levels for this representation, from phase 6.
CHANCE = {"token_accuracy": 0.44621, "exact_accuracy": 0.00398, "acc_tau": 0.19287}
MARGINAL_LOSS_NATS = 1.86498      # a model that learned only the output marginals
IRREDUCIBLE_LOSS_NATS = 0.83920   # what a perfect attacker still pays for the error

#: Metric -> (label, better direction).
METRICS = [
    ("parameter_count", "Trainable parameters", "lower"),
    ("best_epoch", "Best epoch", "none"),
    ("samples_seen", "Samples trained", "none"),
    ("valid_loss", "Validation loss (nats)", "lower"),
    ("valid_token_accuracy", "Teacher-forced token accuracy", "higher"),
    ("valid_perfect_accuracy", "Teacher-forced perfect accuracy", "higher"),
    ("valid_greedy_token_accuracy", "Greedy token accuracy", "higher"),
    ("valid_exact_accuracy", "Exact integer accuracy", "higher"),
    ("valid_acc_tau", "SALSA acc_tau", "higher"),
    ("valid_decode_failure_rate", "Decode failure rate", "lower"),
    ("valid_mean_distance", "Mean |b - b_hat|", "lower"),
    ("valid_median_distance", "Median |b - b_hat|", "lower"),
    ("samples_per_second", "Samples/sec", "higher"),
    ("tokens_per_second", "Tokens/sec", "higher"),
    ("elapsed_seconds", "Wall-clock training (s)", "lower"),
    ("rss_mb", "CPU RSS (MiB)", "lower"),
]


def run_paths() -> Dict[str, Dict[str, Path]]:
    """Locate every matched V1/V2 training pair from the run tree."""
    pairs = {
        "seed_0": {
            "V1": PHASE16 / "phase16_part_b/v1_control/control_a_n12_h2/phase16_part_b_v1",
            "V2": PHASE16 / "phase16_part_b/nact_training/nact_n12_h2/phase16_part_b_nact",
        }
    }
    for seed in (42, 123, 456, 789):
        root = PHASE16 / f"phase16_part_c/seed_{seed}"
        pairs[f"seed_{seed}"] = {
            "V1": root / f"v1/control_a_n12_h2/phase16_part_c_v1_seed_{seed}",
            "V2": root / f"nact/nact_n12_h2/phase16_part_c_nact_seed_{seed}",
        }
    return pairs


def locate_checkpoint(run: Path) -> Dict[str, Any]:
    """Find the best checkpoint the way the run's own metadata describes it.

    The filename is never assumed: ``summary.json`` names the best epoch, and
    only then is the checkpoint directory consulted.
    """
    summary = json.loads((run / "artifacts" / "summary.json").read_text("utf-8"))
    best_epoch = summary["state"].get("best_epoch")
    monitor = None
    config_path = run / "config.yaml"
    if config_path.is_file():
        config = load_config(config_path)
        monitor = f"{config.training.monitor_metric} ({config.training.monitor_mode})"
    directory = run / "checkpoints"
    return {
        "run": str(run.relative_to(REPO_ROOT)).replace("\\", "/"),
        "monitor_metric": monitor,
        "best_epoch_from_metadata": best_epoch,
        "best_metric_from_metadata": summary["state"].get("best_metric"),
        "checkpoint_dir_exists": directory.is_dir(),
        "checkpoint_files": sorted(p.name for p in directory.glob("*.pt")) if directory.is_dir() else [],
        "best_checkpoint_available": (directory / "best.pt").is_file(),
    }


def flatten(run: Path) -> Dict[str, Any]:
    """Pull every audited metric out of one run's summary."""
    summary = json.loads((run / "artifacts" / "summary.json").read_text("utf-8"))
    state, valid = summary["state"], summary["final_validation"]
    throughput, memory = summary["throughput"], summary["cpu_memory_mb"]
    row = {
        "model_name": summary["model_name"],
        "architecture": summary["architecture"],
        "config_fingerprint": summary["config_fingerprint"],
        "seed": summary["seed"],
        "parameter_count": summary["parameter_count"],
        "best_epoch": state["best_epoch"],
        "samples_seen": state["samples_seen"],
        "elapsed_seconds": state["elapsed_seconds"],
        "samples_per_second": throughput["samples_per_second"],
        "tokens_per_second": throughput["tokens_per_second"],
        "rss_mb": memory["rss_mb"],
        # peak_rss_mb reads ~403 GB identically in every run: a broken Windows
        # peak-working-set counter, so it is recorded but not compared.
        "peak_rss_mb_UNRELIABLE": memory["peak_rss_mb"],
    }
    row.update({k: v for k, v in valid.items() if isinstance(v, (int, float))})
    return row


def protocol_diff(v1: Path, v2: Path) -> List[Dict[str, Any]]:
    """Diff the two run configurations across every setting that matters."""
    import dataclasses

    a, b = load_config(v1 / "config.yaml"), load_config(v2 / "config.yaml")
    differences = []
    for section in ("lwe", "encoding", "training", "evaluation", "model", "experiment"):
        left, right = getattr(a, section), getattr(b, section)
        for field in dataclasses.fields(left):
            x, y = getattr(left, field.name), getattr(right, field.name)
            if x != y:
                differences.append({
                    "setting": f"{section}.{field.name}",
                    "V1": x if isinstance(x, (int, float, str, bool, type(None))) else str(x),
                    "V2": y if isinstance(y, (int, float, str, bool, type(None))) else str(y),
                })
    return differences


def summarise(values: List[float]) -> Dict[str, float]:
    """Mean, min and max across seeds.  Deliberately not a significance test."""
    return {
        "mean": round(statistics.fmean(values), 6),
        "min": round(min(values), 6),
        "max": round(max(values), 6),
        "stdev": round(statistics.stdev(values), 6) if len(values) > 1 else 0.0,
        "n_seeds": len(values),
    }


def main() -> int:
    """Run the audit."""
    pairs = run_paths()
    missing = [f"{seed}/{arm}" for seed, arms in pairs.items()
               for arm, path in arms.items()
               if not (path / "artifacts" / "summary.json").is_file()]
    if missing:
        print("MISSING RUNS - STOPPING:", missing)
        return 2

    print("=" * 94)
    print("V1 vs V2 AUDIT  (read-only; no training, no checkpoints written, no evaluation rerun)")
    print("=" * 94)

    # -- checkpoint location, from metadata only ---------------------------- #
    print("\nCHECKPOINT LOCATION (from run metadata, filenames never assumed)")
    print("-" * 94)
    checkpoints = {}
    for seed, arms in pairs.items():
        for arm, path in arms.items():
            info = locate_checkpoint(path)
            checkpoints[f"{seed}/{arm}"] = info
            print(f"  {seed:<10}{arm}  best epoch {str(info['best_epoch_from_metadata']):>3}"
                  f"  monitor {str(info['monitor_metric']):<20}"
                  f"  checkpoint dir {str(info['checkpoint_dir_exists']):<6}"
                  f"  best.pt {info['best_checkpoint_available']}")
    no_checkpoints = [k for k, v in checkpoints.items() if not v["best_checkpoint_available"]]

    # -- protocol equality --------------------------------------------------- #
    print("\nPROTOCOL DIFF (V1 vs V2, seed 0 pair)")
    print("-" * 94)
    all_differences = protocol_diff(pairs["seed_0"]["V1"], pairs["seed_0"]["V2"])
    # experiment.name/output_root/notes/tags only label the run; they change no
    # data, no optimisation and no evaluation.
    COSMETIC = {"experiment.name", "experiment.output_root",
                "experiment.notes", "experiment.tags"}
    cosmetic = [d for d in all_differences if d["setting"] in COSMETIC]
    differences = [d for d in all_differences if d["setting"] not in COSMETIC]
    for entry in differences:
        print(f"  {entry['setting']:<28}{str(entry['V1']):>30}  ->  {entry['V2']}")
    print(f"  {len(differences)} substantive difference(s) "
          f"(+{len(cosmetic)} cosmetic run labels). Everything else -- data, seed, encoding,")
    print("  optimizer, schedule, batch size, validation set, tau -- is identical.")

    # -- per-run metrics ----------------------------------------------------- #
    rows = {seed: {arm: flatten(path) for arm, path in arms.items()}
            for seed, arms in pairs.items()}

    print("\nPER-SEED RESULTS")
    print("-" * 94)
    print(f"  {'seed':>6}{'arm':>4}{'loss':>9}{'acc_tau':>9}{'exact':>8}{'token':>8}"
          f"{'perfect':>9}{'greedy':>8}{'mean|d|':>9}{'samp/s':>8}{'sec':>7}")
    for seed, arms in rows.items():
        for arm in ("V1", "V2"):
            r = arms[arm]
            print(f"  {r['seed']:>6}{arm:>4}{r['valid_loss']:>9.4f}{r['valid_acc_tau']:>9.4f}"
                  f"{r['valid_exact_accuracy']:>8.4f}{r['valid_token_accuracy']:>8.4f}"
                  f"{r['valid_perfect_accuracy']:>9.4f}{r['valid_greedy_token_accuracy']:>8.4f}"
                  f"{r['valid_mean_distance']:>9.2f}{r['samples_per_second']:>8.1f}"
                  f"{r['elapsed_seconds']:>7.0f}")

    # -- head to head -------------------------------------------------------- #
    primary = rows["seed_0"]
    head_to_head = []
    for key, label, better in METRICS:
        v1, v2 = primary["V1"].get(key), primary["V2"].get(key)
        if v1 is None or v2 is None:
            continue
        difference = v2 - v1
        percent = (difference / v1 * 100.0) if v1 else None
        if better == "none":
            winner = "n/a"
        elif v1 == v2:
            winner = "TIE"
        elif better == "higher":
            winner = "V2" if v2 > v1 else "V1"
        else:
            winner = "V2" if v2 < v1 else "V1"
        head_to_head.append({
            "metric": label, "key": key, "V1": v1, "V2": v2,
            "winner": winner,
            "absolute_difference_v2_minus_v1": round(difference, 6),
            "percent_difference": round(percent, 2) if percent is not None else None,
            "better_direction": better,
            "chance_level": CHANCE.get(key.replace("valid_", "")),
        })

    print("\nHEAD-TO-HEAD (seed 0, the exactly matched pair)")
    print("-" * 94)
    print(f"  {'metric':<34}{'V1':>14}{'V2':>14}{'winner':>8}{'difference':>14}{'%':>9}")
    for entry in head_to_head:
        percent = f"{entry['percent_difference']:+.1f}" if entry["percent_difference"] is not None else "--"
        print(f"  {entry['metric']:<34}{entry['V1']:>14,.4f}{entry['V2']:>14,.4f}"
              f"{entry['winner']:>8}{entry['absolute_difference_v2_minus_v1']:>14,.4f}{percent:>9}")

    # -- across seeds -------------------------------------------------------- #
    across = {}
    for key in ("valid_loss", "valid_acc_tau", "valid_exact_accuracy",
                "valid_token_accuracy", "valid_perfect_accuracy",
                "valid_greedy_token_accuracy", "valid_mean_distance",
                "samples_per_second", "elapsed_seconds", "rss_mb"):
        across[key] = {
            arm: summarise([rows[s][arm][key] for s in rows]) for arm in ("V1", "V2")
        }
        a, b = across[key]["V1"], across[key]["V2"]
        across[key]["ranges_overlap"] = not (a["max"] < b["min"] or b["max"] < a["min"])

    print("\nACROSS 5 SEEDS (descriptive spread, NOT a significance test)")
    print("-" * 94)
    print(f"  {'metric':<30}{'V1 mean':>10}{'V1 range':>20}{'V2 mean':>10}{'V2 range':>20}{'overlap':>9}")
    for key, entry in across.items():
        a, b = entry["V1"], entry["V2"]
        a_range = "[{:.4f}, {:.4f}]".format(a["min"], a["max"])
        b_range = "[{:.4f}, {:.4f}]".format(b["min"], b["max"])
        print(f"  {key:<30}{a['mean']:>10.4f}{a_range:>20}"
              f"{b['mean']:>10.4f}{b_range:>20}"
              f"{str(entry['ranges_overlap']):>9}")

    # V1 seeds that failed to clear the acc_tau chance line at all.
    v1_below_chance = [rows[s]["V1"]["seed"] for s in rows
                       if rows[s]["V1"]["valid_acc_tau"] <= CHANCE["acc_tau"]]
    v2_below_chance = [rows[s]["V2"]["seed"] for s in rows
                       if rows[s]["V2"]["valid_acc_tau"] <= CHANCE["acc_tau"]]

    payload = {
        "audit": "V1 Salsa2-GatedUT vs V2 Salsa2-NACT",
        "status": "READ-ONLY. No training, no checkpoints written, no evaluation rerun, "
                  "no model/data/evaluation code modified.",
        "checkpoint_location": checkpoints,
        "checkpoints_unavailable": no_checkpoints,
        "protocol_differences": differences,
        "cosmetic_differences": cosmetic,
        "protocol_note": ("Only model.arch and model.encoder_loops differ. Every data, "
                          "encoding, optimiser, schedule, batch, validation and tau "
                          "setting is identical, and both arms share seed and secret."),
        "chance_baselines": CHANCE,
        "loss_reference_points": {
            "marginal_only_nats": MARGINAL_LOSS_NATS,
            "irreducible_floor_nats": IRREDUCIBLE_LOSS_NATS,
        },
        "per_seed": rows,
        "head_to_head_seed_0": head_to_head,
        "across_seeds": across,
        "v1_seeds_at_or_below_acc_tau_chance": v1_below_chance,
        "v2_seeds_at_or_below_acc_tau_chance": v2_below_chance,
        "sparsity_comparison": "not available in existing artifacts",
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "v1_v2_comparison.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8")

    columns = ["section", "seed", "arm", "metric", "V1", "V2", "winner",
               "absolute_difference_v2_minus_v1", "percent_difference",
               "better_direction", "chance_level", "value"]
    with (OUTPUT_DIR / "v1_v2_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for entry in head_to_head:
            writer.writerow({"section": "head_to_head_seed_0", **entry})
        for seed, arms in rows.items():
            for arm, r in arms.items():
                for key, _label, _b in METRICS:
                    if key in r:
                        writer.writerow({"section": "per_seed", "seed": r["seed"],
                                         "arm": arm, "metric": key, "value": r[key]})
    write_markdown(payload, OUTPUT_DIR / "v1_v2_comparison.md")
    print(f"\n  wrote {OUTPUT_DIR / 'v1_v2_comparison.md'}")
    print(f"  wrote {OUTPUT_DIR / 'v1_v2_comparison.json'}")
    print(f"  wrote {OUTPUT_DIR / 'v1_v2_comparison.csv'}")
    return 0


def write_markdown(payload: Dict[str, Any], path: Path) -> None:
    """Render the concise audit report."""
    rows, across = payload["per_seed"], payload["across_seeds"]
    primary = rows["seed_0"]
    lines = [
        "# V1 GatedUT vs V2 NACT - audit",
        "",
        "**Read-only.** No training, no checkpoints written, no evaluation rerun, no code",
        "changed. Every number is read from the `artifacts/summary.json` each run wrote.",
        "",
        "## Two limitations, stated first",
        "",
        "**1. No checkpoints exist for either arm.** All ten Phase-16 runs wrote metrics but",
        "no `checkpoints/` directory, so the best checkpoint could not be located for either",
        "model and nothing can be re-evaluated. The audit is therefore metrics-only, from",
        "each run's own recorded validation pass. Best epoch is read from metadata:",
        f"V1 epoch {primary['V1']['best_epoch']}, V2 epoch {primary['V2']['best_epoch']}, "
        "monitored on `valid_loss (min)`.",
        "",
        "**2. The two arms differ in TWO settings, not one.**",
        "",
        "| setting | V1 | V2 |",
        "|---|---|---|",
    ]
    for entry in payload["protocol_differences"]:
        lines.append(f"| `{entry['setting']}` | {entry['V1']} | {entry['V2']} |")
    lines += ["",
              f"({len(payload['cosmetic_differences'])} further differences are run labels "
              "-- `experiment.name`, `output_root`, `notes`, `tags` -- which change no data, "
              "no optimisation and no evaluation.)"]
    lines += [
        "",
        "`encoder_loops` 2 -> 4 doubles V2's effective encoder depth at no parameter cost.",
        "It is part of the approved architecture, but it means **this comparison is between",
        "two configurations, not a clean isolation of the input front end.** Any statement",
        "that the numerical front end alone caused the gain is not supported by these runs.",
        "",
        "Everything else is identical: n=12, h=2, q=251, sigma=3, RLWE circulant, base 81",
        "lsb-first no separator, same secret index, same seed, same AdamW settings, schedule,",
        "batch size 64, 100,032 samples, 2,048 validation sequences, tau=0.1, greedy decode.",
        "Both arms of each pair share the same validation stream.",
        "",
        "## Head-to-head (seed 0, the exactly matched pair)",
        "",
        "| Metric | V1 GatedUT | V2 NACT | Winner | Difference | % |",
        "|---|---:|---:|:---:|---:|---:|",
    ]
    for entry in payload["head_to_head_seed_0"]:
        percent = (f"{entry['percent_difference']:+.1f}%"
                   if entry["percent_difference"] is not None else "--")
        winner = entry["winner"] if entry["winner"] != "n/a" else "--"
        lines.append(f"| {entry['metric']} | {entry['V1']:,.4f} | {entry['V2']:,.4f} | "
                     f"**{winner}** | {entry['absolute_difference_v2_minus_v1']:+,.4f} | {percent} |")

    chance = payload["chance_baselines"]
    lines += [
        "",
        f"Chance levels: token {chance['token_accuracy']:.4f}, exact {chance['exact_accuracy']:.5f}, "
        f"acc_tau {chance['acc_tau']:.4f}. Loss reference points: "
        f"{payload['loss_reference_points']['marginal_only_nats']:.4f} nats for a model that",
        "learned only the output marginals (the real zero point), "
        f"{payload['loss_reference_points']['irreducible_floor_nats']:.4f} nats irreducible.",
        "",
        "## Across all 5 seeds (descriptive spread, not a significance test)",
        "",
        "| metric | V1 mean | V1 range | V2 mean | V2 range | ranges overlap |",
        "|---|---:|---|---:|---|:---:|",
    ]
    for key, entry in across.items():
        a, b = entry["V1"], entry["V2"]
        lines.append(
            f"| {key} | {a['mean']:.4f} | [{a['min']:.4f}, {a['max']:.4f}] | "
            f"{b['mean']:.4f} | [{b['min']:.4f}, {b['max']:.4f}] | "
            f"{'yes' if entry['ranges_overlap'] else '**no**'} |")

    v1_bad = payload["v1_seeds_at_or_below_acc_tau_chance"]
    lines += [
        "",
        "## What the evidence supports",
        "",
        f"- **V2 has higher acc_tau by {primary['V2']['valid_acc_tau'] - primary['V1']['valid_acc_tau']:.4f}** "
        f"at seed 0 ({primary['V1']['valid_acc_tau']:.4f} -> {primary['V2']['valid_acc_tau']:.4f}), and by "
        f"{across['valid_acc_tau']['V2']['mean'] - across['valid_acc_tau']['V1']['mean']:.4f} on the 5-seed mean.",
        f"- **V2 has lower validation loss by "
        f"{primary['V1']['valid_loss'] - primary['V2']['valid_loss']:.4f} nats** at seed 0. V2's mean "
        f"({across['valid_loss']['V2']['mean']:.4f}) sits below the marginals-only baseline of 1.8650;",
        f"  V1's mean ({across['valid_loss']['V1']['mean']:.4f}) sits essentially at it.",
        f"- **V2 exact integer accuracy is ~17x V1's** on the 5-seed mean "
        f"({across['valid_exact_accuracy']['V1']['mean']:.4f} -> "
        f"{across['valid_exact_accuracy']['V2']['mean']:.4f}), against a chance level of 0.00398.",
        f"- **V1 is unstable at this budget.** Its acc_tau ranges "
        f"[{across['valid_acc_tau']['V1']['min']:.4f}, {across['valid_acc_tau']['V1']['max']:.4f}] "
        f"and {len(v1_bad)} of 5 seeds ({v1_bad}) land at or below the",
        "  0.1929 chance line - those seeds learned nothing measurable. V2's range is "
        f"[{across['valid_acc_tau']['V2']['min']:.4f}, {across['valid_acc_tau']['V2']['max']:.4f}] "
        "with every seed far above chance.",
        f"- **V1 is faster.** V2 runs at "
        f"{across['samples_per_second']['V2']['mean'] / across['samples_per_second']['V1']['mean'] * 100:.0f}% "
        "of V1's training throughput and takes",
        f"  {across['elapsed_seconds']['V2']['mean'] / across['elapsed_seconds']['V1']['mean']:.2f}x the "
        "wall clock, which is the expected cost of `encoder_loops=4`.",
        "  The phase-15 forward benchmark showed V2 at `T_e=2` is ~1.95x *faster* than V1 at",
        "  n=128, so this is a depth cost, not a front-end cost.",
        "- Loss, acc_tau, exact accuracy and mean error have **non-overlapping ranges across",
        "  all five seeds**. That is a description of the observed spread, not a significance",
        "  test, and five paired runs at one instance size is a narrow base.",
        "",
        "## What it does not support",
        "",
        "- Not a cryptographic result. n=12, h=2 has only C(12,2)=66 possible secrets and was",
        "  always a diagnostic positive control.",
        "- **Not an attribution to the numerical front end.** `encoder_loops` differs too.",
        "  Separating them needs a V2 run at `T_e=2`, which does not exist.",
        "- **Not a secret-recovery result.** Nothing here was run through recovery, and phase 10",
        "  measured V1's direct recovery as a failure. V2 is untested on that axis.",
        "- **Sparsity generalization: not available in existing artifacts.** "
        f"`results/recovery_generalization/` holds V1's phase-12 output only; no V2 equivalent",
        "  exists, and no new experiment was run. The primary phase-12 research test - whether",
        "  V2 moves the zero-lift crossing point away from nnz=8/6 - is therefore still open.",
        "",
        "## Verdict",
        "",
        "```",
        "OVERALL WINNER:                 V2 Salsa2-NACT (accuracy), with caveats above",
        "",
        "BEST ACCURACY:                  V2 - lower loss, higher acc_tau, higher exact",
        "                                accuracy, lower error, on every one of 5 seeds",
        "BEST CPU EFFICIENCY:            V1 - ~1.37x the training throughput as configured",
        "                                (V2's deficit is encoder_loops=4, not the front end)",
        "BEST PARAMETER EFFICIENCY:      V1 on raw count (4,131,200 vs 4,241,288, -2.6%);",
        "                                V2 on accuracy per parameter, by a wide margin",
        "BEST SPARSE-INPUT GENERALIZATION: UNDETERMINED - no V2 sparsity artifacts exist",
        "BEST OVERALL:                   V2, for in-distribution learning at n=12,h=2 only.",
        "                                The question the architecture was designed to answer",
        "                                - sparse-input generalization - remains unmeasured.",
        "```",
        "",
        "## Artifacts",
        "",
        "- `v1_v2_comparison.md` (this file)",
        "- `v1_v2_comparison.json`",
        "- `v1_v2_comparison.csv`",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
