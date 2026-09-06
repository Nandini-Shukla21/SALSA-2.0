"""Phase 27 report: end-to-end pipeline integration and replay.

Read-only apart from running the integrated pipeline itself, which trains
nothing.  Both established results are replayed and compared against the
numbers the earlier phases recorded.
"""

import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from salsa.pipeline import run_salsa2_pipeline  # noqa: E402

OUTPUT_DIR = REPO_ROOT / "results" / "final_pipeline"

#: What the earlier phases established, to check the replay against.
EXPECTED = {
    "nact_f_n12": {
        "source": "phase 24 (results/nact_ablation_F_recovery)",
        "exact_recovery": True, "coordinate_accuracy": 1.0,
        "hamming_distance": 0, "successful_k": 8, "verification_passed": True,
        "parameter_count": 4_238_208,
    },
    "nact_f_n20": {
        "source": "phase 26 (results/n20_nact_f_recovery)",
        "exact_recovery": True, "coordinate_accuracy": 1.0,
        "hamming_distance": 0, "successful_k": 8, "verification_passed": True,
        "parameter_count": 4_238_208,
    },
}

CONFIGS = {"nact_f_n12": "configs/pipeline_n12.yaml",
           "nact_f_n20": "configs/pipeline_n20.yaml"}


def compare(result, expected: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Check a replay against the previously established numbers."""
    actual = {
        "exact_recovery": result.exact_recovery,
        "coordinate_accuracy": result.coordinate_accuracy,
        "hamming_distance": result.hamming_distance,
        "successful_k": len(result.successful_k),
        "verification_passed": result.verification_passed,
        "parameter_count": result.parameter_count,
    }
    return [
        {"field": key, "expected": expected[key], "replayed": actual[key],
         "matches": expected[key] == actual[key]}
        for key in actual
    ]


def main() -> int:
    """Replay both results through the integrated pipeline and report."""
    print("=" * 92)
    print("PHASE 27 - END-TO-END PIPELINE INTEGRATION AND REPLAY  (no training)")
    print("=" * 92)

    results, checks, discrepancies = {}, {}, []
    for label, config in CONFIGS.items():
        result = run_salsa2_pipeline(config, output_dir=OUTPUT_DIR)
        results[label] = result
        checks[label] = compare(result, EXPECTED[label])
        failed = [c for c in checks[label] if not c["matches"]]
        discrepancies.extend({"label": label, **c} for c in failed)
        print(f"\n  {result.summary_line()}")
        for check in checks[label]:
            print(f"    [{'OK ' if check['matches'] else 'DIFFERS'}] "
                  f"{check['field']:<22}expected {str(check['expected']):>10}  "
                  f"replayed {check['replayed']}")

    if discrepancies:
        print("\n  DISCREPANCIES FOUND - stopping before drawing conclusions:")
        for entry in discrepancies:
            print(f"    {entry}")
    else:
        print("\n  Both replays reproduce the established results exactly.")

    payload = {
        "phase": "27 - end-to-end pipeline integration",
        "status": "INTEGRATION + READ-ONLY REPLAY. No training, no new experiment, "
                  "no checkpoint created or modified.",
        "entry_point": "salsa.pipeline.run_salsa2_pipeline",
        "cli": "scripts/run_pipeline.py <config.yaml>",
        "workflow": ["public LWE/RLWE data", "encoding", "NACT-F model",
                     "predict b", "direct secret recovery", "candidate secret",
                     "independent residual verification", "structured result"],
        "stage_separation": {
            "recovery": "public probes and model predictions only",
            "verification": "(A, b, candidate, q, sigma) only",
            "evaluation": "the only stage that loads the true secret, and it runs last",
        },
        "recovery_rule": {
            "default": "aggregate (separation-weighted vote)",
            "diagnostic": "best_margin (single K, guarded by min_separation)",
            "guard": "min_separation defaults to floor(sigma) + 1 = 4 at sigma=3, "
                     "which excludes K=1 and K=250 (separation 1) from selection",
            "phase26_defect": "the unguarded best_margin rule selected K=1 at n=20 "
                              "because a constant answer at separation 1 yields the "
                              "maximum possible margin",
            "still_secret_free": True,
        },
        "expected": EXPECTED,
        "replay_checks": checks,
        "discrepancies": discrepancies,
        "results": {label: result.to_dict() for label, result in results.items()},
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "pipeline_report.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8")

    columns = ["section", "label", "field", "expected", "replayed", "matches",
               "metric", "value"]
    with (OUTPUT_DIR / "pipeline_report.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for label, entries in checks.items():
            for entry in entries:
                writer.writerow({"section": "replay_check", "label": label, **entry})
        for label, result in results.items():
            for metric in ("parameter_count", "coordinate_accuracy", "hamming_distance",
                           "exact_recovery", "verification_passed",
                           "probe_decode_validity", "runtime_seconds"):
                writer.writerow({"section": "result", "label": label,
                                 "metric": metric, "value": getattr(result, metric)})

    write_markdown(payload, results, OUTPUT_DIR / "pipeline_report.md")
    print()
    for name in ("pipeline_report.md", "pipeline_report.json", "pipeline_report.csv"):
        print(f"  wrote {OUTPUT_DIR / name}")
    return 1 if discrepancies else 0


def write_markdown(payload: Dict[str, Any], results, path: Path) -> None:
    """Render the integration report."""
    n12, n20 = results["nact_f_n12"], results["nact_f_n20"]
    rule = payload["recovery_rule"]

    lines = [
        "# Phase 27 — Salsa 2.0 end-to-end pipeline",
        "",
        "**Integration and read-only replay.** No training, no new experiment, no",
        "checkpoint created or modified.",
        "",
        "## The workflow, in one call",
        "",
        "```",
        "  public LWE/RLWE data",
        "          |",
        "      encoding  (LatticeCodec, base 81, lsb-first, no separator)",
        "          |",
        "      NACT-F    (one coordinate token per a_i, encoder sequence n+2)",
        "          |",
        "      predict b (greedy decode)",
        "          |",
        "  direct secret recovery   <- STAGE 1: no secret in scope",
        "          |",
        "     candidate secret",
        "          |",
        "  residual verification    <- STAGE 2: only (A, b, candidate, q, sigma)",
        "          |",
        "    structured result      <- STAGE 3: the secret enters here, last",
        "```",
        "",
        "```python",
        "from salsa import run_salsa2_pipeline",
        "",
        "result = run_salsa2_pipeline('configs/pipeline_n12.yaml')",
        "print(result.exact_recovery, result.verification_passed)",
        "```",
        "",
        "or `python scripts/run_pipeline.py configs/pipeline_n20.yaml`.",
        "",
        "## Why the three stages are separate fields",
        "",
        "They have different epistemic status, and merging them is how a recovery",
        "experiment fools itself:",
        "",
        "| stage | what it may see |",
        "|---|---|",
    ]
    for stage, sees in payload["stage_separation"].items():
        lines.append(f"| **{stage}** | {sees} |")
    lines += [
        "",
        "The ordering is enforced by construction — the secret is not read until a",
        "candidate *and* a verification verdict already exist — and `evaluate=False`",
        "runs the whole thing with no ground truth at all, which is the real attack",
        "setting.",
        "",
        "## Recovery rule hardening",
        "",
        f"**The defect.** {rule['phase26_defect']}.",
        "",
        "**What changed.** Two rules are now named and documented:",
        "",
        f"- **`{rule['default']}`** — the default.",
        f"- **`{rule['diagnostic']}`** — retained as a diagnostic.",
        "",
        f"**The guard.** {rule['guard']}. It is derived from the problem rather than",
        "picked: a probe can only discriminate if its two hypotheses are further apart",
        "than the error scale. Excluded K are still measured and still vote, with the",
        "negligible weight their separation earns them — they are excluded from",
        "*selection*, not from the evidence.",
        "",
        f"**Still secret-free: {rule['still_secret_free']}.** Separation is a function of",
        "K and q only. Nothing in the rule consults the secret.",
        "",
        "Measured effect: at n=20 the unguarded rule selected K=1; the guarded rule",
        f"selects K={n20.selected_k}.",
        "",
        "## Replay of the established results",
        "",
        "| check | n=12 expected | n=12 replayed | n=20 expected | n=20 replayed |",
        "|---|---:|---:|---:|---:|",
    ]
    for a, b in zip(payload["replay_checks"]["nact_f_n12"],
                    payload["replay_checks"]["nact_f_n20"]):
        mark_a = "" if a["matches"] else " **DIFFERS**"
        mark_b = "" if b["matches"] else " **DIFFERS**"
        lines.append(f"| {a['field']} | {a['expected']} | {a['replayed']}{mark_a} | "
                     f"{b['expected']} | {b['replayed']}{mark_b} |")
    lines += [
        "",
        f"n=12 source: {payload['expected']['nact_f_n12']['source']}. "
        f"n=20 source: {payload['expected']['nact_f_n20']['source']}.",
        "",
        ("**Both replays reproduce the established results exactly.**"
         if not payload["discrepancies"] else
         f"**DISCREPANCIES: {payload['discrepancies']}**"),
        "",
        "Verification residuals on fresh samples:",
        "",
        "| | n=12 | n=20 |",
        "|---|---:|---:|",
        f"| verification split | {n12.verification_split} | {n20.verification_split} |",
        f"| data seed | {n12.verification_data_seed} | {n20.verification_data_seed} |",
        f"| residual std | {n12.residual_statistics['std']:.3f} | "
        f"{n20.residual_statistics['std']:.3f} |",
        f"| all-zeros baseline std | "
        f"{n12.residual_baselines['all_zeros']['std']:.3f} | "
        f"{n20.residual_baselines['all_zeros']['std']:.3f} |",
        f"| runtime (s) | {n12.runtime_seconds:.2f} | {n20.runtime_seconds:.2f} |",
        "",
        "The whole pipeline runs in about a second per configuration, because nothing",
        "in it trains.",
        "",
        "## Final scientific summary",
        "",
        "| Stage | n=12, h=2 | n=20, h=2 |",
        "|---|---|---|",
        "| learning | measured | measured |",
        "| direct recovery | **YES** | **YES** |",
        "| coordinate accuracy | **100%** | **100%** |",
        "| verification | **PASS** | **PASS** |",
        f"| search space C(n,2) | {n12.instance['search_space']} | "
        f"{n20.instance['search_space']} |",
        f"| successful K | {len(n12.successful_k)}/10 | {len(n20.successful_k)}/10 |",
        "",
        "### How the project got here",
        "",
        "- **V1 Salsa2-GatedUT, 4,131,200 parameters.** Learned in-distribution at",
        "  n=12 but direct recovery failed completely: 58% of probes were undecodable",
        "  and the model answered a single constant on 8 of 10 K values.",
        "- **The sparse-input problem.** Phase 12 measured V1's lift over the best",
        "  input-blind constant falling monotonically with sparsity and crossing zero",
        "  between nnz=8 and nnz=6, reaching −0.44 on the `K·e_i` probes. Phase 11 had",
        "  measured why: a probe is roughly a 1-in-10²⁵ input under the training",
        "  distribution, and detecting a zero coordinate is a conjunction across two",
        "  token positions.",
        "- **NACT was introduced** to attack that: one token per coordinate, absolute",
        "  coordinate identity, and numerical features chosen for the modular structure.",
        "- **NACT-F, 4,238,208 parameters,** is the ablation that removed every extra",
        "  numerical, zero-aware and sparse-bias component and kept only the one-token",
        "  representation. It matched full NACT within one-seed noise, which is why the",
        "  **one-token representation is the important simplification** — the features",
        "  were not doing the heavy lifting.",
        "- **Recovery improvement.** V1: no exact recovery, probe decode validity 0.417.",
        f"  NACT-F: exact recovery at both dimensions, probe decode validity "
        f"{n12.probe_decode_validity:.3f} and {n20.probe_decode_validity:.3f}.",
        "- **Verification.** Residual std ~3.0 against a configured sigma of 3.0, while",
        "  every incorrect candidate sits near 72.5 — a separation of roughly 24x.",
        "",
        "### Currently demonstrated",
        "",
        "**n=12, h=2 and n=20, h=2** — learning, exact direct secret recovery, and",
        "independent residual verification, end to end.",
        "",
        "### Not demonstrated",
        "",
        "**n=30. n=50. n=128. A practical cryptanalytic attack. A general security",
        "break.** These are diagnostic instances: C(12,2) = 66 and C(20,2) = 190",
        "possible secrets. One secret, one checkpoint and one seed per dimension. Two",
        "dimensions is not a scaling law.",
        "",
        "## Artifacts",
        "",
        "- `pipeline_report.md` (this file)",
        "- `pipeline_report.json`",
        "- `pipeline_report.csv`",
        "- `nact_f_n12_result.json`, `nact_f_n20_result.json` — full result objects",
        "",
        "**PHASE 27 COMPLETE — END-TO-END PIPELINE INTEGRATION ONLY. "
        "NO NEW TRAINING PERFORMED.**",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
