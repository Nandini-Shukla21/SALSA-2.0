"""Run direct secret recovery against a trained checkpoint, then score it.

The two phases are kept strictly apart:

1. **Recovery** -- :class:`~salsa.recovery.DirectRecovery` sees the model, the
   codec and public parameters.  It produces candidate secrets.
2. **Evaluation** -- only after recovery has returned does this script load the
   true secret and compare.  Nothing measured in phase 1 depends on it.

The checkpoint is located from the run's own metadata rather than an assumed
filename, and every architectural and cryptographic value in it is verified
against the configuration before the model is built.

No training is performed.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from salsa.data import LatticeCodec  # noqa: E402
from salsa.data.secrets import secret_from_config  # noqa: E402
from salsa.models import build_model, count_trainable_parameters  # noqa: E402
from salsa.recovery import DirectRecovery, probe_separation  # noqa: E402
from salsa.utils import load_config  # noqa: E402


def locate_checkpoint(run_dir: Path) -> Path:
    """Find the best checkpoint using the run's metadata.

    Args:
        run_dir: A completed run directory.

    Returns:
        Path to the checkpoint selected by the run's monitor metric.

    Raises:
        FileNotFoundError: If no usable checkpoint is present.
    """
    summary_path = run_dir / "artifacts" / "summary.json"
    checkpoints = run_dir / "checkpoints"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text("utf-8"))
        best_epoch = summary["state"].get("best_epoch")
        if best_epoch is not None and (checkpoints / "best.pt").is_file():
            return checkpoints / "best.pt"
    for name in ("best.pt", "last.pt"):
        if (checkpoints / name).is_file():
            return checkpoints / name
    raise FileNotFoundError(f"no checkpoint under {checkpoints}")


def verify(checkpoint: Dict[str, Any], config, codec: LatticeCodec) -> List[str]:
    """Check the checkpoint against the configuration.

    Returns:
        A list of mismatch descriptions; empty means everything agrees.
    """
    saved, spec = checkpoint["config"], checkpoint["spec"]
    expectations = [
        ("parameter count", 4_131_200, checkpoint["parameter_count"]),
        ("architecture", "gated_universal_transformer", spec["arch"]),
        ("n", config.lwe.n, saved["lwe"]["n"]),
        ("h", config.lwe.resolved_hamming_weight, saved["lwe"]["hamming_weight"]),
        ("q", config.lwe.q, saved["lwe"]["q"]),
        ("sigma", config.lwe.sigma, saved["lwe"]["sigma"]),
        ("base", config.encoding.base, saved["encoding"]["base"]),
        ("digit_order", "lsb_first", saved["encoding"]["digit_order"]),
        ("separator", False, saved["encoding"]["separator"]),
        ("fixed_width", True, saved["encoding"]["fixed_width"]),
        ("vocabulary", codec.vocabulary.size, spec["vocab_size"]),
        ("seed", config.experiment.seed, saved["experiment"]["seed"]),
    ]
    return [
        f"{name}: config/expected {expected!r} vs checkpoint {actual!r}"
        for name, expected, actual in expectations
        if expected != actual
    ]


def main(argv: Optional[List[str]] = None) -> int:
    """Run recovery and report."""
    parser = argparse.ArgumentParser(description="Direct secret recovery.")
    parser.add_argument("--config", type=Path,
                        default=REPO_ROOT / "configs" / "recovery_n12_h2.yaml")
    parser.add_argument("--run-dir", type=Path,
                        default=REPO_ROOT / "results" / "control_a_n12_h2" / "control")
    parser.add_argument("--method", type=str, default="anchor")
    parser.add_argument("--out", type=Path,
                        default=REPO_ROOT / "results" / "direct_recovery")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    config.validate()
    codec = LatticeCodec.from_config(config)

    checkpoint_path = locate_checkpoint(args.run_dir)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    print("=" * 78)
    print("CHECKPOINT VERIFICATION")
    print("=" * 78)
    print(f"  run dir    : {args.run_dir}")
    print(f"  checkpoint : {checkpoint_path.name}  "
          f"(epoch {checkpoint['state']['epoch']}, "
          f"{checkpoint['state']['samples_seen']:,} samples)")
    mismatches = verify(checkpoint, config, codec)
    if mismatches:
        print("  MISMATCHES FOUND - STOPPING:")
        for line in mismatches:
            print(f"    ! {line}")
        return 2
    print(f"  all values verified; fingerprint {checkpoint['config_fingerprint']}")

    model = build_model(config)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    print(f"  model loaded: {model.name}, {count_trainable_parameters(model):,} parameters")
    print(f"  codec: n={codec.n} q={codec.q} L_in={codec.input_length} "
          f"vocab={codec.vocabulary.size}")

    # ---------------- PHASE 1: recovery (no ground truth anywhere) --------- #
    k_values = list(config.recovery.direct_k_values)
    print()
    print("=" * 78)
    print(f"PHASE 1 - DIRECT RECOVERY  (method={args.method}, {len(k_values)} K values)")
    print("=" * 78)
    recovery = DirectRecovery(model, codec, method=args.method)
    report = recovery.recover(k_values)
    print(f"  K sweep      : {k_values}")
    print(f"  separations  : {[probe_separation(k, codec.q) for k in k_values]}")
    print(f"  selected K   : {report.selected_K}  (highest mean margin; no secret used)")
    print(f"  aggregate    : {report.aggregate_candidate.tolist()}")

    # ---------------- PHASE 2: evaluation (truth enters only here) --------- #
    true_secret = secret_from_config(config)
    print()
    print("=" * 78)
    print("PHASE 2 - EVALUATION  (the true secret is consulted only from here on)")
    print("=" * 78)
    print(f"  true secret  : {true_secret.tolist()}  (weight {int(true_secret.sum())})")
    print()

    def score(candidate: np.ndarray) -> Dict[str, Any]:
        matches = int((candidate == true_secret).sum())
        return {
            "coordinate_accuracy": matches / len(true_secret),
            "matches": matches,
            "hamming_distance": int((candidate != true_secret).sum()),
            "exact": bool(np.array_equal(candidate, true_secret)),
        }

    # A weight-h secret over n coordinates is mostly zeros, so an all-zeros
    # guess already scores (n-h)/n. Any candidate must beat that to mean anything.
    zeros_score = score(np.zeros_like(true_secret))
    print(f"  TRIVIAL BASELINE - all-zeros candidate: accuracy "
          f"{zeros_score['coordinate_accuracy']:.3f}, hamming "
          f"{zeros_score['hamming_distance']}. Nothing below this is evidence.")
    print()
    print(f"  {'K':>6} {'sep':>4} {'margin':>7} {'fail':>5} {'uniq':>5} "
          f"{'candidate':>28} {'acc':>6} {'ham':>4} {'exact':>6}")
    print("  " + "-" * 82)
    rows = []
    for result in report.per_k:
        s = score(result.candidate)
        # How many DISTINCT predictions the model gave across the n probes.
        # One distinct value means the model did not discriminate coordinates
        # at all -- the probe carried no signal through it.
        distinct = len({o.decoded_b for o in result.outcomes})
        s["distinct_predictions"] = distinct
        s["beats_all_zeros"] = bool(
            s["coordinate_accuracy"] > zeros_score["coordinate_accuracy"]
        )
        rows.append({**result.to_dict(), "evaluation": s})
        print(f"  {result.K:>6} {result.separation:>4} {result.mean_margin:>7.3f} "
              f"{result.decode_failure_rate:>5.2f} {distinct:>5} "
              f"{str(result.candidate.tolist()):>28} "
              f"{s['coordinate_accuracy']:>6.3f} {s['hamming_distance']:>4} "
              f"{'YES' if s['exact'] else 'no':>6}")

    aggregate_score = score(report.aggregate_candidate)
    print("  " + "-" * 76)
    print(f"  {'AGG':>6} {'':>4} {'':>7} {'':>5} "
          f"{str(report.aggregate_candidate.tolist()):>28} "
          f"{aggregate_score['coordinate_accuracy']:>6.3f} "
          f"{aggregate_score['hamming_distance']:>4} "
          f"{'YES' if aggregate_score['exact'] else 'no':>6}")

    selected = next(r for r in report.per_k if r.K == report.selected_K)
    selected_score = score(selected.candidate)

    print()
    print("  Per-coordinate detail for the selected K "
          f"(K={selected.K}, separation {selected.separation}):")
    print(f"    {'i':>3} {'true':>5} {'rec':>4} {'decoded b':>10} {'d(b,0)':>7} "
          f"{'d(b,K)':>7} {'score':>7} {'ok':>4}")
    for outcome in selected.outcomes:
        truth = int(true_secret[outcome.coordinate])
        flag = "OK" if truth == outcome.decision else "WRONG"
        print(f"    {outcome.coordinate:>3} {truth:>5} {outcome.decision:>4} "
              f"{outcome.decoded_b:>10} {outcome.distance_to_zero:>7} "
              f"{outcome.distance_to_K:>7} {outcome.score:>7.3f} {flag:>5}")

    best = max(rows, key=lambda r: r["evaluation"]["coordinate_accuracy"])
    print()
    print("=" * 78)
    print("RESULT")
    print("=" * 78)
    print(f"  selected K (secret-free rule) : K={selected.K}  "
          f"accuracy {selected_score['coordinate_accuracy']:.3f}  "
          f"hamming {selected_score['hamming_distance']}  "
          f"exact {'YES' if selected_score['exact'] else 'NO'}")
    print(f"  aggregate vote                : accuracy "
          f"{aggregate_score['coordinate_accuracy']:.3f}  "
          f"hamming {aggregate_score['hamming_distance']}  "
          f"exact {'YES' if aggregate_score['exact'] else 'NO'}")
    print(f"  best K in hindsight (eval only): K={best['K']}  "
          f"accuracy {best['evaluation']['coordinate_accuracy']:.3f}  "
          f"exact {'YES' if best['evaluation']['exact'] else 'NO'}")
    any_exact = any(r["evaluation"]["exact"] for r in rows) or aggregate_score["exact"]
    beats = [r["K"] for r in rows if r["evaluation"]["beats_all_zeros"]]
    collapsed = [r["K"] for r in rows if r["evaluation"]["distinct_predictions"] <= 1]
    print(f"  all-zeros baseline            : accuracy "
          f"{zeros_score['coordinate_accuracy']:.3f}  "
          f"hamming {zeros_score['hamming_distance']}")
    print(f"  K values beating that baseline: {beats if beats else 'NONE'}")
    print(f"  K values where the model gave a single constant answer to every "
          f"probe: {collapsed}")
    print()
    print(f"  EXACT SECRET RECOVERY: {'YES' if any_exact else 'NO'}")

    args.out.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "DIRECT RECOVERY AGAINST A TRAINED CHECKPOINT",
        "checkpoint": str(checkpoint_path),
        "config": str(args.config),
        "config_fingerprint": checkpoint["config_fingerprint"],
        "parameter_count": checkpoint["parameter_count"],
        "instance": {"n": codec.n, "q": codec.q,
                     "h": config.lwe.resolved_hamming_weight,
                     "sigma": config.lwe.sigma,
                     "representation": "R", "vocab": codec.vocabulary.size},
        "recovery": report.to_dict(),
        "evaluation": {
            "note": "computed AFTER recovery; the attack never saw these values",
            "trivial_all_zeros_baseline": zeros_score,
            "baseline_warning": (
                "A weight-h secret over n coordinates is mostly zeros, so an "
                "all-zeros guess already reaches (n-h)/n coordinate accuracy. "
                "Coordinate accuracy at or below that number is not evidence of "
                "recovery. Undecodable probes default to bit 0, which pushes "
                "candidates toward the all-zeros answer."
            ),
            "true_secret": true_secret.tolist(),
            "true_hamming_weight": int(true_secret.sum()),
            "per_k": [{"K": r["K"], **r["evaluation"]} for r in rows],
            "selected_K": {"K": selected.K, **selected_score},
            "aggregate": aggregate_score,
            "exact_recovery": any_exact,
        },
    }
    (args.out / "direct_recovery_n12_h2.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n  wrote {args.out / 'direct_recovery_n12_h2.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
