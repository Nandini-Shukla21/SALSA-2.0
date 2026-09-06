"""Run the Salsa 2.0 end-to-end pipeline from a YAML configuration.

    python scripts/run_pipeline.py configs/pipeline_n12.yaml
    python scripts/run_pipeline.py configs/pipeline_n20.yaml --no-evaluate

No training is performed.  The pipeline loads a completed checkpoint, verifies
its identity, and runs inference, recovery and verification.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from salsa.pipeline import run_salsa2_pipeline  # noqa: E402


def main(argv: Optional[List[str]] = None) -> int:
    """Run one pipeline configuration and print a summary."""
    parser = argparse.ArgumentParser(description="Salsa 2.0 end-to-end pipeline.")
    parser.add_argument("config", type=Path, help="pipeline YAML configuration")
    parser.add_argument("--out", type=Path, default=None,
                        help="override pipeline.output_dir")
    parser.add_argument("--no-evaluate", action="store_true",
                        help="skip the ground-truth comparison (attack setting)")
    parser.add_argument("--no-verify", action="store_true",
                        help="skip residual verification")
    args = parser.parse_args(argv)

    result = run_salsa2_pipeline(
        args.config,
        evaluate=not args.no_evaluate,
        verify=False if args.no_verify else None,
        output_dir=args.out,
    )

    print("=" * 88)
    print(f"SALSA 2.0 PIPELINE - {result.label}")
    print("=" * 88)
    print(f"  checkpoint   {result.checkpoint}")
    print(f"  model        {result.model_name} ({result.architecture}), "
          f"{result.parameter_count:,} parameters")
    print(f"  instance     n={result.instance['n']} h={result.instance['h']} "
          f"q={result.instance['q']} sigma={result.instance['sigma']}  "
          f"search space {result.instance['search_space']}")
    print()
    print("  STAGE 1 - RECOVERY (no secret in scope)")
    print(f"    rule            {result.selection_rule} "
          f"(min_separation {result.min_separation})")
    print(f"    K sweep         {result.k_values}")
    print(f"    separations     {result.k_separations}")
    print(f"    excluded        {result.excluded_k}  (degenerate: separation below guard)")
    print(f"    diagnostic K    {result.selected_k}")
    print(f"    candidate       {result.recovered_candidate}")
    print(f"    probe validity  {result.probe_decode_validity:.4f}")
    print()
    if result.verification_ran:
        stats = result.residual_statistics
        print("  STAGE 2 - VERIFICATION (only A, b, candidate, q, sigma)")
        print(f"    split/seed      {result.verification_split} / "
              f"{result.verification_data_seed}, {result.verification_samples} samples")
        print(f"    residual std    {stats['std']:.3f}   mean|r| {stats['mean_abs']:.3f}"
              f"   <=3sigma {stats['fraction_within_3_sigma']:.4f}")
        for name, entry in result.residual_baselines.items():
            print(f"    baseline {name:<12}std {entry['std']:.3f}")
        print(f"    VERIFICATION    {'PASS' if result.verification_passed else 'FAIL'}")
        print()
    if result.true_secret:
        print("  STAGE 3 - EVALUATION (the secret enters only here)")
        print(f"    true secret     {result.true_secret}")
        print(f"    coordinate acc  {result.coordinate_accuracy:.4f}  "
              f"(all-zeros gets {result.baselines['all_zeros_coordinate_accuracy']:.4f} free)")
        print(f"    hamming         {result.hamming_distance}")
        print(f"    successful K    {result.successful_k}")
        print(f"    EXACT RECOVERY  {'YES' if result.exact_recovery else 'NO'}")
    else:
        print("  STAGE 3 - EVALUATION skipped (no ground truth consulted)")
    print()
    print(f"  runtime {result.runtime_seconds:.2f}s  {result.stage_seconds}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
