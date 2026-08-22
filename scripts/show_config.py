"""Inspect, validate and fingerprint a Lightweight SALSA configuration.

This is the phase-1 entry point.  It resolves a config file (including its
``extends`` chain and any CLI overrides), validates it, prints the resolved
research assumptions, and reports the machine this experiment would run on.

Examples::

    python scripts/show_config.py configs/target_4_5m.yaml
    python scripts/show_config.py configs/target_4_5m.yaml --set lwe.n=50
    python scripts/show_config.py configs/baseline_cpu.yaml --dump-yaml
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

# Allow running the script directly from a source checkout without installing.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from salsa.utils import (  # noqa: E402
    ConfigError,
    device_report,
    load_config,
    resolve_device,
    save_config,
)


def build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser for this script."""
    parser = argparse.ArgumentParser(
        description="Resolve, validate and fingerprint a Lightweight SALSA config."
    )
    parser.add_argument("config", type=Path, help="Path to a YAML configuration file.")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a setting, e.g. --set training.batch_size=64 (repeatable).",
    )
    parser.add_argument(
        "--dump-yaml",
        type=Path,
        default=None,
        nargs="?",
        const=Path("-"),
        help="Write the fully resolved config to a file ('-' or no value = stdout).",
    )
    parser.add_argument(
        "--no-device-report",
        action="store_true",
        help="Skip the machine/environment report.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Run the config inspector.

    Args:
        argv: Optional argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code: 0 on success, 2 on a configuration error.
    """
    args = build_parser().parse_args(argv)

    try:
        config = load_config(args.config, overrides=args.overrides)
        warnings = config.validate()
    except ConfigError as exc:
        print(f"CONFIG ERROR: {exc}", file=sys.stderr)
        return 2

    lwe, enc, mdl = config.lwe, config.encoding, config.model
    dev = resolve_device(
        config.device.prefer,
        allow_gpu_fallback=config.device.allow_gpu_fallback,
        dtype=config.device.dtype,
    )

    print("=" * 72)
    print(f"Lightweight SALSA  |  config: {args.config}")
    print("=" * 72)
    print(f"experiment          : {config.experiment.name}")
    print(f"seed                : {config.experiment.seed} "
          f"(deterministic={config.experiment.deterministic})")
    print(f"fingerprint         : {config.fingerprint()}")
    print(f"output root         : {config.experiment.output_root}")
    print()
    print("-- LWE instance (the scientific control) " + "-" * 31)
    print(f"structure           : {lwe.structure}")
    print(f"n / q / sigma       : {lwe.n} / {lwe.q} / {lwe.sigma}")
    print(f"secret              : {lwe.secret_distribution}, "
          f"h={lwe.resolved_hamming_weight}, d={lwe.resolved_density:.4f}")
    print(f"max a value         : {lwe.max_a_fraction:.2f} * q")
    print(f"samples (tr/va/te)  : {lwe.num_train_samples} / "
          f"{lwe.num_valid_samples} / {lwe.num_test_samples} "
          f"(reuse x{lwe.sample_reuse})")
    print(f"encoding            : base {enc.resolved_input_base} in / "
          f"{enc.resolved_output_base} out, separator={enc.separator}")
    print()
    print("-- Model capacity (counts measured in phase 5) " + "-" * 25)
    print(f"arch                : {mdl.arch}")
    print(f"encoder             : dim {mdl.encoder_dim}, "
          f"{mdl.encoder_layers} layers, {mdl.encoder_heads} heads")
    print(f"decoder             : dim {mdl.decoder_dim}, "
          f"{mdl.decoder_layers} layers, {mdl.decoder_heads} heads")
    print(f"parameter budget    : max {mdl.max_parameters:,}"
          + (f", nominal target {mdl.target_parameters:,}"
             if mdl.target_parameters else "")
          + (f", min {mdl.min_parameters:,}" if mdl.min_parameters else ""))
    print("                      (NOMINAL ONLY - no model has been built or "
          "measured yet)")
    print()
    print("-- Compute " + "-" * 61)
    print(f"requested / resolved: {dev.requested} -> {dev.name} ({dev.dtype})")
    for note in dev.warnings:
        print(f"  ! {note}")
    if not args.no_device_report:
        report = device_report()
        print(f"platform            : {report['platform']}")
        print(f"logical CPUs        : {report['cpu_count_logical']}")
        print(f"total RAM (GB)      : {report['total_ram_gb']}")
        print(f"torch               : {report['torch_version'] or 'NOT INSTALLED'}")
        print(f"cuda available      : {report['cuda_available']} | "
              f"mps available: {report['mps_available']}")
    print()

    if warnings:
        print("-- Validation warnings " + "-" * 49)
        for warning in warnings:
            print(f"  ! {warning}")
        print()
    else:
        print("Validation: OK (no warnings).\n")

    if args.dump_yaml is not None:
        if str(args.dump_yaml) == "-":
            import yaml

            print(yaml.safe_dump(config.to_dict(), sort_keys=True))
        else:
            path = save_config(config, args.dump_yaml)
            print(f"Resolved config written to {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
