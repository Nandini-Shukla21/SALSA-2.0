"""Train a SALSA 2.0 model on the ``a -> b`` task, CPU-first.

Examples::

    python scripts/train.py configs/experiment_4_13m_n30_r.yaml
    python scripts/train.py configs/smoke_train.yaml --run-id smoke
    python scripts/train.py configs/experiment_4_13m_n30_r.yaml --set lwe.num_train_samples=50000

The run writes everything into ``<output_root>/<experiment>/<run-id>/``:
config, metadata, log, ``metrics.jsonl`` / ``metrics.csv``, checkpoints and a
final ``summary.json``.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from salsa.training import Trainer  # noqa: E402
from salsa.utils import ConfigError, load_config  # noqa: E402
from salsa.utils.logging import RunContext  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser."""
    parser = argparse.ArgumentParser(description="Train a SALSA 2.0 model on CPU.")
    parser.add_argument(
        "config",
        type=Path,
        nargs="?",
        help="Path to a YAML configuration file (positional form).",
    )
    parser.add_argument(
        "--config",
        dest="config_flag",
        type=Path,
        default=None,
        help="Path to a YAML configuration file (flag form; equivalent).",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a setting, e.g. --set training.batch_size=32 (repeatable).",
    )
    parser.add_argument("--run-id", type=str, default=None, help="Explicit run id.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse the run directory and continue from its last checkpoint.",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Do not mirror the log to stdout."
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Run one training experiment.

    Args:
        argv: Optional argument list.

    Returns:
        Process exit code: 0 on success, 2 on a configuration error.
    """
    args = build_parser().parse_args(argv)
    config_path = args.config_flag or args.config
    if config_path is None:
        print("ERROR: provide a configuration path.", file=sys.stderr)
        return 2

    try:
        config = load_config(config_path, overrides=args.overrides)
        warnings = config.validate()
    except ConfigError as exc:
        print(f"CONFIG ERROR: {exc}", file=sys.stderr)
        return 2

    context = RunContext.create(
        config, run_id=args.run_id, console=not args.quiet, exist_ok=args.resume
    )
    for warning in warnings:
        context.logger.warning(warning)

    trainer = Trainer(config, context=context)
    if args.resume and trainer.maybe_resume():
        context.logger.info("Resumed an existing run.")

    try:
        state = trainer.train()
    finally:
        context.close()

    print(f"\nRun directory: {trainer.context.run_dir}")
    print(f"Stop reason  : {state.stop_reason}")
    print(f"Samples seen : {state.samples_seen:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
