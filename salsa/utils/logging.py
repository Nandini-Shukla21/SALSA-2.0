"""Run directories, file logging and structured metric streams.

Every Lightweight SALSA run writes a self-contained directory::

    <output_root>/<experiment_name>/<run_id>/
        config.yaml        exact resolved configuration (incl. defaults)
        metadata.json      environment, seed, fingerprint, package versions
        run.log            human-readable log
        metrics.jsonl      one JSON object per logged evaluation (append-only)
        metrics.csv        the same rows as a spreadsheet-friendly table
        checkpoints/       model checkpoints
        artifacts/         plots, recovered secrets, benchmark dumps

The JSONL stream is the source of truth; the CSV is rewritten whenever a new
metric key appears, so adding a metric mid-run never corrupts the table.
"""

import csv
import json
import logging
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

__all__ = [
    "MetricsWriter",
    "RunContext",
    "create_run_dir",
    "environment_metadata",
    "setup_logging",
]

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def _timestamp_id() -> str:
    """Return a filesystem-safe timestamp used as a default run id."""
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def create_run_dir(
    output_root: "Path | str",
    experiment_name: str,
    run_id: Optional[str] = None,
    exist_ok: bool = False,
) -> Path:
    """Create (and return) the directory tree for one run.

    Args:
        output_root: Root directory for all experiment outputs.
        experiment_name: Experiment name; becomes the second path component.
        run_id: Explicit run identifier.  Defaults to a timestamp, made unique
            with a numeric suffix if that directory already exists.
        exist_ok: Reuse an existing ``run_id`` directory instead of uniquifying
            it (used when resuming a run).

    Returns:
        The path of the run directory, with ``checkpoints/`` and ``artifacts/``
        already created.
    """
    root = Path(output_root) / experiment_name
    root.mkdir(parents=True, exist_ok=True)

    candidate_id = run_id or _timestamp_id()
    run_dir = root / candidate_id
    if run_dir.exists() and not exist_ok:
        suffix = 1
        while (root / f"{candidate_id}-{suffix}").exists():
            suffix += 1
        run_dir = root / f"{candidate_id}-{suffix}"

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    (run_dir / "artifacts").mkdir(exist_ok=True)
    return run_dir


def setup_logging(
    log_dir: Optional["Path | str"] = None,
    name: str = "salsa",
    level: str = "INFO",
    console: bool = True,
    filename: str = "run.log",
) -> logging.Logger:
    """Configure and return a logger that writes to a file and the console.

    Calling this repeatedly with the same ``name`` replaces the handlers rather
    than stacking duplicates.

    Args:
        log_dir: Directory for the log file (None disables file logging).
        name: Logger name.
        level: Logging level name, e.g. ``"INFO"`` or ``"DEBUG"``.
        console: Also emit records to stdout.
        filename: Log file name inside ``log_dir``.

    Returns:
        The configured :class:`logging.Logger`.
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, str(level).upper(), logging.INFO))
    logger.propagate = False

    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    if console:
        stream_handler = logging.StreamHandler(stream=sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    if log_dir is not None:
        directory = Path(log_dir)
        directory.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(
            directory / filename, mode="a", encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


class MetricsWriter:
    """Append-only JSONL metric stream with a mirrored CSV table.

    Example:
        >>> writer = MetricsWriter(run_dir)                  # doctest: +SKIP
        >>> writer.log({"epoch": 0, "valid_loss": 5.12})     # doctest: +SKIP
    """

    def __init__(self, run_dir: "Path | str", stem: str = "metrics") -> None:
        """Create a writer that stores ``<stem>.jsonl`` / ``<stem>.csv``."""
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.run_dir / f"{stem}.jsonl"
        self.csv_path = self.run_dir / f"{stem}.csv"
        self._start_time = time.time()
        self._rows: List[Dict[str, Any]] = []
        self._columns: List[str] = []
        if self.jsonl_path.exists():
            self._load_existing()

    def _load_existing(self) -> None:
        """Reload previously written rows so a resumed run keeps its history."""
        with self.jsonl_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                self._rows.append(row)
                for key in row:
                    if key not in self._columns:
                        self._columns.append(key)

    @property
    def rows(self) -> List[Dict[str, Any]]:
        """Return all rows logged so far (including reloaded ones)."""
        return list(self._rows)

    def log(self, metrics: Mapping[str, Any], step: Optional[int] = None) -> Dict[str, Any]:
        """Record one metric row.

        Args:
            metrics: Mapping of metric name to a JSON-serialisable value.
            step: Optional global step recorded alongside the metrics.

        Returns:
            The row as written, including ``wall_time_sec``.
        """
        row: Dict[str, Any] = dict(metrics)
        if step is not None:
            row.setdefault("step", step)
        row.setdefault("wall_time_sec", round(time.time() - self._start_time, 3))

        self._rows.append(row)
        new_columns = [k for k in row if k not in self._columns]
        self._columns.extend(new_columns)

        with self.jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, default=str) + "\n")

        if new_columns or not self.csv_path.exists():
            self._rewrite_csv()
        else:
            with self.csv_path.open("a", encoding="utf-8", newline="") as handle:
                csv.DictWriter(handle, fieldnames=self._columns).writerow(
                    {k: row.get(k, "") for k in self._columns}
                )
        return row

    def _rewrite_csv(self) -> None:
        """Rewrite the whole CSV table (used when the column set grows)."""
        with self.csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self._columns)
            writer.writeheader()
            for row in self._rows:
                writer.writerow({k: row.get(k, "") for k in self._columns})

    def close(self) -> None:
        """Flush state.  Present so callers can treat this like a file handle."""
        self._rewrite_csv()


def _git_commit() -> Optional[str]:
    """Return the current git commit hash, or None outside a repository."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    commit = result.stdout.strip()
    return commit or None


def _package_versions(packages: Sequence[str]) -> Dict[str, Optional[str]]:
    """Return installed versions for ``packages`` (None when absent)."""
    from importlib import metadata

    versions: Dict[str, Optional[str]] = {}
    for name in packages:
        try:
            versions[name] = metadata.version(name)
        except Exception:
            versions[name] = None
    return versions


def environment_metadata(
    config: Optional[Any] = None, extra: Optional[Mapping[str, Any]] = None
) -> Dict[str, Any]:
    """Collect everything needed to reproduce a run.

    Args:
        config: Optional :class:`~salsa.utils.config.Config`; its fingerprint
            and full dictionary are embedded when given.
        extra: Additional key/values to merge into the metadata.

    Returns:
        A JSON-serialisable metadata dictionary.
    """
    from . import device as device_utils

    metadata: Dict[str, Any] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "python_version": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
        "command": " ".join(sys.argv),
        "cwd": str(Path.cwd()),
        "git_commit": _git_commit(),
        "packages": _package_versions(("torch", "numpy", "pyyaml", "psutil", "pytest")),
        "device_report": device_utils.device_report(),
    }

    if config is not None:
        try:
            metadata["config_fingerprint"] = config.fingerprint()
            metadata["config"] = config.to_dict()
            metadata["seed"] = config.experiment.seed
        except AttributeError:  # pragma: no cover - defensive
            metadata["config"] = str(config)

    if extra:
        metadata.update(dict(extra))
    return metadata


@dataclass
class RunContext:
    """Bundle of the artefacts every phase of a run needs.

    Attributes:
        run_dir: Root directory of this run.
        logger: Configured logger writing to ``run_dir/run.log``.
        metrics: Metric stream writer.
        config: The resolved configuration for this run.
    """

    run_dir: Path
    logger: logging.Logger
    metrics: MetricsWriter
    config: Any

    @property
    def checkpoint_dir(self) -> Path:
        """Directory holding this run's checkpoints."""
        return self.run_dir / "checkpoints"

    @property
    def artifact_dir(self) -> Path:
        """Directory holding this run's non-checkpoint artefacts."""
        return self.run_dir / "artifacts"

    @classmethod
    def create(
        cls,
        config: Any,
        run_id: Optional[str] = None,
        log_level: str = "INFO",
        console: bool = True,
        exist_ok: bool = False,
    ) -> "RunContext":
        """Create a run directory and populate it with config and metadata.

        Args:
            config: A resolved :class:`~salsa.utils.config.Config`.
            run_id: Explicit run identifier (defaults to a timestamp).
            log_level: Logging level for the run logger.
            console: Also log to stdout.
            exist_ok: Reuse an existing run directory (resuming).

        Returns:
            The initialised :class:`RunContext`.
        """
        from .config import save_config

        run_dir = create_run_dir(
            config.experiment.output_root,
            config.experiment.name,
            run_id=run_id,
            exist_ok=exist_ok,
        )
        logger = setup_logging(run_dir, name="salsa", level=log_level, console=console)
        save_config(config, run_dir / "config.yaml")

        metadata = environment_metadata(config)
        with (run_dir / "metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, default=str)

        logger.info("Run directory: %s", run_dir)
        logger.info("Config fingerprint: %s", metadata.get("config_fingerprint", "n/a"))

        return cls(
            run_dir=run_dir,
            logger=logger,
            metrics=MetricsWriter(run_dir),
            config=config,
        )

    def save_json(self, name: str, payload: Any) -> Path:
        """Write ``payload`` as JSON into the run's artifact directory."""
        path = self.artifact_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, default=str)
        return path

    def close(self) -> None:
        """Close metric and log handlers."""
        self.metrics.close()
        for handler in list(self.logger.handlers):
            handler.close()
            self.logger.removeHandler(handler)
