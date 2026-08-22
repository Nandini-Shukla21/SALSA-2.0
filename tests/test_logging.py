"""Tests for run directories, file logging and metric streams."""

import json
from pathlib import Path

from salsa.utils.config import load_config
from salsa.utils.logging import (
    MetricsWriter,
    RunContext,
    create_run_dir,
    environment_metadata,
    setup_logging,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"


def test_create_run_dir_builds_the_expected_tree(tmp_path: Path) -> None:
    """A run directory always contains checkpoints/ and artifacts/."""
    run_dir = create_run_dir(tmp_path, "exp", run_id="run1")
    assert run_dir == tmp_path / "exp" / "run1"
    assert (run_dir / "checkpoints").is_dir()
    assert (run_dir / "artifacts").is_dir()


def test_create_run_dir_never_overwrites_an_existing_run(tmp_path: Path) -> None:
    """Re-using a run id produces a new directory, protecting old results."""
    first = create_run_dir(tmp_path, "exp", run_id="run1")
    second = create_run_dir(tmp_path, "exp", run_id="run1")
    assert first != second
    assert second.name.startswith("run1-")


def test_create_run_dir_can_reuse_for_resume(tmp_path: Path) -> None:
    """exist_ok=True reuses the directory (needed to resume a run)."""
    first = create_run_dir(tmp_path, "exp", run_id="run1")
    second = create_run_dir(tmp_path, "exp", run_id="run1", exist_ok=True)
    assert first == second


def test_setup_logging_writes_to_file_without_duplicating_handlers(
    tmp_path: Path,
) -> None:
    """Repeated setup must not multiply log lines."""
    logger = setup_logging(tmp_path, name="salsa-test", console=False)
    logger.info("first message")
    logger = setup_logging(tmp_path, name="salsa-test", console=False)
    logger.info("second message")

    for handler in logger.handlers:
        handler.flush()
    text = (tmp_path / "run.log").read_text(encoding="utf-8")
    assert text.count("first message") == 1
    assert text.count("second message") == 1


def test_metrics_writer_streams_jsonl_and_csv(tmp_path: Path) -> None:
    """Every logged row lands in both the JSONL stream and the CSV table."""
    writer = MetricsWriter(tmp_path)
    writer.log({"epoch": 0, "valid_loss": 5.5})
    writer.log({"epoch": 1, "valid_loss": 4.25})

    lines = writer.jsonl_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[1])["valid_loss"] == 4.25
    assert "wall_time_sec" in json.loads(lines[0])

    csv_text = writer.csv_path.read_text(encoding="utf-8")
    assert "valid_loss" in csv_text.splitlines()[0]
    assert len(csv_text.strip().splitlines()) == 3  # header + 2 rows


def test_metrics_writer_handles_new_columns_mid_run(tmp_path: Path) -> None:
    """Adding a metric later must not corrupt the CSV (a real SALSA bug)."""
    writer = MetricsWriter(tmp_path)
    writer.log({"epoch": 0, "valid_loss": 5.5})
    writer.log({"epoch": 1, "valid_loss": 4.0, "secret_recovered": False})

    header = writer.csv_path.read_text(encoding="utf-8").splitlines()[0]
    assert "secret_recovered" in header
    assert len(writer.csv_path.read_text(encoding="utf-8").strip().splitlines()) == 3


def test_metrics_writer_reloads_history_on_resume(tmp_path: Path) -> None:
    """A resumed run keeps its earlier metric history."""
    first = MetricsWriter(tmp_path)
    first.log({"epoch": 0, "valid_loss": 5.5})

    resumed = MetricsWriter(tmp_path)
    assert len(resumed.rows) == 1
    resumed.log({"epoch": 1, "valid_loss": 4.0})
    assert len(resumed.rows) == 2
    lines = resumed.jsonl_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2


def test_environment_metadata_captures_reproducibility_information() -> None:
    """Metadata must pin down the code, config and machine."""
    cfg = load_config(CONFIG_DIR / "smoke_cpu.yaml")
    metadata = environment_metadata(cfg)
    for key in (
        "created_at",
        "python_version",
        "platform",
        "command",
        "packages",
        "device_report",
        "config_fingerprint",
        "config",
        "seed",
    ):
        assert key in metadata
    assert metadata["config_fingerprint"] == cfg.fingerprint()
    assert metadata["seed"] == cfg.experiment.seed
    json.dumps(metadata, default=str)  # must be serialisable


def test_run_context_creates_a_self_contained_run(tmp_path: Path) -> None:
    """RunContext.create writes config, metadata and a log in one place."""
    cfg = load_config(
        CONFIG_DIR / "smoke_cpu.yaml",
        overrides=[f"experiment.output_root={tmp_path.as_posix()}"],
    )
    context = RunContext.create(cfg, run_id="unit-test", console=False)
    try:
        context.logger.info("hello")
        context.metrics.log({"epoch": 0, "valid_loss": 9.9})
        artifact = context.save_json("candidate.json", {"secret": [0, 1, 0]})

        assert (context.run_dir / "config.yaml").is_file()
        assert (context.run_dir / "metadata.json").is_file()
        assert (context.run_dir / "run.log").is_file()
        assert (context.run_dir / "metrics.jsonl").is_file()
        assert context.checkpoint_dir.is_dir()
        assert artifact.is_file()

        metadata = json.loads((context.run_dir / "metadata.json").read_text("utf-8"))
        assert metadata["config_fingerprint"] == cfg.fingerprint()

        # The saved config must reproduce the run exactly.
        from salsa.utils.config import Config

        saved = Config.from_yaml(context.run_dir / "config.yaml")
        assert saved == cfg
    finally:
        context.close()
