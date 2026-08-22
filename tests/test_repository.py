"""Structural tests: the repository skeleton itself is part of the contract."""

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

EXPECTED_PACKAGES = [
    "salsa",
    "salsa/data",
    "salsa/models",
    "salsa/training",
    "salsa/recovery",
    "salsa/verification",
    "salsa/evaluation",
    "salsa/utils",
]

EXPECTED_FILES = [
    "README.md",
    "requirements.txt",
    "pyproject.toml",
    ".gitignore",
    "configs/base.yaml",
    "configs/baseline_cpu.yaml",
    "configs/small_30m.yaml",
    "configs/small_15m.yaml",
    "configs/small_8m.yaml",
    "configs/target_4_5m.yaml",
    "scripts/show_config.py",
]


@pytest.mark.parametrize("package", EXPECTED_PACKAGES)
def test_packages_are_importable(package: str) -> None:
    """Every planned subpackage exists and has an __init__.py."""
    path = REPO_ROOT / package
    assert path.is_dir(), f"missing package directory: {package}"
    assert (path / "__init__.py").is_file(), f"missing __init__.py in {package}"


@pytest.mark.parametrize("relative", EXPECTED_FILES)
def test_expected_files_exist(relative: str) -> None:
    """The phase-1 deliverables are present."""
    assert (REPO_ROOT / relative).is_file(), f"missing file: {relative}"


def test_package_imports_cleanly() -> None:
    """Importing the package must not require torch or any optional extra."""
    import salsa

    assert salsa.__version__


def test_utils_public_api() -> None:
    """The phase-1 public API is stable for later phases to build on."""
    from salsa import utils

    for name in (
        "Config",
        "ConfigError",
        "load_config",
        "save_config",
        "config_fingerprint",
        "resolve_device",
        "device_report",
        "RunContext",
        "MetricsWriter",
        "setup_logging",
    ):
        assert hasattr(utils, name), f"salsa.utils is missing {name}"


def test_no_hardcoded_absolute_paths() -> None:
    """No Windows drive letters or POSIX home paths baked into the source."""
    offenders = []
    for path in (REPO_ROOT / "salsa").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for marker in ("C:\\Users", "/home/", "/Users/", "/private/home"):
            if marker in text:
                offenders.append(f"{path.name}: {marker}")
    assert not offenders, f"hard-coded absolute paths found: {offenders}"
