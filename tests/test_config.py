"""Tests for the configuration system.

These tests protect the scientific rules of the project:
* unknown keys must be rejected (no silent drift in q / sigma / secrets);
* every shipped config must load, validate and round-trip;
* every model config in the sweep must share one identical LWE block;
* the 4-5M parameter budget must be present as a hard ceiling.
"""

from pathlib import Path

import pytest

from salsa.utils import config as config_module
from salsa.utils.config import (
    Config,
    ConfigError,
    apply_overrides,
    config_fingerprint,
    load_config,
    save_config,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"

SWEEP_CONFIGS = [
    "baseline_cpu.yaml",
    "small_30m.yaml",
    "small_15m.yaml",
    "small_8m.yaml",
    "target_4_5m.yaml",
]
ALL_CONFIGS = ["base.yaml"] + SWEEP_CONFIGS + ["smoke_cpu.yaml"]


# --------------------------------------------------------------------------- #
# Shipped configuration files
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", ALL_CONFIGS)
def test_shipped_configs_load_and_validate(name: str) -> None:
    """Every config in configs/ loads and passes validation."""
    cfg = load_config(CONFIG_DIR / name)
    warnings = cfg.validate()
    assert isinstance(warnings, list)
    assert cfg.experiment.name


@pytest.mark.parametrize("name", SWEEP_CONFIGS)
def test_sweep_shares_identical_lwe_instance(name: str) -> None:
    """The cryptographic problem must be identical across the model sweep."""
    base = load_config(CONFIG_DIR / "base.yaml")
    cfg = load_config(CONFIG_DIR / name)
    assert cfg.lwe == base.lwe, "sweep configs must not alter the LWE instance"
    assert cfg.encoding == base.encoding, "sweep configs must not alter the encoding"
    assert cfg.evaluation == base.evaluation
    assert cfg.verification == base.verification


def test_target_config_enforces_the_hard_budget() -> None:
    """The primary target config declares the 4-5M parameter budget."""
    cfg = load_config(CONFIG_DIR / "target_4_5m.yaml")
    assert cfg.model.max_parameters == 5_000_000
    assert cfg.model.min_parameters == 4_000_000
    assert cfg.model.target_parameters is not None
    assert cfg.model.min_parameters <= cfg.model.target_parameters
    assert cfg.model.target_parameters <= cfg.model.max_parameters


def test_configs_default_to_cpu() -> None:
    """CPU-first: no shipped config may silently request a GPU."""
    for name in ALL_CONFIGS:
        cfg = load_config(CONFIG_DIR / name)
        assert cfg.device.prefer == "cpu", f"{name} does not default to CPU"


# --------------------------------------------------------------------------- #
# Strictness
# --------------------------------------------------------------------------- #
def test_unknown_top_level_key_is_rejected() -> None:
    """A stray top-level section is an error, not a silent no-op."""
    with pytest.raises(ConfigError, match="unknown key"):
        Config.from_dict({"lwe": {"n": 30}, "not_a_section": {}})


def test_unknown_nested_key_is_rejected() -> None:
    """A typo inside a section names the offending path."""
    with pytest.raises(ConfigError, match=r"lwe: unknown key"):
        Config.from_dict({"lwe": {"sigmaa": 3.0}})


def test_type_errors_are_reported() -> None:
    """Wrong scalar types raise rather than coercing nonsense."""
    with pytest.raises(ConfigError):
        Config.from_dict({"lwe": {"n": "thirty"}})
    with pytest.raises(ConfigError):
        Config.from_dict({"lwe": {"n": True}})  # bool is not an int here
    with pytest.raises(ConfigError):
        Config.from_dict({"experiment": {"deterministic": "yes"}})


def test_numeric_coercion_is_allowed_when_lossless() -> None:
    """3.0 -> 3 is fine; 3 -> 3.0 is fine."""
    cfg = Config.from_dict({"lwe": {"n": 30.0, "sigma": 3}})
    assert cfg.lwe.n == 30 and isinstance(cfg.lwe.n, int)
    assert cfg.lwe.sigma == 3.0 and isinstance(cfg.lwe.sigma, float)


def test_missing_file_raises_config_error(tmp_path: Path) -> None:
    """A missing config file produces a ConfigError, not a bare OSError."""
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "does_not_exist.yaml")


# --------------------------------------------------------------------------- #
# Inheritance and overrides
# --------------------------------------------------------------------------- #
def test_extends_merges_deeply(tmp_path: Path) -> None:
    """A child config overrides single keys without dropping its siblings."""
    (tmp_path / "parent.yaml").write_text(
        "lwe:\n  n: 30\n  q: 251\n  sigma: 3.0\n", encoding="utf-8"
    )
    (tmp_path / "child.yaml").write_text(
        "extends: parent.yaml\nlwe:\n  n: 50\n", encoding="utf-8"
    )
    cfg = load_config(tmp_path / "child.yaml")
    assert cfg.lwe.n == 50
    assert cfg.lwe.q == 251
    assert cfg.lwe.sigma == 3.0


def test_extends_detects_cycles(tmp_path: Path) -> None:
    """A circular extends chain is reported instead of hanging."""
    (tmp_path / "a.yaml").write_text("extends: b.yaml\n", encoding="utf-8")
    (tmp_path / "b.yaml").write_text("extends: a.yaml\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="Circular"):
        load_config(tmp_path / "a.yaml")


def test_cli_overrides_apply_last() -> None:
    """Dotted CLI overrides are parsed with YAML scalar rules."""
    cfg = load_config(
        CONFIG_DIR / "target_4_5m.yaml",
        overrides=["training.batch_size=64", "lwe.n=50", "experiment.name=ovr"],
    )
    assert cfg.training.batch_size == 64
    assert cfg.lwe.n == 50
    assert cfg.experiment.name == "ovr"


def test_override_parses_lists_and_nulls() -> None:
    """Overrides support YAML literals, not just strings."""
    raw = apply_overrides({}, ["a.b=[1, 2]", "a.c=null", "a.d=true", "a.e=1.5"])
    assert raw["a"] == {"b": [1, 2], "c": None, "d": True, "e": 1.5}


def test_malformed_override_is_rejected() -> None:
    """An override without '=' is an error."""
    with pytest.raises(ConfigError, match="Malformed override"):
        apply_overrides({}, ["training.batch_size"])


# --------------------------------------------------------------------------- #
# Validation rules
# --------------------------------------------------------------------------- #
def test_secret_sparsity_must_be_explicit() -> None:
    """Exactly one of hamming_weight / density must be given."""
    both = load_config(CONFIG_DIR / "base.yaml", overrides=["lwe.density=0.1"])
    with pytest.raises(ConfigError, match="exactly one"):
        both.validate()

    neither = load_config(
        CONFIG_DIR / "base.yaml", overrides=["lwe.hamming_weight=null"]
    )
    with pytest.raises(ConfigError, match="exactly one"):
        neither.validate()


def test_density_resolves_to_hamming_weight() -> None:
    """Density is converted to an integer Hamming weight consistently."""
    cfg = load_config(
        CONFIG_DIR / "base.yaml",
        overrides=["lwe.hamming_weight=null", "lwe.density=0.1", "lwe.n=50"],
    )
    cfg.validate()
    assert cfg.lwe.resolved_hamming_weight == 5
    assert cfg.lwe.resolved_density == pytest.approx(0.1)


def test_hamming_weight_cannot_exceed_dimension() -> None:
    """h > n is impossible for a binary secret."""
    cfg = load_config(CONFIG_DIR / "base.yaml", overrides=["lwe.hamming_weight=99"])
    with pytest.raises(ConfigError, match="Hamming weight"):
        cfg.validate()


def test_head_divisibility_is_enforced() -> None:
    """Model width must divide evenly among attention heads."""
    cfg = load_config(CONFIG_DIR / "target_4_5m.yaml", overrides=["model.encoder_heads=7"])
    with pytest.raises(ConfigError, match="divisible"):
        cfg.validate()


def test_target_parameters_cannot_exceed_the_ceiling() -> None:
    """A nominal target above the hard ceiling is a contradiction."""
    cfg = load_config(
        CONFIG_DIR / "target_4_5m.yaml", overrides=["model.target_parameters=9000000"]
    )
    with pytest.raises(ConfigError, match="target_parameters"):
        cfg.validate()


def test_non_prime_modulus_warns_but_does_not_fail() -> None:
    """q = 250 is allowed but must be flagged as a deviation from the paper."""
    cfg = load_config(CONFIG_DIR / "base.yaml", overrides=["lwe.q=250"])
    warnings = cfg.validate()
    assert any("not prime" in w for w in warnings)


def test_indiscriminate_verification_threshold_warns() -> None:
    """A threshold that cannot separate correct from wrong secrets is flagged."""
    cfg = load_config(
        CONFIG_DIR / "base.yaml", overrides=["verification.std_tolerance_factor=30"]
    )
    warnings = cfg.validate()
    assert any("verification" in w for w in warnings)


def test_bad_device_preference_is_rejected() -> None:
    """Only cpu/auto/cuda/mps are valid device preferences."""
    cfg = load_config(CONFIG_DIR / "base.yaml", overrides=["device.prefer=tpu"])
    with pytest.raises(ConfigError, match="device.prefer"):
        cfg.validate()


def test_tolerance_bounds_are_enforced() -> None:
    """acc_tau tolerance must be a sensible fraction of q."""
    cfg = load_config(CONFIG_DIR / "base.yaml", overrides=["evaluation.tolerance=0.9"])
    with pytest.raises(ConfigError, match="tolerance"):
        cfg.validate()


# --------------------------------------------------------------------------- #
# Serialisation and fingerprints
# --------------------------------------------------------------------------- #
def test_roundtrip_preserves_every_field(tmp_path: Path) -> None:
    """Saving and reloading a config yields an identical object."""
    cfg = load_config(CONFIG_DIR / "target_4_5m.yaml")
    path = save_config(cfg, tmp_path / "resolved.yaml")
    reloaded = Config.from_yaml(path)
    assert reloaded == cfg
    assert reloaded.fingerprint() == cfg.fingerprint()


def test_fingerprint_is_stable_and_sensitive() -> None:
    """The fingerprint identifies the settings, not the file formatting."""
    a = load_config(CONFIG_DIR / "target_4_5m.yaml")
    b = load_config(CONFIG_DIR / "target_4_5m.yaml")
    assert config_fingerprint(a) == config_fingerprint(b)

    changed = load_config(CONFIG_DIR / "target_4_5m.yaml", overrides=["lwe.sigma=4.0"])
    assert config_fingerprint(changed) != config_fingerprint(a)
    assert len(config_fingerprint(a)) == 16


def test_defaults_match_the_salsa_reference_setting() -> None:
    """Code defaults mirror the paper so an omitted key is never a surprise."""
    cfg = Config()
    assert cfg.lwe.q == 251
    assert cfg.lwe.sigma == 3.0
    assert cfg.lwe.secret_distribution == "binary"
    assert cfg.encoding.base == 81
    assert cfg.evaluation.tolerance == 0.1
    assert cfg.evaluation.beam_size == 1
    assert cfg.device.prefer == "cpu"
    assert cfg.model.max_parameters == 5_000_000


def test_is_prime_helper() -> None:
    """Sanity-check the advisory primality test."""
    assert config_module._is_prime(251)
    assert config_module._is_prime(2)
    assert not config_module._is_prime(1)
    assert not config_module._is_prime(250)
    assert not config_module._is_prime(9)
