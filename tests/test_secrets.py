"""Tests for secret generation.

The property that matters most: the Hamming weight is **exact**, never merely
expected.  A secret whose weight drifts from run to run would silently change
the difficulty of the attack.
"""

from pathlib import Path

import numpy as np
import pytest

from salsa.data.secrets import (
    BINARY,
    TERNARY,
    generate_binary_secret,
    generate_secret,
    generate_ternary_secret,
    hamming_weight,
    is_valid_secret,
    resolve_rng,
    secret_from_config,
)
from salsa.utils.config import load_config

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"

DIMENSIONS = [4, 12, 30, 50, 128]


# --------------------------------------------------------------------------- #
# Exact Hamming weight
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n", DIMENSIONS)
@pytest.mark.parametrize("h", [1, 3, 5])
def test_binary_secret_has_exactly_h_ones(n: int, h: int) -> None:
    """Every generated binary secret has exactly h ones."""
    if h > n:
        pytest.skip(f"h={h} is impossible for n={n}")
    for seed in range(10):
        secret = generate_binary_secret(n, h, seed=seed)
        assert secret.shape == (n,)
        assert int(secret.sum()) == h
        assert hamming_weight(secret) == h


@pytest.mark.parametrize("n", DIMENSIONS)
def test_binary_secret_with_h_equal_n_is_all_ones(n: int) -> None:
    """h = n is the fully dense corner case."""
    secret = generate_binary_secret(n, n, seed=1)
    assert int(secret.sum()) == n
    assert np.array_equal(secret, np.ones(n, dtype=np.int64))


def test_binary_secret_with_h_zero_is_all_zeros() -> None:
    """h = 0 is allowed (a degenerate instance) and gives the zero vector."""
    secret = generate_binary_secret(10, 0, seed=0)
    assert int(secret.sum()) == 0


@pytest.mark.parametrize("n,h", [(10, 3), (30, 3), (50, 5)])
def test_hamming_weight_is_exact_over_many_draws(n: int, h: int) -> None:
    """The weight never fluctuates - it is a count, not an expectation."""
    weights = {hamming_weight(generate_binary_secret(n, h, seed=s)) for s in range(500)}
    assert weights == {h}


# --------------------------------------------------------------------------- #
# Alphabet
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n", DIMENSIONS)
def test_binary_secret_alphabet(n: int) -> None:
    """Binary secrets contain only 0 and 1."""
    secret = generate_binary_secret(n, 3, seed=7)
    assert set(np.unique(secret).tolist()).issubset({0, 1})
    assert secret.dtype == np.int64
    assert is_valid_secret(secret, BINARY, expected_weight=3)


@pytest.mark.parametrize("n", [12, 30, 50])
def test_ternary_secret_alphabet_and_weight(n: int) -> None:
    """Ternary secrets have exactly h non-zeros, each in {-1, +1}."""
    secret = generate_ternary_secret(n, 5, seed=3)
    assert set(np.unique(secret).tolist()).issubset({-1, 0, 1})
    assert hamming_weight(secret) == 5
    assert is_valid_secret(secret, TERNARY, expected_weight=5)


def test_ternary_secret_uses_both_signs() -> None:
    """Over many draws both signs occur (they are not silently all +1)."""
    signs = set()
    for seed in range(50):
        secret = generate_ternary_secret(20, 6, seed=seed)
        signs.update(np.unique(secret[secret != 0]).tolist())
    assert signs == {-1, 1}


def test_is_valid_secret_rejects_bad_alphabets() -> None:
    """A ternary value is not a valid binary secret."""
    assert not is_valid_secret(np.array([0, 1, 2]), BINARY)
    assert not is_valid_secret(np.array([0, -1, 1]), BINARY)
    assert is_valid_secret(np.array([0, -1, 1]), TERNARY)
    assert not is_valid_secret(np.array([[0, 1], [1, 0]]), BINARY)
    assert not is_valid_secret(np.array([0, 1, 1]), BINARY, expected_weight=3)


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n", DIMENSIONS)
def test_same_seed_gives_the_same_secret(n: int) -> None:
    """Reproducibility rule: seed in, identical secret out."""
    first = generate_binary_secret(n, 3, seed=1234)
    second = generate_binary_secret(n, 3, seed=1234)
    assert np.array_equal(first, second)


def test_different_seeds_give_different_secrets() -> None:
    """Different seeds normally give different supports (n = 50, h = 3)."""
    secrets = {generate_binary_secret(50, 3, seed=s).tobytes() for s in range(20)}
    assert len(secrets) > 15


def test_generator_and_seed_are_interchangeable() -> None:
    """Passing a Generator is equivalent to passing its seed."""
    from_seed = generate_binary_secret(30, 3, seed=99)
    from_rng = generate_binary_secret(30, 3, rng=np.random.default_rng(99))
    assert np.array_equal(from_seed, from_rng)


def test_support_is_not_biased_to_low_indices() -> None:
    """Every coordinate can carry a one (the support is a uniform subset)."""
    seen = np.zeros(10, dtype=np.int64)
    for seed in range(300):
        seen += generate_binary_secret(10, 1, seed=seed)
    assert int(seen.min()) > 0, f"some coordinate never selected: {seen}"


# --------------------------------------------------------------------------- #
# Dispatch, validation and config integration
# --------------------------------------------------------------------------- #
def test_generate_secret_dispatches_on_distribution() -> None:
    """generate_secret routes to the binary / ternary samplers."""
    binary = generate_secret(20, 4, distribution=BINARY, seed=0)
    ternary = generate_secret(20, 4, distribution=TERNARY, seed=0)
    assert set(np.unique(binary).tolist()).issubset({0, 1})
    assert hamming_weight(binary) == hamming_weight(ternary) == 4


def test_unknown_distribution_is_rejected() -> None:
    """An unknown secret distribution is an error, never a silent default."""
    with pytest.raises(ValueError, match="Unknown secret distribution"):
        generate_secret(10, 3, distribution="gaussian", seed=0)


@pytest.mark.parametrize(
    "n,h",
    [(0, 0), (-1, 1), (10, 11), (10, -1)],
)
def test_invalid_parameters_are_rejected(n: int, h: int) -> None:
    """Impossible (n, h) combinations raise instead of producing junk."""
    with pytest.raises(ValueError):
        generate_binary_secret(n, h, seed=0)


def test_resolve_rng_rejects_bad_types() -> None:
    """resolve_rng only accepts a Generator, an int or None."""
    assert isinstance(resolve_rng(seed=0), np.random.Generator)
    assert isinstance(resolve_rng(5), np.random.Generator)
    with pytest.raises(TypeError):
        resolve_rng("not-a-seed")


def test_secret_from_config_is_stable_and_matches_the_config() -> None:
    """The experiment's secret follows from its config and master seed alone."""
    cfg = load_config(CONFIG_DIR / "base.yaml")
    first = secret_from_config(cfg)
    second = secret_from_config(cfg)
    assert np.array_equal(first, second)
    assert first.shape == (cfg.lwe.n,)
    assert hamming_weight(first) == cfg.lwe.resolved_hamming_weight


def test_secret_changes_with_the_master_seed() -> None:
    """A different master seed means a different secret."""
    cfg_a = load_config(CONFIG_DIR / "base.yaml", overrides=["experiment.seed=0"])
    cfg_b = load_config(CONFIG_DIR / "base.yaml", overrides=["experiment.seed=1"])
    assert not np.array_equal(secret_from_config(cfg_a), secret_from_config(cfg_b))


def test_different_secret_indices_give_different_secrets() -> None:
    """Multi-secret experiments get independent secrets."""
    cfg = load_config(CONFIG_DIR / "base.yaml")
    assert not np.array_equal(
        secret_from_config(cfg, secret_index=0), secret_from_config(cfg, secret_index=1)
    )
