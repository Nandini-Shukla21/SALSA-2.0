"""Tests for LWE sample generation.

Every generated sample is checked *mathematically* -- ``b == (A s + e) mod q``
for all instances -- not merely for its shape.  The official experiment
parameters (q = 251, sigma = 3, sparse binary secret, h = 3) are used for the
main tests; alternative q / sigma values are exercised in a clearly separated
section so they cannot be mistaken for the experiment configuration.
"""

import dataclasses
import inspect
from pathlib import Path
from typing import Iterator

import numpy as np
import pytest

from salsa.data import build_problem
from salsa.data.lwe import (
    GroundTruth,
    LabeledSample,
    LWEParams,
    LWEProblem,
    LWESample,
    generate_batch,
    generate_error,
    generate_lwe_sample,
    generate_uniform_matrix,
)
from salsa.data.secrets import generate_binary_secret, hamming_weight
from salsa.utils.config import load_config

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"

# The official experiment configuration.
Q = 251
SIGMA = 3.0
H = 3
DIMENSIONS = [4, 12, 30, 50, 128]


def official_params(n: int, **overrides) -> LWEParams:
    """Return the official parameter set for dimension ``n``."""
    settings = dict(
        n=n,
        q=Q,
        sigma=SIGMA,
        secret_distribution="binary",
        hamming_weight=H,
        structure="lwe",
    )
    settings.update(overrides)
    return LWEParams(**settings)


# --------------------------------------------------------------------------- #
# The LWE relation itself
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n", DIMENSIONS)
def test_lwe_relation_holds_exactly(n: int) -> None:
    """b == (A @ s + e) mod q for every instance, checked element-wise."""
    params = official_params(n)
    sample = generate_lwe_sample(params, num_instances=64, seed=0)

    expected = (sample.A @ sample.secret + sample.error) % Q
    assert np.array_equal(expected, sample.b)
    assert sample.verify_relation()


@pytest.mark.parametrize("n", DIMENSIONS)
def test_shapes_and_dtypes(n: int) -> None:
    """A is (m, n), b and e are (m,), s is (n,); everything is int64."""
    m = 32
    sample = generate_lwe_sample(official_params(n), num_instances=m, seed=1)
    assert sample.A.shape == (m, n)
    assert sample.b.shape == (m,)
    assert sample.error.shape == (m,)
    assert sample.secret.shape == (n,)
    for array in (sample.A, sample.b, sample.error, sample.secret):
        assert array.dtype == np.int64
    assert sample.num_instances == m
    assert len(sample) == m
    assert sample.n == n


@pytest.mark.parametrize("n", DIMENSIONS)
def test_published_values_stay_in_the_modular_range(n: int) -> None:
    """A and b are always reduced into [0, q); the error is not."""
    sample = generate_lwe_sample(official_params(n), num_instances=128, seed=2)
    assert sample.A.min() >= 0 and sample.A.max() < Q
    assert sample.b.min() >= 0 and sample.b.max() < Q
    assert sample.public.in_range()
    # The error is centred and signed, so it must not be reduced mod q.
    assert sample.error.min() < 0


@pytest.mark.parametrize("n", DIMENSIONS)
def test_secret_has_the_configured_hamming_weight(n: int) -> None:
    """The generator does not quietly alter the secret's sparsity."""
    sample = generate_lwe_sample(official_params(n), num_instances=8, seed=3)
    assert hamming_weight(sample.secret) == H
    assert set(np.unique(sample.secret).tolist()).issubset({0, 1})


def test_supplied_secret_is_used_unchanged() -> None:
    """An explicitly supplied secret is used verbatim."""
    secret = generate_binary_secret(30, H, seed=42)
    sample = generate_lwe_sample(official_params(30), 16, secret=secret, seed=0)
    assert np.array_equal(sample.secret, secret)
    assert sample.verify_relation()


def test_wrong_shaped_secret_is_rejected() -> None:
    """A secret of the wrong length is an error, not a broadcast surprise."""
    with pytest.raises(ValueError, match="shape"):
        generate_lwe_sample(official_params(30), 4, secret=np.zeros(29), seed=0)


def test_secret_outside_the_alphabet_is_rejected() -> None:
    """A non-binary secret cannot be smuggled into a binary experiment."""
    bad = np.zeros(30, dtype=np.int64)
    bad[0] = 5
    with pytest.raises(ValueError, match="alphabet"):
        generate_lwe_sample(official_params(30), 4, secret=bad, seed=0)


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n", DIMENSIONS)
def test_same_seed_gives_an_identical_instance(n: int) -> None:
    """Same seed -> identical A, b, e and secret."""
    first = generate_lwe_sample(official_params(n), 32, seed=7)
    second = generate_lwe_sample(official_params(n), 32, seed=7)
    assert np.array_equal(first.A, second.A)
    assert np.array_equal(first.b, second.b)
    assert np.array_equal(first.error, second.error)
    assert np.array_equal(first.secret, second.secret)


@pytest.mark.parametrize("n", DIMENSIONS)
def test_different_seeds_give_a_different_instance(n: int) -> None:
    """Different seed -> different data (collision probability is negligible)."""
    first = generate_lwe_sample(official_params(n), 32, seed=7)
    second = generate_lwe_sample(official_params(n), 32, seed=8)
    assert not np.array_equal(first.A, second.A)
    assert not np.array_equal(first.b, second.b)


# --------------------------------------------------------------------------- #
# Error distribution
# --------------------------------------------------------------------------- #
def test_error_is_integer_centred_and_has_the_right_spread() -> None:
    """The discrete Gaussian has mean ~ 0 and standard deviation ~ sigma."""
    error = generate_error(50_000, sigma=SIGMA, seed=0)
    assert error.dtype == np.int64
    assert abs(float(error.mean())) < 0.15
    assert abs(float(error.std()) - SIGMA) < 0.15


def test_error_is_symmetric() -> None:
    """Positive and negative errors are equally likely."""
    error = generate_error(50_000, sigma=SIGMA, seed=1)
    positives = int((error > 0).sum())
    negatives = int((error < 0).sum())
    assert abs(positives - negatives) < 0.05 * len(error)


def test_error_respects_the_tail_cut() -> None:
    """No error term exceeds the configured truncation point."""
    error = generate_error(20_000, sigma=SIGMA, seed=2, tail_cut=4.0)
    assert int(np.abs(error).max()) <= int(np.ceil(4.0 * SIGMA))


def test_zero_sigma_and_none_distribution_give_no_error() -> None:
    """The error-free ablation really is error-free."""
    assert not generate_error(100, sigma=0.0, seed=0).any()
    assert not generate_error(100, sigma=SIGMA, distribution="none", seed=0).any()


def test_rounded_gaussian_is_available_and_distinct() -> None:
    """The alternative error model works and is not silently the default."""
    rounded = generate_error(20_000, sigma=SIGMA, distribution="rounded_gaussian", seed=0)
    assert rounded.dtype == np.int64
    assert abs(float(rounded.std()) - SIGMA) < 0.2
    exact = generate_error(20_000, sigma=SIGMA, seed=0)
    assert not np.array_equal(rounded, exact)


def test_unknown_error_distribution_is_rejected() -> None:
    """An unknown error model raises rather than falling back."""
    with pytest.raises(ValueError, match="Unknown error distribution"):
        generate_error(10, sigma=SIGMA, distribution="laplace", seed=0)


def test_error_reproducibility() -> None:
    """Same seed -> same error vector."""
    assert np.array_equal(
        generate_error(1000, SIGMA, seed=5), generate_error(1000, SIGMA, seed=5)
    )


# --------------------------------------------------------------------------- #
# Bounded-a ablation (must not be the default)
# --------------------------------------------------------------------------- #
def test_bounded_a_is_off_by_default() -> None:
    """The default instance samples a from the full range [0, q)."""
    params = official_params(30)
    assert params.max_a_fraction == 1.0
    assert params.max_a_value == Q
    assert params.bounded_a is False
    sample = generate_lwe_sample(params, 2000, seed=0)
    assert sample.A.max() > int(0.9 * Q), "default a should span the full range"


def test_shipped_configs_do_not_enable_bounded_a() -> None:
    """No shipped experiment config silently activates the ablation."""
    for path in sorted(CONFIG_DIR.glob("*.yaml")):
        cfg = load_config(path)
        assert cfg.lwe.max_a_fraction == 1.0, f"{path.name} enables bounded-a"


@pytest.mark.parametrize("fraction", [0.35, 0.4, 0.5, 0.65])
def test_bounded_a_limits_the_coefficients(fraction: float) -> None:
    """With the ablation on, every entry of a stays below floor(alpha * q)."""
    params = official_params(30, max_a_fraction=fraction)
    bound = int(fraction * Q)
    assert params.bounded_a is True
    assert params.max_a_value == bound

    sample = generate_lwe_sample(params, 2000, seed=0)
    assert sample.A.max() < bound
    assert sample.A.min() >= 0
    assert sample.verify_relation()


def test_bounded_a_still_reduces_b_mod_q() -> None:
    """Bounding a does not change the modular arithmetic on b."""
    sample = generate_lwe_sample(official_params(30, max_a_fraction=0.4), 512, seed=1)
    assert sample.b.min() >= 0 and sample.b.max() < Q


def test_invalid_bounds_are_rejected() -> None:
    """A bound of zero or above q is a configuration error."""
    with pytest.raises(ValueError):
        official_params(30, max_a_fraction=0.0)
    with pytest.raises(ValueError):
        official_params(30, max_a_fraction=1.5)
    with pytest.raises(ValueError):
        generate_uniform_matrix(4, 30, Q, seed=0, max_a_value=Q + 1)


# --------------------------------------------------------------------------- #
# Public / private separation
# --------------------------------------------------------------------------- #
def test_public_sample_carries_no_ground_truth() -> None:
    """LWESample must not expose the secret or the error, by construction."""
    field_names = {f.name for f in dataclasses.fields(LWESample)}
    assert field_names == {"A", "b", "q", "structure"}
    assert "secret" not in field_names and "error" not in field_names

    sample = generate_lwe_sample(official_params(12), 8, seed=0).as_public()
    assert isinstance(sample, LWESample)
    assert not hasattr(sample, "secret")
    assert not hasattr(sample, "error")


def test_ground_truth_is_reachable_only_through_the_labeled_wrapper() -> None:
    """The private data lives in its own type, held by LabeledSample."""
    labeled = generate_lwe_sample(official_params(12), 8, seed=0)
    assert isinstance(labeled, LabeledSample)
    assert isinstance(labeled.public, LWESample)
    assert isinstance(labeled.ground_truth, GroundTruth)
    assert labeled.ground_truth.hamming_weight == H


def test_problem_sample_returns_the_public_view_only() -> None:
    """problem.sample() is the method attack code will use."""
    problem = LWEProblem(official_params(12), data_seed=0)
    assert isinstance(problem.sample(8), LWESample)
    assert isinstance(problem.labeled_sample(8), LabeledSample)
    assert isinstance(problem.batch(0, 8), LWESample)
    assert isinstance(problem.batch(0, 8, labeled=True), LabeledSample)


def test_revealing_the_secret_is_explicit_and_returns_a_copy() -> None:
    """The only way to the secret is a loudly named method."""
    problem = LWEProblem(official_params(12), data_seed=0)
    revealed = problem.reveal_secret()
    revealed[:] = 0
    assert hamming_weight(problem.reveal_secret()) == H
    assert "controlled experiments only" in LWEProblem.reveal_secret.__doc__.lower()


def test_repr_does_not_leak_the_secret() -> None:
    """Logging a problem object must not print the secret."""
    problem = LWEProblem(official_params(12), data_seed=0)
    text = repr(problem)
    secret_text = str(problem.reveal_secret().tolist())
    assert secret_text not in text
    assert "n=12" in text and "q=251" in text


# --------------------------------------------------------------------------- #
# Streaming / batching
# --------------------------------------------------------------------------- #
def test_iter_batches_yields_exactly_the_requested_instances() -> None:
    """The stream length is exact, with a truncated final batch."""
    problem = LWEProblem(official_params(30), data_seed=0)
    batches = list(problem.iter_batches(num_samples=250, batch_size=64))
    assert [b.num_instances for b in batches] == [64, 64, 64, 58]
    assert sum(b.num_instances for b in batches) == 250


def test_iter_batches_is_lazy() -> None:
    """iter_batches is a generator: no dataset is materialised up front."""
    problem = LWEProblem(official_params(30), data_seed=0)
    stream = problem.iter_batches(num_samples=10 ** 9, batch_size=8)
    assert isinstance(stream, Iterator)
    assert inspect.isgenerator(stream)
    first = next(stream)  # must return immediately despite the huge total
    assert first.num_instances == 8
    stream.close()


def test_batch_is_reproducible_and_order_independent() -> None:
    """Batch i is always the same data, however it was reached."""
    problem = LWEProblem(official_params(30), data_seed=123)
    direct = problem.batch(5, 32)
    replayed = list(problem.iter_batches(num_samples=6 * 32, batch_size=32))[5]
    assert np.array_equal(direct.A, replayed.A)
    assert np.array_equal(direct.b, replayed.b)

    other = LWEProblem(official_params(30), data_seed=123)
    assert np.array_equal(other.batch(5, 32).A, direct.A)


def test_distinct_batches_hold_distinct_data() -> None:
    """Consecutive batches are not accidentally the same draw."""
    problem = LWEProblem(official_params(30), data_seed=0)
    assert not np.array_equal(problem.batch(0, 32).A, problem.batch(1, 32).A)


def test_all_streamed_batches_satisfy_the_lwe_relation() -> None:
    """Streaming does not break the maths."""
    problem = LWEProblem(official_params(50), data_seed=0)
    for batch in problem.iter_batches(200, batch_size=64, labeled=True):
        assert batch.verify_relation()
        assert batch.public.in_range()


def test_batch_arguments_are_validated() -> None:
    """Negative indices and empty batches are rejected."""
    problem = LWEProblem(official_params(12), data_seed=0)
    with pytest.raises(ValueError):
        problem.batch(-1, 8)
    with pytest.raises(ValueError):
        problem.batch(0, 0)
    with pytest.raises(ValueError):
        list(problem.iter_batches(-1, 8))


def test_memory_estimate_is_reported() -> None:
    """A batch's footprint is small and predictable (rule: stream, don't hoard)."""
    problem = LWEProblem(official_params(128), data_seed=0)
    total_bytes, megabytes = problem.memory_estimate(batch_size=128)
    assert total_bytes == 128 * (128 + 1) * 8
    assert megabytes < 1.0


def test_slice_keeps_the_relation_intact() -> None:
    """Slicing a labelled sample keeps A, b and e aligned."""
    sample = generate_lwe_sample(official_params(30), 64, seed=0)
    part = sample.slice(10, 20)
    assert part.num_instances == 10
    assert part.verify_relation()
    assert np.array_equal(part.b, sample.b[10:20])


# --------------------------------------------------------------------------- #
# Config integration
# --------------------------------------------------------------------------- #
def test_params_from_config_match_the_config_file() -> None:
    """LWEParams mirrors the config without reinterpreting it."""
    cfg = load_config(CONFIG_DIR / "base.yaml")
    params = LWEParams.from_config(cfg)
    assert params.n == cfg.lwe.n
    assert params.q == cfg.lwe.q
    assert params.sigma == cfg.lwe.sigma
    assert params.hamming_weight == cfg.lwe.resolved_hamming_weight
    assert params.structure == cfg.lwe.structure
    assert params.max_a_fraction == cfg.lwe.max_a_fraction


def test_all_splits_attack_the_same_secret_with_different_data() -> None:
    """Train / valid / test must share one secret but not share samples."""
    cfg = load_config(CONFIG_DIR / "base.yaml", overrides=["lwe.structure=lwe"])
    train = build_problem(cfg, split="train")
    valid = build_problem(cfg, split="valid")
    test = build_problem(cfg, split="test")

    assert np.array_equal(train.reveal_secret(), valid.reveal_secret())
    assert np.array_equal(train.reveal_secret(), test.reveal_secret())
    assert not np.array_equal(train.batch(0, 32).A, valid.batch(0, 32).A)
    assert not np.array_equal(valid.batch(0, 32).A, test.batch(0, 32).A)


def test_build_problem_dispatches_on_structure() -> None:
    """The factory returns the class matching config.lwe.structure."""
    from salsa.data.rlwe import RLWEProblem

    cfg = load_config(CONFIG_DIR / "base.yaml", overrides=["lwe.structure=lwe"])
    assert type(build_problem(cfg)) is LWEProblem

    cfg_ring = load_config(CONFIG_DIR / "base.yaml")
    assert cfg_ring.lwe.structure == "rlwe"
    assert isinstance(build_problem(cfg_ring), RLWEProblem)


def test_generate_batch_dispatches_on_structure() -> None:
    """generate_batch produces structured rows for an rlwe instance."""
    from salsa.data.rlwe import circulant_matrix

    plain = generate_batch(official_params(12), 12, seed=0)
    ring = generate_batch(official_params(12, structure="rlwe"), 12, seed=0)
    assert plain.verify_relation() and ring.verify_relation()
    assert np.array_equal(ring.A, circulant_matrix(ring.A[0], q=Q))
    assert not np.array_equal(plain.A, circulant_matrix(plain.A[0], q=Q))


def test_config_seed_reproduces_the_whole_instance() -> None:
    """Two problems built from the same config are byte-identical."""
    cfg = load_config(CONFIG_DIR / "base.yaml", overrides=["lwe.structure=lwe"])
    first, second = build_problem(cfg), build_problem(cfg)
    assert np.array_equal(first.reveal_secret(), second.reveal_secret())
    assert np.array_equal(first.batch(3, 16).b, second.batch(3, 16).b)


# --------------------------------------------------------------------------- #
# Alternative q / sigma -- NOT the official experiment configuration
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("q_alt", [1009, 3329])
def test_alternative_modulus_still_satisfies_the_relation(q_alt: int) -> None:
    """Sanity check at other moduli. NOT the official experiment setting."""
    params = LWEParams(n=30, q=q_alt, sigma=SIGMA, hamming_weight=H)
    sample = generate_lwe_sample(params, 128, seed=0)
    assert sample.verify_relation()
    assert sample.b.max() < q_alt and sample.A.max() < q_alt


@pytest.mark.parametrize("sigma_alt", [1.0, 8.0])
def test_alternative_sigma_still_satisfies_the_relation(sigma_alt: float) -> None:
    """Sanity check at other error widths. NOT the official experiment setting."""
    params = LWEParams(n=30, q=Q, sigma=sigma_alt, hamming_weight=H)
    sample = generate_lwe_sample(params, 4000, seed=0)
    assert sample.verify_relation()
    assert abs(float(sample.error.std()) - sigma_alt) < 0.3


def test_ternary_secret_instance_is_supported() -> None:
    """Ternary secrets work end to end. NOT the official experiment setting."""
    params = LWEParams(
        n=30, q=Q, sigma=SIGMA, hamming_weight=H, secret_distribution="ternary"
    )
    sample = generate_lwe_sample(params, 64, seed=0)
    assert sample.verify_relation()
    assert set(np.unique(sample.secret).tolist()).issubset({-1, 0, 1})
    assert sample.public.in_range()


# --------------------------------------------------------------------------- #
# Parameter validation and platform constraints
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "kwargs",
    [
        dict(n=0),
        dict(q=1),
        dict(sigma=-1.0),
        dict(hamming_weight=99),
        dict(structure="ring"),
        dict(secret_distribution="uniform"),
        dict(error_distribution="cauchy"),
        dict(rlwe_variant="toeplitz"),
        dict(tail_cut=0.0),
    ],
)
def test_invalid_params_are_rejected_at_construction(kwargs: dict) -> None:
    """LWEParams validates eagerly, so bad settings never reach generation."""
    settings = dict(n=30, q=Q, sigma=SIGMA, hamming_weight=H)
    settings.update(kwargs)
    with pytest.raises(ValueError):
        LWEParams(**settings)


def test_data_layer_has_no_torch_or_cuda_dependency() -> None:
    """The data layer is pure NumPy: no torch import, no CUDA anywhere."""
    for path in sorted((REPO_ROOT / "salsa" / "data").glob("*.py")):
        source = path.read_text(encoding="utf-8").lower()
        assert "import torch" not in source, f"{path.name} imports torch"
        assert "cuda" not in source, f"{path.name} mentions CUDA"
