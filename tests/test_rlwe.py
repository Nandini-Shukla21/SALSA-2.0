"""Tests for RLWE (structured) sample generation.

Two things are checked: the algebraic *structure* of the generated matrix
(rows really are rotations of one generator vector) and the LWE relation
``b == (A s + e) mod q`` on top of it.
"""

from pathlib import Path

import numpy as np
import pytest

from salsa.data import build_problem
from salsa.data.lwe import LWEParams, LWESample
from salsa.data.rlwe import (
    RLWEProblem,
    circulant_matrix,
    generate_rlwe_matrix,
    generate_rlwe_sample,
    negacyclic_matrix,
    rotation_block,
)
from salsa.data.secrets import hamming_weight
from salsa.utils.config import load_config

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"

# Official experiment configuration.
Q = 251
SIGMA = 3.0
H = 3
DIMENSIONS = [4, 12, 30, 50, 128]


def ring_params(n: int, **overrides) -> LWEParams:
    """Return the official RLWE parameter set for dimension ``n``."""
    settings = dict(
        n=n,
        q=Q,
        sigma=SIGMA,
        secret_distribution="binary",
        hamming_weight=H,
        structure="rlwe",
        rlwe_variant="circulant",
    )
    settings.update(overrides)
    return LWEParams(**settings)


# --------------------------------------------------------------------------- #
# Circulant structure
# --------------------------------------------------------------------------- #
def test_circulant_matrix_is_built_from_rotations() -> None:
    """Row i is the generator vector rolled right by i; row 0 is the vector."""
    vector = np.array([1, 2, 3, 4])
    matrix = circulant_matrix(vector)
    expected = np.array(
        [
            [1, 2, 3, 4],
            [4, 1, 2, 3],
            [3, 4, 1, 2],
            [2, 3, 4, 1],
        ]
    )
    assert np.array_equal(matrix, expected)
    assert np.array_equal(matrix[0], vector)
    for i in range(4):
        assert np.array_equal(matrix[i], np.roll(vector, i))


@pytest.mark.parametrize("n", DIMENSIONS)
def test_generated_rlwe_block_is_circulant(n: int) -> None:
    """The first n rows of A are exactly the rotations of A[0]."""
    matrix = generate_rlwe_matrix(n, n, Q, seed=0)
    assert matrix.shape == (n, n)
    assert np.array_equal(matrix, circulant_matrix(matrix[0], q=Q))
    for i in range(n):
        assert np.array_equal(matrix[i], np.roll(matrix[0], i))


@pytest.mark.parametrize("n", [4, 12, 30])
def test_every_row_is_a_rotation_of_every_other_row(n: int) -> None:
    """Within a block, all rows share one multiset obtained by rotation."""
    matrix = generate_rlwe_matrix(n, n, Q, seed=1)
    for i in range(n):
        shift = (i - 2) % n
        assert np.array_equal(matrix[i], np.roll(matrix[2], shift))


def test_multiple_blocks_use_independent_generator_vectors() -> None:
    """m > n draws ceil(m/n) generators; each block is internally circulant."""
    n = 12
    matrix = generate_rlwe_matrix(30, n, Q, seed=0)
    assert matrix.shape == (30, n)

    first, second, tail = matrix[:n], matrix[n : 2 * n], matrix[2 * n :]
    assert np.array_equal(first, circulant_matrix(first[0], q=Q))
    assert np.array_equal(second, circulant_matrix(second[0], q=Q))
    assert not np.array_equal(first, second), "blocks must be independent"
    # The truncated final block is still a prefix of a circulant block.
    assert np.array_equal(tail, circulant_matrix(tail[0], q=Q)[: len(tail)])


def test_row_count_is_exact_for_partial_blocks() -> None:
    """Requesting fewer rows than n returns exactly that many rows."""
    for m in (1, 5, 12, 13, 24, 25):
        assert generate_rlwe_matrix(m, 12, Q, seed=0).shape == (m, 12)
    assert generate_rlwe_matrix(0, 12, Q, seed=0).shape == (0, 12)


# --------------------------------------------------------------------------- #
# Negacyclic variant
# --------------------------------------------------------------------------- #
def test_negacyclic_matrix_negates_wrapped_entries() -> None:
    """Entries that wrap around are negated mod q (the Z_q[x]/(x^n+1) ring)."""
    vector = np.array([1, 2, 3])
    matrix = negacyclic_matrix(vector, q=251)
    expected = np.array(
        [
            [1, 2, 3],
            [251 - 3, 1, 2],
            [251 - 2, 251 - 3, 1],
        ]
    )
    assert np.array_equal(matrix, expected)
    assert np.array_equal(matrix[0], vector)


@pytest.mark.parametrize("n", [4, 12, 30])
def test_negacyclic_structure_of_generated_matrices(n: int) -> None:
    """A generated negacyclic block matches the rebuilt block from its row 0."""
    matrix = generate_rlwe_matrix(n, n, Q, seed=0, variant="negacyclic")
    assert np.array_equal(matrix, negacyclic_matrix(matrix[0], q=Q))
    # Below the diagonal the entries are the negated rotations.
    rebuilt = circulant_matrix(matrix[0], q=Q)
    rows, cols = np.indices((n, n))
    wrapped = cols < rows
    assert np.array_equal(matrix[wrapped], (-rebuilt[wrapped]) % Q)


def test_negacyclic_and_circulant_differ() -> None:
    """The two ring variants are genuinely different data."""
    circ = generate_rlwe_matrix(12, 12, Q, seed=0, variant="circulant")
    nega = generate_rlwe_matrix(12, 12, Q, seed=0, variant="negacyclic")
    assert np.array_equal(circ[0], nega[0])  # row 0 never wraps
    assert not np.array_equal(circ, nega)


def test_unknown_variant_is_rejected() -> None:
    """An unknown ring variant raises instead of silently defaulting."""
    with pytest.raises(ValueError, match="Unknown RLWE variant"):
        generate_rlwe_matrix(4, 4, Q, seed=0, variant="toeplitz")
    with pytest.raises(ValueError, match="Unknown RLWE variant"):
        rotation_block(np.arange(4), q=Q, variant="hankel")


def test_matrix_builders_reject_bad_vectors() -> None:
    """Empty or multi-dimensional generator vectors are errors."""
    for builder in (lambda v: circulant_matrix(v), lambda v: negacyclic_matrix(v, Q)):
        with pytest.raises(ValueError):
            builder(np.array([]))
        with pytest.raises(ValueError):
            builder(np.ones((2, 2)))


# --------------------------------------------------------------------------- #
# The LWE relation under the ring structure
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n", DIMENSIONS)
def test_rlwe_relation_holds_exactly(n: int) -> None:
    """b == (A @ s + e) mod q for structured samples too."""
    sample = generate_rlwe_sample(ring_params(n), num_instances=2 * n, seed=0)
    expected = (sample.A @ sample.secret + sample.error) % Q
    assert np.array_equal(expected, sample.b)
    assert sample.verify_relation()


@pytest.mark.parametrize("n", DIMENSIONS)
def test_rlwe_shapes_ranges_and_secret(n: int) -> None:
    """Shapes, modular range and secret weight are all as configured."""
    m = 2 * n
    sample = generate_rlwe_sample(ring_params(n), num_instances=m, seed=1)
    assert sample.A.shape == (m, n)
    assert sample.b.shape == (m,)
    assert sample.error.shape == (m,)
    assert sample.secret.shape == (n,)
    assert sample.A.min() >= 0 and sample.A.max() < Q
    assert sample.b.min() >= 0 and sample.b.max() < Q
    assert hamming_weight(sample.secret) == H
    assert sample.public.structure == "rlwe"


@pytest.mark.parametrize("n", [4, 12, 30])
def test_negacyclic_samples_satisfy_the_relation(n: int) -> None:
    """The alternative ring variant is mathematically consistent as well."""
    sample = generate_rlwe_sample(
        ring_params(n, rlwe_variant="negacyclic"), num_instances=2 * n, seed=0
    )
    assert sample.verify_relation()
    assert sample.public.in_range()


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n", DIMENSIONS)
def test_same_seed_gives_the_same_rlwe_instance(n: int) -> None:
    """Same seed -> identical structured instance."""
    first = generate_rlwe_sample(ring_params(n), n, seed=11)
    second = generate_rlwe_sample(ring_params(n), n, seed=11)
    assert np.array_equal(first.A, second.A)
    assert np.array_equal(first.b, second.b)
    assert np.array_equal(first.secret, second.secret)


@pytest.mark.parametrize("n", DIMENSIONS)
def test_different_seed_gives_a_different_rlwe_instance(n: int) -> None:
    """Different seed -> different generator vectors."""
    first = generate_rlwe_sample(ring_params(n), n, seed=11)
    second = generate_rlwe_sample(ring_params(n), n, seed=12)
    assert not np.array_equal(first.A, second.A)


# --------------------------------------------------------------------------- #
# Bounded-a under the ring structure
# --------------------------------------------------------------------------- #
def test_bounded_a_limits_circulant_entries() -> None:
    """For the circulant variant the bound carries over to the whole matrix."""
    params = ring_params(30, max_a_fraction=0.4)
    sample = generate_rlwe_sample(params, 300, seed=0)
    assert sample.A.max() < int(0.4 * Q)
    assert sample.verify_relation()


def test_bounded_a_does_not_bound_negacyclic_entries() -> None:
    """Documented caveat: negation moves bounded entries near q.

    This is asserted rather than hidden, so the limitation of the bounded-a
    ablation under the negacyclic ring is part of the test record.
    """
    params = ring_params(30, max_a_fraction=0.4, rlwe_variant="negacyclic")
    sample = generate_rlwe_sample(params, 300, seed=0)
    assert sample.A.max() > int(0.4 * Q)
    assert sample.verify_relation()
    assert sample.public.in_range()


# --------------------------------------------------------------------------- #
# Streaming problem
# --------------------------------------------------------------------------- #
def test_rlwe_problem_streams_structured_batches() -> None:
    """RLWEProblem batches keep the ring structure and the LWE relation."""
    problem = RLWEProblem(ring_params(12), data_seed=0)
    for batch in problem.iter_batches(60, batch_size=12, labeled=True):
        assert batch.verify_relation()
        assert np.array_equal(batch.A, circulant_matrix(batch.A[0], q=Q))


def test_rlwe_problem_batches_are_reproducible() -> None:
    """Batch i of an RLWE stream is stable and independent of history."""
    problem = RLWEProblem(ring_params(30), data_seed=5)
    direct = problem.batch(4, 30)
    replayed = list(problem.iter_batches(5 * 30, batch_size=30))[4]
    assert np.array_equal(direct.A, replayed.A)
    assert np.array_equal(direct.b, replayed.b)


def test_rlwe_problem_returns_public_samples() -> None:
    """The public/private separation is identical for the ring problem."""
    problem = RLWEProblem(ring_params(12), data_seed=0)
    sample = problem.sample(12)
    assert isinstance(sample, LWESample)
    assert not hasattr(sample, "secret")


def test_default_config_builds_a_circulant_rlwe_problem() -> None:
    """The shipped base config is the paper's circulant RLWE setting."""
    cfg = load_config(CONFIG_DIR / "base.yaml")
    assert cfg.lwe.structure == "rlwe"
    assert cfg.lwe.rlwe_variant == "circulant"

    problem = build_problem(cfg, split="train")
    assert isinstance(problem, RLWEProblem)

    batch = problem.batch(0, cfg.lwe.n, labeled=True)
    assert batch.verify_relation()
    assert np.array_equal(batch.A, circulant_matrix(batch.A[0], q=cfg.lwe.q))
    assert hamming_weight(batch.secret) == cfg.lwe.resolved_hamming_weight


def test_config_can_select_the_negacyclic_ring() -> None:
    """The ring variant is a recorded configuration choice, not a constant."""
    cfg = load_config(CONFIG_DIR / "base.yaml", overrides=["lwe.rlwe_variant=negacyclic"])
    cfg.validate()
    problem = build_problem(cfg)
    batch = problem.batch(0, cfg.lwe.n, labeled=True)
    assert batch.verify_relation()
    assert np.array_equal(batch.A, negacyclic_matrix(batch.A[0], q=cfg.lwe.q))
