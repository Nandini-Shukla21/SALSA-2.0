"""Secret-key sampling for LWE instances.

The SALSA setting attacks *sparse* secrets: a vector ``s`` of length ``n`` whose
non-zero entries number exactly ``h`` (the Hamming weight).  Sparse binary and
ternary secrets are used in real homomorphic-encryption libraries for
efficiency, which is why they are the interesting attack target.

Two properties matter for the experiments and are enforced here:

* **Exactness.**  ``h`` is an exact count, not an expectation.  Sampling each
  coordinate independently with probability ``h/n`` would give a *random*
  Hamming weight and would silently change the difficulty of the problem from
  one run to the next.
* **Reproducibility.**  Given a seed, the secret is always the same, and it is
  derived from the experiment's master seed by a stable labelled derivation so
  that the training/validation/test splits all attack the *same* secret.

This module deals only with generating secrets for controlled experiments.
Attack code never calls it; see :mod:`salsa.data.lwe` for the public/private
data separation.
"""

from typing import Optional, Union

import numpy as np

__all__ = [
    "BINARY",
    "TERNARY",
    "SECRET_DISTRIBUTIONS",
    "generate_binary_secret",
    "generate_secret",
    "generate_ternary_secret",
    "hamming_weight",
    "is_valid_secret",
    "resolve_rng",
    "secret_from_config",
]

BINARY = "binary"
TERNARY = "ternary"
SECRET_DISTRIBUTIONS = (BINARY, TERNARY)

#: dtype used for every integer array in the data layer.
SECRET_DTYPE = np.int64

RngLike = Union[np.random.Generator, int, None]


def resolve_rng(rng: RngLike = None, seed: Optional[int] = None) -> np.random.Generator:
    """Return a NumPy :class:`~numpy.random.Generator` from flexible input.

    Args:
        rng: An existing ``Generator``, an integer seed, or None.
        seed: An integer seed, used when ``rng`` is None.

    Returns:
        A ``Generator``.  If neither ``rng`` nor ``seed`` is given, a
        non-deterministic generator seeded from OS entropy is returned; callers
        that need reproducibility must supply one of the two.

    Raises:
        TypeError: If ``rng`` is neither a Generator, an int nor None.
    """
    if isinstance(rng, np.random.Generator):
        return rng
    if isinstance(rng, (int, np.integer)):
        return np.random.default_rng(int(rng))
    if rng is not None:
        raise TypeError(
            f"rng must be a numpy Generator, an int seed or None, got {type(rng).__name__}."
        )
    if seed is not None:
        return np.random.default_rng(int(seed))
    return np.random.default_rng()


def _validate_secret_params(n: int, weight: int) -> None:
    """Validate the dimension / Hamming-weight pair."""
    if not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError(f"n must be a positive integer, got {n!r}.")
    if not isinstance(weight, (int, np.integer)) or weight < 0:
        raise ValueError(f"hamming_weight must be a non-negative integer, got {weight!r}.")
    if weight > n:
        raise ValueError(
            f"hamming_weight ({weight}) cannot exceed the dimension n ({n})."
        )


def generate_binary_secret(
    n: int,
    hamming_weight: int,
    rng: RngLike = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Sample a binary secret in ``{0, 1}^n`` with *exactly* ``h`` ones.

    The support is drawn uniformly at random from the ``C(n, h)`` possible
    supports (a uniform random subset of coordinates, without replacement).

    Args:
        n: Secret length / lattice dimension.
        hamming_weight: Exact number of ones.
        rng: Generator or integer seed.
        seed: Integer seed used when ``rng`` is None.

    Returns:
        An ``int64`` array of shape ``(n,)`` containing exactly
        ``hamming_weight`` ones.

    Raises:
        ValueError: If ``n < 1`` or ``h`` is outside ``[0, n]``.

    Example:
        >>> s = generate_binary_secret(10, 3, seed=0)
        >>> int(s.sum())
        3
    """
    _validate_secret_params(n, hamming_weight)
    generator = resolve_rng(rng, seed)
    secret = np.zeros(int(n), dtype=SECRET_DTYPE)
    if hamming_weight:
        support = generator.choice(int(n), size=int(hamming_weight), replace=False)
        secret[support] = 1
    return secret


def generate_ternary_secret(
    n: int,
    hamming_weight: int,
    rng: RngLike = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Sample a ternary secret in ``{-1, 0, 1}^n`` with exactly ``h`` non-zeros.

    The support is uniform over all size-``h`` subsets and each non-zero entry
    is independently ``+1`` or ``-1`` with probability 1/2.

    Args:
        n: Secret length / lattice dimension.
        hamming_weight: Exact number of non-zero entries.
        rng: Generator or integer seed.
        seed: Integer seed used when ``rng`` is None.

    Returns:
        An ``int64`` array of shape ``(n,)`` with exactly ``hamming_weight``
        non-zero entries, each in ``{-1, +1}``.

    Raises:
        ValueError: If ``n < 1`` or ``h`` is outside ``[0, n]``.
    """
    _validate_secret_params(n, hamming_weight)
    generator = resolve_rng(rng, seed)
    secret = np.zeros(int(n), dtype=SECRET_DTYPE)
    if hamming_weight:
        support = generator.choice(int(n), size=int(hamming_weight), replace=False)
        signs = generator.integers(0, 2, size=int(hamming_weight)) * 2 - 1
        secret[support] = signs.astype(SECRET_DTYPE)
    return secret


def generate_secret(
    n: int,
    hamming_weight: int,
    distribution: str = BINARY,
    rng: RngLike = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Sample a secret from the configured distribution.

    Args:
        n: Secret length / lattice dimension.
        hamming_weight: Exact number of non-zero entries.
        distribution: ``"binary"`` or ``"ternary"``.
        rng: Generator or integer seed.
        seed: Integer seed used when ``rng`` is None.

    Returns:
        An ``int64`` array of shape ``(n,)``.

    Raises:
        ValueError: On an unknown distribution or invalid ``n`` / ``h``.
    """
    if distribution == BINARY:
        return generate_binary_secret(n, hamming_weight, rng=rng, seed=seed)
    if distribution == TERNARY:
        return generate_ternary_secret(n, hamming_weight, rng=rng, seed=seed)
    raise ValueError(
        f"Unknown secret distribution '{distribution}'; expected one of "
        f"{list(SECRET_DISTRIBUTIONS)}."
    )


def hamming_weight(secret: np.ndarray) -> int:
    """Return the number of non-zero entries of ``secret``."""
    return int(np.count_nonzero(np.asarray(secret)))


def is_valid_secret(
    secret: np.ndarray,
    distribution: str = BINARY,
    expected_weight: Optional[int] = None,
) -> bool:
    """Check that ``secret`` is well formed for the given distribution.

    Args:
        secret: Candidate secret vector.
        distribution: ``"binary"`` or ``"ternary"``.
        expected_weight: If given, also require this exact Hamming weight.

    Returns:
        True when the secret's alphabet (and optionally its weight) is correct.
    """
    values = np.asarray(secret)
    if values.ndim != 1:
        return False
    allowed = {0, 1} if distribution == BINARY else {-1, 0, 1}
    if not set(np.unique(values).tolist()).issubset(allowed):
        return False
    if expected_weight is not None and hamming_weight(values) != int(expected_weight):
        return False
    return True


def secret_from_config(config, secret_index: Optional[int] = None) -> np.ndarray:
    """Derive the experiment's secret from its configuration.

    The seed is derived from the master seed with the stable label
    ``("secret", index)``, so the secret is identical across the train,
    validation and test splits and across processes -- attacking a different
    secret in validation than in training would invalidate the whole
    experiment.

    Args:
        config: A :class:`~salsa.utils.config.Config`.
        secret_index: Which secret to build (defaults to
            ``config.lwe.secret_index``).

    Returns:
        An ``int64`` secret array of shape ``(config.lwe.n,)``.
    """
    from ..training.seed import derive_seed

    lwe = config.lwe
    index = lwe.secret_index if secret_index is None else int(secret_index)
    seed = derive_seed(config.experiment.seed, "secret", index)
    return generate_secret(
        n=lwe.n,
        hamming_weight=lwe.resolved_hamming_weight,
        distribution=lwe.secret_distribution,
        seed=seed,
    )
