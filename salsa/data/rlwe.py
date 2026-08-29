"""RLWE (structured / ring) sample generation.

The SALSA paper generates its data in the Ring-LWE setting and describes the
construction as: *"each a is a line of a circulant matrix generated from an
initial vector in Z_q^n"*.  A single random generator vector therefore produces
``n`` LWE instances whose rows are rotations of one another, which is far more
structure than plain LWE offers -- the paper notes this extra structure may help
the model learn.

Two variants are provided, selected by ``lwe.rlwe_variant`` in the config:

``circulant`` (default, matches the paper's wording)
    Row ``i`` is the generator vector rotated right by ``i``::

        A[i, j] = a[(j - i) mod n]

    Row 0 is the generator vector itself.

``negacyclic``
    The same rotation, but every entry that wrapped around is negated::

        A[i, j] = a[(j - i) mod n]        for j >= i
        A[i, j] = -a[(j - i) mod n] mod q for j <  i

    This sign flip on wrap-around is the structure induced by working in the
    ring ``Z_q[x] / (x^n + 1)``, which is what deployed ring-LWE schemes
    actually use.  It is offered so the effect of the ring choice can itself be
    measured rather than assumed.

Because ``n`` instances come from one generator vector, requesting ``m``
instances draws ``ceil(m / n)`` independent generator vectors and keeps the
first ``m`` rows.  Rows inside one block are algebraically dependent by
construction; this is a property of RLWE, not a defect of the generator.
"""

from typing import Optional

import numpy as np

from .lwe import (
    INT_DTYPE,
    LWEParams,
    LWEProblem,
    LabeledSample,
    _assemble,
    _prepare_secret,
    generate_error,
)
from .secrets import RngLike, resolve_rng

__all__ = [
    "RLWE_VARIANTS",
    "RLWEProblem",
    "circulant_matrix",
    "generate_rlwe_matrix",
    "generate_rlwe_sample",
    "negacyclic_matrix",
    "rotation_block",
]

RLWE_VARIANTS = ("circulant", "negacyclic")


def _rotation_indices(n: int) -> np.ndarray:
    """Return the ``(n, n)`` index matrix ``idx[i, j] = (j - i) mod n``."""
    columns = np.arange(n, dtype=np.int64)[None, :]
    rows = np.arange(n, dtype=np.int64)[:, None]
    return (columns - rows) % n


def circulant_matrix(vector: np.ndarray, q: Optional[int] = None) -> np.ndarray:
    """Build the circulant matrix whose ``i``-th row is ``vector`` rolled by ``i``.

    Args:
        vector: Generator vector of shape ``(n,)``.
        q: Optional modulus; entries are reduced mod ``q`` when given.

    Returns:
        An ``int64`` array of shape ``(n, n)`` with ``A[i, j] = vector[(j-i) mod n]``.
        Row 0 equals ``vector``.

    Raises:
        ValueError: If ``vector`` is not one-dimensional or is empty.

    Example:
        >>> circulant_matrix(np.array([1, 2, 3]))
        array([[1, 2, 3],
               [3, 1, 2],
               [2, 3, 1]])
    """
    values = np.asarray(vector, dtype=INT_DTYPE)
    if values.ndim != 1 or values.size == 0:
        raise ValueError(
            f"vector must be a non-empty 1-D array, got shape {tuple(values.shape)}."
        )
    matrix = values[_rotation_indices(values.size)]
    return matrix % int(q) if q is not None else matrix


def negacyclic_matrix(vector: np.ndarray, q: int) -> np.ndarray:
    """Build the negacyclic (anti-circulant) matrix over ``Z_q``.

    Identical to :func:`circulant_matrix` except that entries which wrapped
    around the end of the generator vector are negated, as they are in
    ``Z_q[x] / (x^n + 1)``.

    Args:
        vector: Generator vector of shape ``(n,)``.
        q: Modulus.  Required, because negation is only meaningful mod ``q``.

    Returns:
        An ``int64`` array of shape ``(n, n)`` with entries in ``[0, q)``.
        Row 0 equals ``vector mod q``.

    Raises:
        ValueError: If ``vector`` is not one-dimensional or is empty.

    Example:
        >>> negacyclic_matrix(np.array([1, 2, 3]), q=251)
        array([[  1,   2,   3],
               [248,   1,   2],
               [249, 248,   1]])
    """
    values = np.asarray(vector, dtype=INT_DTYPE)
    if values.ndim != 1 or values.size == 0:
        raise ValueError(
            f"vector must be a non-empty 1-D array, got shape {tuple(values.shape)}."
        )
    size = values.size
    matrix = values[_rotation_indices(size)]
    columns = np.arange(size, dtype=np.int64)[None, :]
    rows = np.arange(size, dtype=np.int64)[:, None]
    signs = np.where(columns < rows, -1, 1).astype(INT_DTYPE)
    return (matrix * signs) % int(q)


def rotation_block(vector: np.ndarray, q: int, variant: str = "circulant") -> np.ndarray:
    """Build one ``(n, n)`` block of rotations from a generator vector.

    Args:
        vector: Generator vector of shape ``(n,)``.
        q: Modulus.
        variant: ``"circulant"`` or ``"negacyclic"``.

    Returns:
        An ``int64`` array of shape ``(n, n)`` with entries in ``[0, q)``.

    Raises:
        ValueError: On an unknown variant.
    """
    if variant == "circulant":
        return circulant_matrix(vector, q=q)
    if variant == "negacyclic":
        return negacyclic_matrix(vector, q=q)
    raise ValueError(
        f"Unknown RLWE variant '{variant}'; expected one of {list(RLWE_VARIANTS)}."
    )


def generate_rlwe_matrix(
    num_instances: int,
    n: int,
    q: int,
    rng: RngLike = None,
    seed: Optional[int] = None,
    variant: str = "circulant",
    max_a_value: Optional[int] = None,
) -> np.ndarray:
    """Sample the structured public matrix ``A`` for RLWE.

    Draws ``ceil(num_instances / n)`` independent generator vectors, expands
    each into an ``(n, n)`` rotation block, and returns the first
    ``num_instances`` rows.

    Args:
        num_instances: Number of rows ``m`` to return.
        n: Ring degree / lattice dimension.
        q: Modulus.
        rng: Generator or integer seed.
        seed: Integer seed used when ``rng`` is None.
        variant: ``"circulant"`` or ``"negacyclic"``.
        max_a_value: Exclusive upper bound for the *generator vector* entries
            (bounded-``a`` ablation).  ``None`` means the full range ``[0, q)``.

    Returns:
        An ``int64`` array of shape ``(num_instances, n)`` with entries in
        ``[0, q)``.

    Raises:
        ValueError: On invalid shapes, bounds or variant.

    Note:
        Under ``negacyclic``, bounding the generator vector does **not** bound
        the matrix entries: a negated entry ``-a mod q`` lands near ``q``.  The
        bounded-``a`` ablation is therefore only faithful for the ``circulant``
        variant, and this is reported rather than silently ignored.
    """
    if num_instances < 0:
        raise ValueError(f"num_instances must be >= 0, got {num_instances}.")
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}.")
    bound = int(q if max_a_value is None else max_a_value)
    if not 1 <= bound <= int(q):
        raise ValueError(f"max_a_value must be in [1, {int(q)}], got {bound}.")
    if variant not in RLWE_VARIANTS:
        raise ValueError(
            f"Unknown RLWE variant '{variant}'; expected one of {list(RLWE_VARIANTS)}."
        )
    if num_instances == 0:
        return np.zeros((0, int(n)), dtype=INT_DTYPE)

    generator = resolve_rng(rng, seed)
    num_blocks = int(np.ceil(num_instances / float(n)))
    blocks = []
    produced = 0
    for _ in range(num_blocks):
        vector = generator.integers(0, bound, size=int(n), dtype=INT_DTYPE)
        block = rotation_block(vector, q=int(q), variant=variant)
        take = min(int(n), int(num_instances) - produced)
        blocks.append(block[:take])
        produced += take
    return np.concatenate(blocks, axis=0).astype(INT_DTYPE, copy=False)


def generate_rlwe_sample(
    params: LWEParams,
    num_instances: int,
    secret: Optional[np.ndarray] = None,
    rng: RngLike = None,
    seed: Optional[int] = None,
) -> LabeledSample:
    """Generate ``num_instances`` RLWE instances.

    Args:
        params: Cryptographic parameters; ``params.rlwe_variant`` selects the
            ring structure.
        num_instances: Number of instances (rows) to generate.
        secret: Secret to use.  If None, a fresh secret is sampled.
        rng: Generator or integer seed.
        seed: Integer seed used when ``rng`` is None.

    Returns:
        A :class:`~salsa.data.lwe.LabeledSample` satisfying
        ``b = (A s + e) mod q``.

    Example:
        >>> params = LWEParams(n=4, q=251, sigma=3.0, hamming_weight=2,
        ...                    structure="rlwe")
        >>> sample = generate_rlwe_sample(params, num_instances=4, seed=0)
        >>> sample.verify_relation()
        True
    """
    generator = resolve_rng(rng, seed)
    secret_values = _prepare_secret(params, secret, generator)
    matrix = generate_rlwe_matrix(
        num_instances,
        params.n,
        params.q,
        rng=generator,
        variant=params.rlwe_variant,
        max_a_value=params.max_a_value,
    )
    error = generate_error(
        num_instances,
        params.sigma,
        distribution=params.error_distribution,
        rng=generator,
        tail_cut=params.tail_cut,
    )
    return _assemble(params, matrix, secret_values, error)


class RLWEProblem(LWEProblem):
    """A fixed RLWE instance that streams reproducible batches.

    Behaves exactly like :class:`~salsa.data.lwe.LWEProblem` -- same public /
    private separation, same reproducible per-batch seeding -- but draws ``A``
    from rotation blocks instead of i.i.d. uniform rows.

    Example:
        >>> params = LWEParams(n=8, q=251, sigma=3.0, hamming_weight=3,
        ...                    structure="rlwe")
        >>> problem = RLWEProblem(params, data_seed=0)
        >>> problem.batch(0, batch_size=8).num_instances
        8
    """

    def _default_sampler(
        self, num_instances: int, rng: np.random.Generator
    ) -> np.ndarray:
        """Draw ``A`` as rotation blocks of random generator vectors."""
        return generate_rlwe_matrix(
            num_instances,
            self.params.n,
            self.params.q,
            rng=rng,
            variant=self.params.rlwe_variant,
            max_a_value=self.params.max_a_value,
        )
