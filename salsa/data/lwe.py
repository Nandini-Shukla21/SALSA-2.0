"""LWE sample generation.

The Learning-With-Errors problem, following Regev (2005) and the SALSA paper:

    A  in  Z_q^{m x n}     uniformly random public matrix
    s  in  Z_q^n           the secret (sparse binary here)
    e  in  Z^m             error, drawn from a narrow discrete Gaussian
    b = A . s + e  (mod q) in Z_q^m

One *instance* is one row of ``A`` together with the matching entry of ``b``.
An attacker sees ``(A, b)`` and must find ``s``.

Public / private separation
---------------------------
This module deliberately splits what an attacker may see from what only a
controlled experiment may see:

* :class:`LWESample` -- ``A``, ``b``, ``n``, ``q``.  **Public.**  This is the
  only type that may ever be handed to :mod:`salsa.recovery`.
* :class:`GroundTruth` -- ``secret``, ``error``.  Private.
* :class:`LabeledSample` -- a pair of the two, used for generation, evaluation
  and for scoring an attack *after* it has finished.

Recovery code in later phases is typed against :class:`LWESample`, so the
secret cannot leak into an attack by accident.

Memory
------
Datasets are never materialised in full.  :meth:`LWEProblem.iter_batches`
streams fixed-size batches, and each batch is reproducible on its own from a
derived per-batch seed, so a training run can be resumed mid-stream.
"""

from dataclasses import dataclass
from typing import Callable, Iterator, Optional, Tuple

import numpy as np

from .secrets import (
    BINARY,
    SECRET_DISTRIBUTIONS,
    RngLike,
    generate_secret,
    hamming_weight,
    is_valid_secret,
    resolve_rng,
)

__all__ = [
    "ERROR_DISTRIBUTIONS",
    "GroundTruth",
    "LWEParams",
    "LWEProblem",
    "LWESample",
    "LabeledSample",
    "MatrixSampler",
    "generate_batch",
    "generate_error",
    "generate_lwe_sample",
    "generate_uniform_matrix",
]

#: Integer dtype used across the whole data layer.
INT_DTYPE = np.int64

#: Supported error distributions.
ERROR_DISTRIBUTIONS = ("discrete_gaussian", "rounded_gaussian", "none")

#: Tail truncation for the discrete Gaussian, in multiples of sigma.
#: At 10 sigma the discarded mass is < 1e-22, i.e. far below sampling noise.
DEFAULT_TAIL_CUT = 10.0

#: A matrix sampler maps (num_instances, rng) to an (m, n) array over Z_q.
MatrixSampler = Callable[[int, np.random.Generator], np.ndarray]


# --------------------------------------------------------------------------- #
# Parameters
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LWEParams:
    """The cryptographic instance parameters.

    Attributes:
        n: Lattice dimension / secret length.
        q: Modulus.
        sigma: Standard deviation of the error distribution.
        secret_distribution: ``"binary"`` or ``"ternary"``.
        hamming_weight: Exact number of non-zero secret coordinates.
        structure: ``"lwe"`` (uniform rows) or ``"rlwe"`` (structured rows).
        error_distribution: ``"discrete_gaussian"``, ``"rounded_gaussian"`` or
            ``"none"``.
        max_a_fraction: Coefficients of ``a`` are drawn from
            ``[0, floor(max_a_fraction * q))``.  ``1.0`` (the default) means the
            full range ``[0, q)``; smaller values reproduce the bounded-``a``
            ablation of the SALSA paper.
        rlwe_variant: ``"circulant"`` or ``"negacyclic"``; ignored for
            ``structure="lwe"``.
        tail_cut: Discrete-Gaussian truncation in multiples of sigma.
    """

    n: int
    q: int
    sigma: float
    secret_distribution: str = BINARY
    hamming_weight: int = 3
    structure: str = "lwe"
    error_distribution: str = "discrete_gaussian"
    max_a_fraction: float = 1.0
    rlwe_variant: str = "circulant"
    tail_cut: float = DEFAULT_TAIL_CUT

    def __post_init__(self) -> None:
        """Validate the instance eagerly so bad parameters fail at construction."""
        self.validate()

    def validate(self) -> None:
        """Raise :class:`ValueError` if any parameter is invalid."""
        if self.n < 1:
            raise ValueError(f"n must be >= 1, got {self.n}.")
        if self.q < 2:
            raise ValueError(f"q must be >= 2, got {self.q}.")
        if self.sigma < 0:
            raise ValueError(f"sigma must be >= 0, got {self.sigma}.")
        if self.secret_distribution not in SECRET_DISTRIBUTIONS:
            raise ValueError(
                f"Unknown secret distribution '{self.secret_distribution}'; "
                f"expected one of {list(SECRET_DISTRIBUTIONS)}."
            )
        if not 0 <= self.hamming_weight <= self.n:
            raise ValueError(
                f"hamming_weight must satisfy 0 <= h <= n, got {self.hamming_weight} "
                f"with n={self.n}."
            )
        if self.structure not in ("lwe", "rlwe"):
            raise ValueError(
                f"structure must be 'lwe' or 'rlwe', got '{self.structure}'."
            )
        if self.error_distribution not in ERROR_DISTRIBUTIONS:
            raise ValueError(
                f"Unknown error distribution '{self.error_distribution}'; "
                f"expected one of {list(ERROR_DISTRIBUTIONS)}."
            )
        if self.rlwe_variant not in ("circulant", "negacyclic"):
            raise ValueError(
                f"rlwe_variant must be 'circulant' or 'negacyclic', "
                f"got '{self.rlwe_variant}'."
            )
        if not 0.0 < self.max_a_fraction <= 1.0:
            raise ValueError(
                f"max_a_fraction must be in (0, 1], got {self.max_a_fraction}."
            )
        if self.max_a_value < 1:
            raise ValueError(
                f"max_a_fraction={self.max_a_fraction} with q={self.q} leaves no "
                "values to sample from."
            )
        if self.tail_cut <= 0:
            raise ValueError(f"tail_cut must be > 0, got {self.tail_cut}.")

    @property
    def max_a_value(self) -> int:
        """Exclusive upper bound for the coefficients of ``a``."""
        if self.max_a_fraction >= 1.0:
            return int(self.q)
        return int(self.max_a_fraction * self.q)

    @property
    def bounded_a(self) -> bool:
        """True when the bounded-``a`` ablation is active."""
        return self.max_a_value < int(self.q)

    @property
    def density(self) -> float:
        """Secret density ``h / n``."""
        return self.hamming_weight / float(self.n)

    def bytes_per_instance(self) -> int:
        """Approximate memory cost of one instance (one row of A plus one b)."""
        return int(np.dtype(INT_DTYPE).itemsize) * (self.n + 1)

    @classmethod
    def from_config(cls, config) -> "LWEParams":
        """Build parameters from a :class:`~salsa.utils.config.Config`."""
        lwe = config.lwe
        return cls(
            n=int(lwe.n),
            q=int(lwe.q),
            sigma=float(lwe.sigma),
            secret_distribution=str(lwe.secret_distribution),
            hamming_weight=int(lwe.resolved_hamming_weight),
            structure=str(lwe.structure),
            error_distribution=str(lwe.error_distribution),
            max_a_fraction=float(lwe.max_a_fraction),
            rlwe_variant=str(lwe.rlwe_variant),
        )


# --------------------------------------------------------------------------- #
# Sample containers
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LWESample:
    """Public observations of an LWE instance: what an attacker sees.

    This type intentionally carries **no** secret and **no** error.  Every
    secret-recovery entry point in :mod:`salsa.recovery` accepts this type, so
    ground truth cannot reach an attack through the data structures.

    Attributes:
        A: ``(m, n)`` integer matrix over ``Z_q``.
        b: ``(m,)`` integer vector over ``Z_q``.
        q: The modulus (public).
        structure: ``"lwe"`` or ``"rlwe"`` (public knowledge of the scheme).
    """

    A: np.ndarray
    b: np.ndarray
    q: int
    structure: str = "lwe"

    @property
    def num_instances(self) -> int:
        """Number of LWE instances (rows) in this sample."""
        return int(self.A.shape[0])

    @property
    def n(self) -> int:
        """Lattice dimension."""
        return int(self.A.shape[1])

    def __len__(self) -> int:
        """Number of instances, so ``len(sample)`` works."""
        return self.num_instances

    def in_range(self) -> bool:
        """True when every published value lies in ``[0, q)``."""
        return bool(
            self.A.min() >= 0
            and self.A.max() < self.q
            and self.b.min() >= 0
            and self.b.max() < self.q
        )

    def slice(self, start: int, stop: int) -> "LWESample":
        """Return a view of instances ``[start, stop)``."""
        return LWESample(
            A=self.A[start:stop], b=self.b[start:stop], q=self.q, structure=self.structure
        )


@dataclass(frozen=True)
class GroundTruth:
    """Private data of a generated instance: never shown to an attack.

    Attributes:
        secret: ``(n,)`` secret vector.
        error: ``(m,)`` signed error vector (centred, *not* reduced mod q).
    """

    secret: np.ndarray
    error: np.ndarray

    @property
    def hamming_weight(self) -> int:
        """Number of non-zero secret coordinates."""
        return hamming_weight(self.secret)


@dataclass(frozen=True)
class LabeledSample:
    """A generated sample together with its ground truth.

    Used for data generation, for model training targets and for *scoring* an
    attack after it has produced a candidate.  Recovery algorithms must take
    :attr:`public` instead of this object.

    Attributes:
        public: The attacker-visible :class:`LWESample`.
        ground_truth: The private :class:`GroundTruth`.
    """

    public: LWESample
    ground_truth: GroundTruth

    # Convenience accessors for the public part -------------------------------
    @property
    def A(self) -> np.ndarray:
        """Public matrix ``A``."""
        return self.public.A

    @property
    def b(self) -> np.ndarray:
        """Public vector ``b``."""
        return self.public.b

    @property
    def q(self) -> int:
        """Modulus."""
        return self.public.q

    @property
    def n(self) -> int:
        """Lattice dimension."""
        return self.public.n

    @property
    def num_instances(self) -> int:
        """Number of instances."""
        return self.public.num_instances

    def __len__(self) -> int:
        """Number of instances."""
        return self.public.num_instances

    # Private accessors -------------------------------------------------------
    @property
    def secret(self) -> np.ndarray:
        """The true secret.  Controlled experiments only."""
        return self.ground_truth.secret

    @property
    def error(self) -> np.ndarray:
        """The true error vector.  Controlled experiments only."""
        return self.ground_truth.error

    def as_public(self) -> LWESample:
        """Return only the attacker-visible part of this sample."""
        return self.public

    def verify_relation(self) -> bool:
        """Check that ``b == (A . s + e) mod q`` holds for every instance.

        This is a *generator self-consistency* check used by the data tests.
        It is not the attack-side residual verification of phase 10, which may
        never use the true secret.

        Returns:
            True when the LWE relation holds exactly for all instances.
        """
        expected = (self.A @ self.secret + self.error) % self.q
        return bool(np.array_equal(expected.astype(INT_DTYPE), self.b.astype(INT_DTYPE)))

    def slice(self, start: int, stop: int) -> "LabeledSample":
        """Return a view of instances ``[start, stop)``."""
        return LabeledSample(
            public=self.public.slice(start, stop),
            ground_truth=GroundTruth(
                secret=self.ground_truth.secret,
                error=self.ground_truth.error[start:stop],
            ),
        )


# --------------------------------------------------------------------------- #
# Error and matrix sampling
# --------------------------------------------------------------------------- #
def generate_error(
    size: int,
    sigma: float,
    distribution: str = "discrete_gaussian",
    rng: RngLike = None,
    seed: Optional[int] = None,
    tail_cut: float = DEFAULT_TAIL_CUT,
) -> np.ndarray:
    """Sample the LWE error vector.

    Args:
        size: Number of error terms to draw.
        sigma: Standard deviation.  ``0`` yields an all-zero error vector.
        distribution: One of

            * ``"discrete_gaussian"`` (default) -- exact discrete Gaussian on
              the integers, ``P(x) proportional to exp(-x^2 / (2 sigma^2))``,
              truncated at ``tail_cut * sigma``.  This is the distribution named
              in the SALSA paper.
            * ``"rounded_gaussian"`` -- ``round(N(0, sigma))``.  Provided for
              comparison; it is a slightly different distribution.
            * ``"none"`` -- no error (an error-free ablation, not an LWE
              instance in the cryptographic sense).
        rng: Generator or integer seed.
        seed: Integer seed used when ``rng`` is None.
        tail_cut: Truncation point in multiples of sigma.

    Returns:
        A signed ``int64`` array of shape ``(size,)``, centred on zero and
        **not** reduced modulo q.

    Raises:
        ValueError: On a negative ``sigma``, negative ``size`` or unknown
            distribution.
    """
    if size < 0:
        raise ValueError(f"size must be >= 0, got {size}.")
    if sigma < 0:
        raise ValueError(f"sigma must be >= 0, got {sigma}.")
    if distribution not in ERROR_DISTRIBUTIONS:
        raise ValueError(
            f"Unknown error distribution '{distribution}'; expected one of "
            f"{list(ERROR_DISTRIBUTIONS)}."
        )

    if distribution == "none" or sigma == 0:
        return np.zeros(int(size), dtype=INT_DTYPE)

    generator = resolve_rng(rng, seed)

    if distribution == "rounded_gaussian":
        return np.rint(generator.normal(0.0, sigma, size=int(size))).astype(INT_DTYPE)

    # Exact (truncated) discrete Gaussian via its normalised pmf.
    bound = int(np.ceil(tail_cut * sigma))
    support = np.arange(-bound, bound + 1, dtype=INT_DTYPE)
    weights = np.exp(-(support.astype(np.float64) ** 2) / (2.0 * sigma * sigma))
    probabilities = weights / weights.sum()
    return generator.choice(support, size=int(size), p=probabilities).astype(INT_DTYPE)


def generate_uniform_matrix(
    num_instances: int,
    n: int,
    q: int,
    rng: RngLike = None,
    seed: Optional[int] = None,
    max_a_value: Optional[int] = None,
) -> np.ndarray:
    """Sample the public matrix ``A`` with i.i.d. uniform entries.

    Args:
        num_instances: Number of rows ``m``.
        n: Number of columns.
        q: Modulus.
        rng: Generator or integer seed.
        seed: Integer seed used when ``rng`` is None.
        max_a_value: Exclusive upper bound for the entries.  ``None`` means the
            full range ``[0, q)``.  A smaller value activates the bounded-``a``
            ablation from the SALSA paper.

    Returns:
        An ``int64`` array of shape ``(num_instances, n)`` with entries in
        ``[0, max_a_value)``.

    Raises:
        ValueError: On invalid shapes or a bound outside ``[1, q]``.
    """
    if num_instances < 0:
        raise ValueError(f"num_instances must be >= 0, got {num_instances}.")
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}.")
    bound = int(q if max_a_value is None else max_a_value)
    if not 1 <= bound <= int(q):
        raise ValueError(f"max_a_value must be in [1, q]={[1, int(q)]}, got {bound}.")
    generator = resolve_rng(rng, seed)
    return generator.integers(0, bound, size=(int(num_instances), int(n)), dtype=INT_DTYPE)


# --------------------------------------------------------------------------- #
# Sample generation
# --------------------------------------------------------------------------- #
def _prepare_secret(params: LWEParams, secret: Optional[np.ndarray], rng) -> np.ndarray:
    """Return a validated secret, sampling one if none was supplied."""
    if secret is None:
        return generate_secret(
            n=params.n,
            hamming_weight=params.hamming_weight,
            distribution=params.secret_distribution,
            rng=rng,
        )
    values = np.asarray(secret, dtype=INT_DTYPE)
    if values.shape != (params.n,):
        raise ValueError(
            f"secret must have shape ({params.n},), got {tuple(values.shape)}."
        )
    if not is_valid_secret(values, params.secret_distribution):
        raise ValueError(
            f"secret contains values outside the '{params.secret_distribution}' alphabet."
        )
    return values


def _assemble(
    params: LWEParams,
    matrix: np.ndarray,
    secret: np.ndarray,
    error: np.ndarray,
) -> LabeledSample:
    """Combine ``A``, ``s`` and ``e`` into a :class:`LabeledSample`."""
    b = (matrix @ secret + error) % params.q
    return LabeledSample(
        public=LWESample(
            A=matrix.astype(INT_DTYPE, copy=False),
            b=b.astype(INT_DTYPE, copy=False),
            q=int(params.q),
            structure=params.structure,
        ),
        ground_truth=GroundTruth(
            secret=secret.astype(INT_DTYPE, copy=False),
            error=error.astype(INT_DTYPE, copy=False),
        ),
    )


def generate_lwe_sample(
    params: LWEParams,
    num_instances: int,
    secret: Optional[np.ndarray] = None,
    rng: RngLike = None,
    seed: Optional[int] = None,
) -> LabeledSample:
    """Generate ``num_instances`` plain (unstructured) LWE instances.

    Args:
        params: The cryptographic instance parameters.
        num_instances: Number of rows ``m`` to generate.
        secret: The secret to use.  If None, a fresh secret is sampled -- which
            is almost never what an experiment wants, since all splits must
            share one secret.
        rng: Generator or integer seed.
        seed: Integer seed used when ``rng`` is None.

    Returns:
        A :class:`LabeledSample` satisfying ``b = (A s + e) mod q``.

    Example:
        >>> params = LWEParams(n=4, q=251, sigma=3.0, hamming_weight=2)
        >>> sample = generate_lwe_sample(params, num_instances=8, seed=0)
        >>> sample.verify_relation()
        True
    """
    generator = resolve_rng(rng, seed)
    secret_values = _prepare_secret(params, secret, generator)
    matrix = generate_uniform_matrix(
        num_instances,
        params.n,
        params.q,
        rng=generator,
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


def generate_batch(
    params: LWEParams,
    num_instances: int,
    secret: Optional[np.ndarray] = None,
    rng: RngLike = None,
    seed: Optional[int] = None,
) -> LabeledSample:
    """Generate one batch, dispatching on ``params.structure``.

    This is the structure-agnostic entry point: it produces plain LWE rows for
    ``structure="lwe"`` and structured (circulant / negacyclic) rows for
    ``structure="rlwe"``.

    Args:
        params: The cryptographic instance parameters.
        num_instances: Number of instances (rows) to generate.
        secret: Secret to use; sampled if None.
        rng: Generator or integer seed.
        seed: Integer seed used when ``rng`` is None.

    Returns:
        A :class:`LabeledSample`.
    """
    if params.structure == "rlwe":
        from .rlwe import generate_rlwe_sample  # local import avoids a cycle

        return generate_rlwe_sample(
            params, num_instances=num_instances, secret=secret, rng=rng, seed=seed
        )
    return generate_lwe_sample(
        params, num_instances=num_instances, secret=secret, rng=rng, seed=seed
    )


# --------------------------------------------------------------------------- #
# Streaming problem object
# --------------------------------------------------------------------------- #
class LWEProblem:
    """A fixed LWE instance that streams reproducible batches of samples.

    The problem owns the secret.  Batches are generated on demand, so a
    training run never holds more than one batch of data in memory.  Each batch
    is seeded from ``derive_seed(data_seed, "batch", index)``, which makes an
    individual batch reproducible on its own and lets a run resume mid-stream.

    Example:
        >>> params = LWEParams(n=8, q=251, sigma=3.0, hamming_weight=3)
        >>> problem = LWEProblem(params, secret=None, data_seed=0)
        >>> batch = problem.batch(0, batch_size=16)      # public view
        >>> batch.num_instances
        16
    """

    def __init__(
        self,
        params: LWEParams,
        secret: Optional[np.ndarray] = None,
        data_seed: int = 0,
        secret_seed: Optional[int] = None,
        matrix_sampler: Optional[MatrixSampler] = None,
    ) -> None:
        """Create a problem instance.

        Args:
            params: Cryptographic parameters.
            secret: An explicit secret.  If None, one is sampled from
                ``secret_seed`` (or ``data_seed`` when that is also None).
            data_seed: Master seed for the sample stream.
            secret_seed: Separate seed for the secret, so that changing the data
                stream never changes which secret is attacked.
            matrix_sampler: Callable ``(m, rng) -> (m, n) array`` used to draw
                ``A``.  Defaults to i.i.d. uniform sampling.
        """
        self.params = params
        self.data_seed = int(data_seed)
        self._secret_seed = int(data_seed if secret_seed is None else secret_seed)
        self._secret = _prepare_secret(
            params, secret, resolve_rng(seed=self._secret_seed)
        )
        self._matrix_sampler: MatrixSampler = matrix_sampler or self._default_sampler
        self._rng = resolve_rng(seed=self.data_seed)

    # -- construction ------------------------------------------------------- #
    @classmethod
    def from_config(
        cls,
        config,
        split: str = "train",
        secret: Optional[np.ndarray] = None,
    ) -> "LWEProblem":
        """Build a problem from a :class:`~salsa.utils.config.Config`.

        The secret is derived from ``("secret", secret_index)`` and the data
        stream from ``("data", split)``, so all splits attack the same secret
        while drawing disjoint, reproducible sample streams.

        Args:
            config: The experiment configuration.
            split: ``"train"``, ``"valid"``, ``"test"`` or any label.
            secret: Optional explicit secret (overrides derivation).

        Returns:
            An :class:`LWEProblem` (or :class:`~salsa.data.rlwe.RLWEProblem`
            when ``lwe.structure`` is ``"rlwe"``; use
            :func:`salsa.data.build_problem` for automatic dispatch).
        """
        from ..training.seed import derive_seed

        params = LWEParams.from_config(config)
        return cls(
            params=params,
            secret=secret,
            data_seed=derive_seed(config.experiment.seed, "data", split),
            secret_seed=derive_seed(
                config.experiment.seed, "secret", config.lwe.secret_index
            ),
        )

    # -- accessors ---------------------------------------------------------- #
    @property
    def n(self) -> int:
        """Lattice dimension."""
        return self.params.n

    @property
    def q(self) -> int:
        """Modulus."""
        return self.params.q

    def reveal_secret(self) -> np.ndarray:
        """Return a copy of the true secret.

        **Controlled experiments only.**  Nothing in :mod:`salsa.recovery` may
        call this; it exists so that evaluation code can score an attack after
        the attack has finished.
        """
        return self._secret.copy()

    def _default_sampler(self, num_instances: int, rng: np.random.Generator) -> np.ndarray:
        """Draw ``A`` with i.i.d. uniform entries (plain LWE)."""
        return generate_uniform_matrix(
            num_instances,
            self.params.n,
            self.params.q,
            rng=rng,
            max_a_value=self.params.max_a_value,
        )

    # -- sampling ----------------------------------------------------------- #
    def _generate(self, num_instances: int, rng: np.random.Generator) -> LabeledSample:
        """Generate one labelled batch with the given generator."""
        matrix = self._matrix_sampler(num_instances, rng)
        error = generate_error(
            num_instances,
            self.params.sigma,
            distribution=self.params.error_distribution,
            rng=rng,
            tail_cut=self.params.tail_cut,
        )
        return _assemble(self.params, matrix, self._secret, error)

    def labeled_sample(
        self,
        num_instances: int,
        rng: RngLike = None,
        seed: Optional[int] = None,
    ) -> LabeledSample:
        """Draw a labelled sample (public data **and** ground truth).

        Args:
            num_instances: Number of instances to draw.
            rng: Optional generator; defaults to the problem's own stream.
            seed: Optional seed for an independent, reproducible draw.

        Returns:
            A :class:`LabeledSample`.
        """
        generator = self._rng if (rng is None and seed is None) else resolve_rng(rng, seed)
        return self._generate(int(num_instances), generator)

    def sample(
        self,
        num_instances: int,
        rng: RngLike = None,
        seed: Optional[int] = None,
    ) -> LWESample:
        """Draw a **public** sample: ``A`` and ``b`` only.

        This is the method attack code should use.

        Args:
            num_instances: Number of instances to draw.
            rng: Optional generator; defaults to the problem's own stream.
            seed: Optional seed for an independent, reproducible draw.

        Returns:
            An :class:`LWESample` with no ground truth attached.
        """
        return self.labeled_sample(num_instances, rng=rng, seed=seed).as_public()

    def batch(
        self, index: int, batch_size: int, labeled: bool = False
    ) -> "LWESample | LabeledSample":
        """Return batch number ``index`` of the reproducible stream.

        Batch ``index`` always contains the same data for a given problem,
        independently of how many batches were drawn before it.  That makes a
        training stream resumable and each reported batch traceable.

        Args:
            index: Zero-based batch index.
            batch_size: Number of instances in the batch.
            labeled: Return a :class:`LabeledSample` instead of the public view.

        Returns:
            An :class:`LWESample`, or a :class:`LabeledSample` if ``labeled``.

        Raises:
            ValueError: If ``index`` is negative or ``batch_size < 1``.
        """
        from ..training.seed import derive_seed

        if index < 0:
            raise ValueError(f"batch index must be >= 0, got {index}.")
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}.")
        generator = resolve_rng(seed=derive_seed(self.data_seed, "batch", int(index)))
        sample = self._generate(int(batch_size), generator)
        return sample if labeled else sample.as_public()

    def iter_batches(
        self,
        num_samples: int,
        batch_size: int,
        labeled: bool = False,
        start_batch: int = 0,
    ) -> Iterator["LWESample | LabeledSample"]:
        """Stream ``num_samples`` instances in batches of ``batch_size``.

        The dataset is never materialised: each batch is created, yielded and
        then free to be garbage-collected.  The final batch is truncated so the
        stream contains exactly ``num_samples`` instances.

        Args:
            num_samples: Total number of instances to yield.
            batch_size: Instances per batch.
            labeled: Yield :class:`LabeledSample` objects instead of public ones.
            start_batch: First batch index (used when resuming).

        Yields:
            One sample object per batch.

        Raises:
            ValueError: If ``num_samples < 0`` or ``batch_size < 1``.
        """
        if num_samples < 0:
            raise ValueError(f"num_samples must be >= 0, got {num_samples}.")
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}.")

        remaining = int(num_samples)
        index = int(start_batch)
        while remaining > 0:
            size = min(int(batch_size), remaining)
            yield self.batch(index, size, labeled=labeled)
            remaining -= size
            index += 1

    def memory_estimate(self, batch_size: int) -> Tuple[int, float]:
        """Estimate the memory footprint of one batch.

        Args:
            batch_size: Instances per batch.

        Returns:
            ``(bytes, megabytes)`` for the ``A`` and ``b`` arrays of one batch.
        """
        total = int(batch_size) * self.params.bytes_per_instance()
        return total, total / (1024.0 ** 2)

    def __repr__(self) -> str:
        """Return a description that never includes the secret."""
        params = self.params
        return (
            f"{type(self).__name__}(n={params.n}, q={params.q}, sigma={params.sigma}, "
            f"h={params.hamming_weight}, structure='{params.structure}', "
            f"bounded_a={params.bounded_a})"
        )
