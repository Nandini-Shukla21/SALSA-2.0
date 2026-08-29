"""LWE / RLWE data generation.

Public API
----------
Secrets (:mod:`salsa.data.secrets`)::

    generate_binary_secret(n, hamming_weight, seed=...)   exactly h ones
    generate_ternary_secret(n, hamming_weight, seed=...)  exactly h non-zeros
    generate_secret(n, hamming_weight, distribution, ...)
    secret_from_config(config)                            derived from the master seed

Samples (:mod:`salsa.data.lwe`, :mod:`salsa.data.rlwe`)::

    generate_error(size, sigma, distribution, seed=...)
    generate_uniform_matrix(m, n, q, ...)                 plain LWE rows
    generate_rlwe_matrix(m, n, q, variant=..., ...)       structured rows
    generate_lwe_sample(params, m, secret, ...)
    generate_rlwe_sample(params, m, secret, ...)
    generate_batch(params, m, secret, ...)                dispatches on structure

Encoding (:mod:`salsa.data.encoding`)::

    LatticeCodec.from_config(config)       the a -> b token representation
    codec.encode_input(a) / decode_input   token-level, readable path
    codec.encode_batch(A, b)               vectorised id arrays for training
    codec.lengths()                        sequence-length accounting

Streaming problems::

    build_problem(config, split="train")   -> LWEProblem | RLWEProblem
    problem.sample(m)                      -> LWESample      (public)
    problem.labeled_sample(m)              -> LabeledSample  (controlled use)
    problem.batch(i, batch_size)           -> reproducible batch i
    problem.iter_batches(num, batch_size)  -> memory-bounded stream

Secret isolation
----------------
:class:`~salsa.data.lwe.LWESample` carries ``A`` and ``b`` only.  The secret and
the error live in :class:`~salsa.data.lwe.GroundTruth`, reachable solely through
:class:`~salsa.data.lwe.LabeledSample` or an explicit
``problem.reveal_secret()``.  Phase 8+ recovery code takes ``LWESample``.
"""

from .encoding import (
    BOS,
    EOS,
    PAD,
    SEP,
    EncodingError,
    IntegerEncoder,
    LatticeCodec,
    SequenceLengths,
    Vocabulary,
    benchmark_codec,
    decode_integer,
    decode_lattice_input,
    decode_lattice_output,
    decode_vector,
    digit_width,
    encode_integer,
    encode_lattice_input,
    encode_lattice_output,
    encode_vector,
)
from .lwe import (
    ERROR_DISTRIBUTIONS,
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
from .rlwe import (
    RLWE_VARIANTS,
    RLWEProblem,
    circulant_matrix,
    generate_rlwe_matrix,
    generate_rlwe_sample,
    negacyclic_matrix,
    rotation_block,
)
from .secrets import (
    BINARY,
    SECRET_DISTRIBUTIONS,
    TERNARY,
    generate_binary_secret,
    generate_secret,
    generate_ternary_secret,
    hamming_weight,
    is_valid_secret,
    resolve_rng,
    secret_from_config,
)

__all__ = [
    # encoding
    "PAD",
    "BOS",
    "EOS",
    "SEP",
    "EncodingError",
    "Vocabulary",
    "IntegerEncoder",
    "LatticeCodec",
    "SequenceLengths",
    "digit_width",
    "encode_integer",
    "decode_integer",
    "encode_vector",
    "decode_vector",
    "encode_lattice_input",
    "decode_lattice_input",
    "encode_lattice_output",
    "decode_lattice_output",
    "benchmark_codec",
    # secrets
    "BINARY",
    "TERNARY",
    "SECRET_DISTRIBUTIONS",
    "generate_binary_secret",
    "generate_ternary_secret",
    "generate_secret",
    "secret_from_config",
    "hamming_weight",
    "is_valid_secret",
    "resolve_rng",
    # parameters and containers
    "LWEParams",
    "LWESample",
    "GroundTruth",
    "LabeledSample",
    "ERROR_DISTRIBUTIONS",
    "RLWE_VARIANTS",
    # generation
    "generate_error",
    "generate_uniform_matrix",
    "generate_rlwe_matrix",
    "generate_lwe_sample",
    "generate_rlwe_sample",
    "generate_batch",
    "circulant_matrix",
    "negacyclic_matrix",
    "rotation_block",
    # streaming problems
    "LWEProblem",
    "RLWEProblem",
    "build_problem",
]


def build_problem(config, split: str = "train", secret=None):
    """Build the streaming problem described by a configuration.

    Dispatches on ``config.lwe.structure``: ``"lwe"`` gives an
    :class:`~salsa.data.lwe.LWEProblem`, ``"rlwe"`` an
    :class:`~salsa.data.rlwe.RLWEProblem`.

    Args:
        config: A :class:`~salsa.utils.config.Config`.
        split: Data split label, e.g. ``"train"``, ``"valid"``, ``"test"``.
            Different splits draw independent streams from the *same* secret.
        secret: Optional explicit secret (overrides seed derivation).

    Returns:
        A problem object exposing the streaming API.

    Raises:
        ValueError: If ``config.lwe.structure`` is unknown.
    """
    structure = config.lwe.structure
    if structure == "lwe":
        return LWEProblem.from_config(config, split=split, secret=secret)
    if structure == "rlwe":
        return RLWEProblem.from_config(config, split=split, secret=secret)
    raise ValueError(
        f"Unknown lwe.structure '{structure}'; expected 'lwe' or 'rlwe'."
    )
