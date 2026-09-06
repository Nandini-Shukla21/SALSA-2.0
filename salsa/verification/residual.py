"""Independent residual verification of a recovered secret candidate.

This is the phase-20 verifier, promoted from a script into the library so the
end-to-end pipeline can call it directly.  **The mathematics is unchanged**; a
test asserts this module reproduces the phase-20 numbers exactly.

What makes the test meaningful
------------------------------
:func:`residual_statistics` is the whole verifier, and its signature is the
guarantee: it takes ``(A, b, candidate, q, sigma)`` and nothing else.  There is
no parameter through which the true secret, a training label or a model output
could reach it.  The data generator must of course use the secret to build
``b`` -- that is what makes ``b`` a real LWE sample -- but only ``A`` and ``b``
cross into this module.

Why the test discriminates
--------------------------
If the candidate equals the secret then ``b - A c = e (mod q)``: the centered
residual *is* the error, tight around zero with the configured ``sigma``.  If
the candidate is wrong by any nonzero ``delta``, the residual is
``A delta + e (mod q)``, which for a random ``A`` is close to uniform on
``Z_q`` with standard deviation ``sqrt((q^2 - 1) / 12)``.  At ``q = 251`` that
is about 72.5 against a sigma of 3, and that gap of roughly 24x is what carries
the evidence.

Residues are mapped to centered representatives before any statistic is taken.
Without that a residue of ``q - 1`` reads as a large positive error when it is
really ``-1``.
"""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence

import numpy as np

__all__ = [
    "ACCEPTANCE_CRITERIA",
    "VerificationResult",
    "center_residues",
    "residual_statistics",
    "uniform_reference_std",
    "verify_candidate",
]

#: The phase-20 acceptance criteria, stated as text so a report can quote them.
#: They are expressed against the configured sigma and against the baselines --
#: never against the true secret.
ACCEPTANCE_CRITERIA: Dict[str, str] = {
    "c1_std_close_to_sigma": "centered residual std <= 1.5 * sigma",
    "c2_mean_abs_small": "mean |centered residual| <= 1.5 * sigma * sqrt(2/pi)",
    "c3_beats_baselines": "candidate std < 0.25 * (smallest baseline std)",
    "c4_within_three_sigma": "fraction of |centered residual| <= 3 sigma is >= 0.99",
}


def center_residues(values: np.ndarray, q: int) -> np.ndarray:
    """Map residues in ``[0, q)`` to their centered representatives.

    Args:
        values: Integer residues.
        q: Modulus.

    Returns:
        The same values mapped into ``[-q/2, q/2)``.
    """
    reduced = np.asarray(values, dtype=np.int64) % int(q)
    return np.where(reduced > int(q) // 2, reduced - int(q), reduced)


def uniform_reference_std(q: int) -> float:
    """Standard deviation a wrong candidate's residual should show.

    A wrong candidate leaves ``A delta + e mod q``, close to uniform on ``Z_q``.

    Args:
        q: Modulus.

    Returns:
        ``sqrt((q^2 - 1) / 12)``.
    """
    return math.sqrt((int(q) ** 2 - 1) / 12.0)


def residual_statistics(
    A: np.ndarray, b: np.ndarray, candidate: Sequence[int], q: int, sigma: float
) -> Dict[str, Any]:
    """Residual statistics for one candidate on one sample set.

    **This function sees no secret.**  Its only inputs are the public matrix,
    the public targets, the candidate under test and the public parameters.

    Args:
        A: Public matrix, shape ``(m, n)``.
        b: Public targets, shape ``(m,)``.
        candidate: Binary vector under test, length ``n``.
        q: Modulus.
        sigma: Configured error scale, used only as a reference point.

    Returns:
        A dictionary of residual statistics.

    Raises:
        ValueError: If the shapes are inconsistent.
    """
    matrix = np.asarray(A, dtype=np.int64)
    targets = np.asarray(b, dtype=np.int64).reshape(-1)
    vector = np.asarray(candidate, dtype=np.int64).reshape(-1)
    if matrix.ndim != 2:
        raise ValueError(f"A must be 2-D, got {matrix.shape}.")
    if matrix.shape[0] != targets.size:
        raise ValueError(
            f"A has {matrix.shape[0]} rows but b has {targets.size} entries.")
    if matrix.shape[1] != vector.size:
        raise ValueError(
            f"A has {matrix.shape[1]} columns but the candidate has {vector.size}.")

    raw = (targets - matrix @ vector) % int(q)
    centered = center_residues(raw, q)
    absolute = np.abs(centered)
    return {
        "samples": int(centered.size),
        "mean": float(centered.mean()),
        "std": float(centered.std(ddof=1)),
        "median": float(np.median(centered)),
        "mean_abs": float(absolute.mean()),
        "median_abs": float(np.median(absolute)),
        "max_abs": int(absolute.max()),
        "percentiles_abs": {
            str(p): float(np.percentile(absolute, p)) for p in (50, 75, 90, 95, 99, 100)
        },
        "fraction_within_1_sigma": float((absolute <= 1 * sigma).mean()),
        "fraction_within_2_sigma": float((absolute <= 2 * sigma).mean()),
        "fraction_within_3_sigma": float((absolute <= 3 * sigma).mean()),
    }


@dataclass
class VerificationResult:
    """Outcome of verifying one candidate against fresh public samples.

    Attributes:
        passes: Whether every acceptance criterion held.
        criteria: Per-criterion booleans.
        statistics: Residual statistics for the candidate.
        baseline_statistics: The same statistics for each incorrect baseline.
        sigma: The configured error scale used as the reference.
        uniform_reference_std: What a wrong candidate should show.
        samples: Number of fresh samples used.
        notes: Anything worth carrying into a report.
    """

    passes: bool
    criteria: Dict[str, bool]
    statistics: Dict[str, Any]
    baseline_statistics: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    sigma: float = 0.0
    uniform_reference_std: float = 0.0
    samples: int = 0
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Return a serialisable view.  Contains no ground truth."""
        return {
            "passes": self.passes,
            "criteria": self.criteria,
            "acceptance_criteria": ACCEPTANCE_CRITERIA,
            "statistics": self.statistics,
            "baseline_statistics": self.baseline_statistics,
            "sigma": self.sigma,
            "uniform_reference_std": self.uniform_reference_std,
            "samples": self.samples,
            "notes": self.notes,
        }


def verify_candidate(
    A: np.ndarray,
    b: np.ndarray,
    candidate: Sequence[int],
    q: int,
    sigma: float,
    baselines: Optional[Dict[str, Sequence[int]]] = None,
) -> VerificationResult:
    """Run the residual consistency check on a candidate.

    Args:
        A: Public matrix from a fresh sample set.
        b: Public targets from that set.
        candidate: The candidate under test.
        q: Modulus.
        sigma: Configured error scale.
        baselines: Incorrect candidates to compare against.  Defaults to
            all-zeros and all-ones, which are the two that a sparse secret makes
            look deceptively good.

    Returns:
        A :class:`VerificationResult`.  No ground truth is consulted.
    """
    length = np.asarray(candidate).size
    if baselines is None:
        baselines = {
            "all_zeros": np.zeros(length, dtype=np.int64),
            "all_ones": np.ones(length, dtype=np.int64),
        }

    statistics = residual_statistics(A, b, candidate, q, sigma)
    baseline_statistics = {
        name: residual_statistics(A, b, vector, q, sigma)
        for name, vector in baselines.items()
    }
    smallest_baseline = (min(entry["std"] for entry in baseline_statistics.values())
                         if baseline_statistics else uniform_reference_std(q))

    criteria = {
        "c1_std_close_to_sigma": bool(statistics["std"] <= 1.5 * sigma),
        "c2_mean_abs_small": bool(
            statistics["mean_abs"] <= 1.5 * sigma * math.sqrt(2.0 / math.pi)),
        "c3_beats_baselines": bool(statistics["std"] < 0.25 * smallest_baseline),
        "c4_within_three_sigma": bool(statistics["fraction_within_3_sigma"] >= 0.99),
    }
    return VerificationResult(
        passes=all(criteria.values()),
        criteria=criteria,
        statistics=statistics,
        baseline_statistics=baseline_statistics,
        sigma=float(sigma),
        uniform_reference_std=uniform_reference_std(q),
        samples=int(statistics["samples"]),
        notes=("The verifier received only (A, b, candidate, q, sigma). "
               "No secret, model output or training label reached it."),
    )
