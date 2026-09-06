"""Mathematical verification of recovered secret candidates.

``residual`` implements the residual-statistics test on held-out LWE samples:
a correct candidate leaves ``b - A c = e``, tight around zero at the configured
sigma, while any wrong candidate leaves something close to uniform on ``Z_q``.

Nothing in this package accepts, imports or references the true secret. The
verifier's signature is the guarantee: ``(A, b, candidate, q, sigma)``.
"""

from .residual import (
    ACCEPTANCE_CRITERIA,
    VerificationResult,
    center_residues,
    residual_statistics,
    uniform_reference_std,
    verify_candidate,
)

__all__ = [
    "ACCEPTANCE_CRITERIA",
    "VerificationResult",
    "center_residues",
    "residual_statistics",
    "uniform_reference_std",
    "verify_candidate",
]
