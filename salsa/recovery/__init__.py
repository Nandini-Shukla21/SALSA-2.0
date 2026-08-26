"""Secret-recovery algorithms.

Implemented (phase 10):
    ``direct``  SALSA Algorithm 1 -- the chosen-input attack that probes each
        secret coordinate with ``a = K * e_i`` and reads the model's answer.

Planned:
    ``distinguisher``  decision-LWE recovery (phase 11).
    ``candidate``      candidate bookkeeping and de-duplication.

Secret isolation
----------------
Nothing in this package accepts, imports or references the true secret or the
error vector.  Recovery consumes the model, the codec and public parameters
only; comparing a candidate against the truth belongs to the evaluation layer
and happens after recovery has finished.
"""

from .direct import (
    BINARIZATION_METHODS,
    CoordinateOutcome,
    CoordinateProbe,
    DirectRecovery,
    DirectRecoveryReport,
    KRecoveryResult,
    build_probe_matrix,
    build_probes,
    probe_separation,
    ring_distance,
)

__all__ = [
    "BINARIZATION_METHODS",
    "CoordinateOutcome",
    "CoordinateProbe",
    "DirectRecovery",
    "DirectRecoveryReport",
    "KRecoveryResult",
    "build_probe_matrix",
    "build_probes",
    "probe_separation",
    "ring_distance",
]
