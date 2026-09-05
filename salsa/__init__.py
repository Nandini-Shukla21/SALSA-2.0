"""Lightweight SALSA (SALSA 2.0).

An independent, CPU-first re-implementation of the SALSA research workflow for
attacking Learning-With-Errors (LWE) lattice cryptography with neural sequence
models.

Scientific reference
--------------------
Wenger, Chen, Charton, Lauter. *SALSA: Attacking Lattice Cryptography with
Transformers*, NeurIPS 2022.

The reference paper and the original Meta implementation are used only to fix
the mathematical formulation, the data pipeline semantics and the evaluation
methodology.  No code is copied; the learned component is redesigned for a
~4-5M parameter budget that trains on an ordinary CPU.

Package layout
--------------
``salsa.data``          LWE / RLWE sample generation, secrets, integer encoding.
``salsa.models``        Compact neural architecture + parameter accounting.
``salsa.training``      Trainer, losses, metrics, seed management.
``salsa.recovery``      Direct and distinguisher secret-recovery algorithms.
``salsa.verification``  Mathematical residual verification of secret candidates.
``salsa.evaluation``    Evaluator and CPU benchmarking.
``salsa.utils``         Configuration, logging, device abstraction.

Only the phase-1 foundations (config / device / logging / seed) are implemented
at this point.  Every other subpackage is an intentional placeholder.
"""

__version__ = "0.1.0.dev0"
__all__ = ["__version__"]


from .pipeline import PipelineResult, run_salsa2_pipeline  # noqa: E402

__all__ = list(globals().get("__all__", [])) + [
    "run_salsa2_pipeline",
    "PipelineResult",
]
