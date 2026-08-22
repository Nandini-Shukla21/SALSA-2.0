"""Training subsystem.

Phase 1 provides seed management only.  The trainer, losses and metrics arrive
in phases 6-7.
"""

from .seed import SeedState, derive_seed, seed_worker, set_seed, temporary_seed

__all__ = [
    "SeedState",
    "derive_seed",
    "seed_worker",
    "set_seed",
    "temporary_seed",
]
