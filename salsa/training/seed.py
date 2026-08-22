"""Deterministic seed management.

Reproducibility rule for this project: *one master seed per experiment*.  Every
other stream (secret generation, training data, validation data, model init,
recovery sampling) uses a seed **derived** from it by a stable hash, so:

* two runs of the same config produce the same secret and the same data;
* changing the data seed cannot accidentally change the model-init seed;
* derived seeds are stable across processes and Python versions (unlike
  :func:`hash`, which is randomised per interpreter run).

PyTorch is optional here so the seeding utilities can be tested before the
neural code exists.
"""

import contextlib
import hashlib
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional

import numpy as np

__all__ = [
    "SeedState",
    "derive_seed",
    "seed_worker",
    "set_seed",
    "temporary_seed",
]

#: Seeds are kept inside the range accepted by NumPy's legacy RandomState.
_SEED_MODULUS = 2 ** 32


@dataclass
class SeedState:
    """Record of what was seeded, stored in the run metadata.

    Attributes:
        seed: The master seed that was applied.
        deterministic: Whether deterministic algorithms were requested.
        torch_seeded: True when PyTorch was present and seeded.
        notes: Any caveats (e.g. a backend that cannot be made deterministic).
    """

    seed: int
    deterministic: bool
    torch_seeded: bool
    notes: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Return a serialisable description of the seeding state."""
        return {
            "seed": self.seed,
            "deterministic": self.deterministic,
            "torch_seeded": self.torch_seeded,
            "notes": list(self.notes),
        }


def derive_seed(master_seed: int, *parts: Any) -> int:
    """Derive a stable child seed from a master seed and labels.

    The derivation uses BLAKE2b over the textual representation of the inputs,
    so it is deterministic across processes, machines and Python versions.

    Args:
        master_seed: The experiment's master seed.
        *parts: Labels identifying the stream, e.g. ``"train"``, ``epoch``.

    Returns:
        An integer in ``[0, 2**32)`` suitable for NumPy and PyTorch.

    Example:
        >>> derive_seed(0, "train") == derive_seed(0, "train")
        True
        >>> derive_seed(0, "train") == derive_seed(0, "valid")
        False
    """
    payload = "|".join([str(int(master_seed))] + [str(p) for p in parts])
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % _SEED_MODULUS


def set_seed(seed: int, deterministic: bool = True) -> SeedState:
    """Seed Python, NumPy and (when installed) PyTorch.

    Args:
        seed: The master seed.  Must be non-negative.
        deterministic: Request deterministic algorithms and disable cuDNN
            autotuning.  Deterministic kernels can be slower; this is accepted
            because reproducibility outranks speed in this project.

    Returns:
        A :class:`SeedState` describing exactly what was seeded.

    Raises:
        ValueError: If ``seed`` is negative.
    """
    if seed < 0:
        raise ValueError("seed must be >= 0.")
    seed = int(seed) % _SEED_MODULUS
    notes: List[str] = []

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch_seeded = False
    try:
        import torch
    except ImportError:
        notes.append("PyTorch is not installed; only Python and NumPy were seeded.")
        torch = None  # type: ignore[assignment]

    if torch is not None:
        torch.manual_seed(seed)
        try:
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except Exception:  # pragma: no cover - driver-dependent
            notes.append("Could not seed CUDA devices.")
        torch_seeded = True

        if deterministic:
            try:
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True
            except Exception:  # pragma: no cover - build-dependent
                notes.append("cuDNN determinism flags are unavailable in this build.")
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except TypeError:  # pragma: no cover - older torch
                torch.use_deterministic_algorithms(True)
            except Exception:  # pragma: no cover - build-dependent
                notes.append("torch.use_deterministic_algorithms is unavailable.")

    return SeedState(
        seed=seed,
        deterministic=deterministic,
        torch_seeded=torch_seeded,
        notes=notes,
    )


def seed_worker(worker_id: int) -> None:
    """DataLoader ``worker_init_fn`` giving each worker a distinct stream.

    Args:
        worker_id: Worker index supplied by the DataLoader.
    """
    try:
        import torch

        base_seed = int(torch.initial_seed()) % _SEED_MODULUS
    except ImportError:  # pragma: no cover - torch-free fallback
        base_seed = 0
    worker_seed = derive_seed(base_seed, "worker", worker_id)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


@contextlib.contextmanager
def temporary_seed(seed: int) -> Iterator[None]:
    """Temporarily seed Python/NumPy/torch, restoring the previous state after.

    Useful for a reproducible sub-computation (e.g. drawing recovery probes)
    that must not disturb the training random stream.

    Args:
        seed: Seed applied inside the context.

    Yields:
        None.
    """
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state: Optional[Any] = None
    try:
        import torch

        torch_state = torch.get_rng_state()
    except ImportError:
        torch = None  # type: ignore[assignment]

    random.seed(int(seed) % _SEED_MODULUS)
    np.random.seed(int(seed) % _SEED_MODULUS)
    if torch is not None:
        torch.manual_seed(int(seed) % _SEED_MODULUS)

    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        if torch is not None and torch_state is not None:
            torch.set_rng_state(torch_state)
