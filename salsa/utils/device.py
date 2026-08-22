"""CPU-first device abstraction.

Rules enforced by this module:

* **CPU is the default.**  An accelerator is used only when the configuration
  explicitly asks for one (``device.prefer`` = ``cuda`` / ``mps`` / ``auto``).
* **No CUDA assumptions.**  Nothing here imports CUDA APIs unconditionally, and
  a missing accelerator degrades to CPU with a recorded warning.
* **Torch is imported lazily.**  The repository stays importable (and phase-1
  tests stay runnable) on a machine where PyTorch is not installed yet.

The environment report produced here is part of the experimental record: CPU
timings are meaningless without the machine description that produced them.
"""

import os
import platform
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

__all__ = [
    "DeviceSpec",
    "configure_threads",
    "device_report",
    "memory_usage_mb",
    "resolve_device",
    "torch_available",
]


def torch_available() -> bool:
    """Return True when PyTorch can be imported in this environment."""
    try:
        import torch  # noqa: F401
    except Exception:  # pragma: no cover - depends on the environment
        return False
    return True


def _import_torch() -> Any:
    """Import and return the ``torch`` module, or raise a helpful error."""
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "PyTorch is required for this operation. Install the CPU build with:\n"
            "  python -m pip install torch --index-url "
            "https://download.pytorch.org/whl/cpu"
        ) from exc
    return torch


def _cuda_available() -> bool:
    """Return True when a usable CUDA device is present."""
    if not torch_available():
        return False
    torch = _import_torch()
    try:
        return bool(torch.cuda.is_available())
    except Exception:  # pragma: no cover - driver-dependent
        return False


def _mps_available() -> bool:
    """Return True when Apple Metal (MPS) acceleration is present."""
    if not torch_available():
        return False
    torch = _import_torch()
    try:
        backend = getattr(torch.backends, "mps", None)
        return bool(backend is not None and backend.is_available())
    except Exception:  # pragma: no cover - platform-dependent
        return False


@dataclass
class DeviceSpec:
    """A resolved compute placement.

    Attributes:
        type: ``"cpu"``, ``"cuda"`` or ``"mps"``.
        index: Device index for multi-GPU machines (None on CPU/MPS).
        requested: What the configuration asked for.
        dtype: Requested compute dtype name.
        warnings: Human-readable notes about any fallback that occurred.
    """

    type: str = "cpu"
    index: Optional[int] = None
    requested: str = "cpu"
    dtype: str = "float32"
    warnings: List[str] = field(default_factory=list)

    @property
    def is_cpu(self) -> bool:
        """True when this specification resolved to the CPU."""
        return self.type == "cpu"

    @property
    def name(self) -> str:
        """Torch-style device string, e.g. ``"cpu"`` or ``"cuda:0"``."""
        if self.type == "cuda" and self.index is not None:
            return f"cuda:{self.index}"
        return self.type

    def torch_device(self) -> Any:
        """Return the corresponding ``torch.device`` (requires PyTorch)."""
        torch = _import_torch()
        return torch.device(self.name)

    def torch_dtype(self) -> Any:
        """Return the corresponding ``torch`` dtype (requires PyTorch)."""
        torch = _import_torch()
        mapping = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        if self.dtype not in mapping:
            raise ValueError(f"Unsupported dtype '{self.dtype}'.")
        return mapping[self.dtype]

    def to_dict(self) -> Dict[str, Any]:
        """Return a serialisable description of this device placement."""
        return {
            "type": self.type,
            "index": self.index,
            "name": self.name,
            "requested": self.requested,
            "dtype": self.dtype,
            "warnings": list(self.warnings),
        }


def resolve_device(
    prefer: str = "cpu",
    allow_gpu_fallback: bool = True,
    dtype: str = "float32",
    index: Optional[int] = None,
) -> DeviceSpec:
    """Resolve a device request into a concrete :class:`DeviceSpec`.

    Args:
        prefer: ``"cpu"`` (default), ``"auto"``, ``"cuda"`` or ``"mps"``.
            ``"auto"`` picks an accelerator when one exists, otherwise CPU.
        allow_gpu_fallback: When True, an unavailable accelerator falls back to
            the CPU with a warning; when False it raises ``RuntimeError``.
        dtype: Requested compute dtype name.
        index: Optional CUDA device index.

    Returns:
        The resolved :class:`DeviceSpec`.  ``spec.warnings`` records any
        fallback so it can be written into the run metadata.

    Raises:
        ValueError: If ``prefer`` is not a recognised value.
        RuntimeError: If an accelerator was demanded and is unavailable while
            ``allow_gpu_fallback`` is False.
    """
    prefer = str(prefer).lower()
    if prefer not in ("cpu", "auto", "cuda", "mps"):
        raise ValueError(
            f"Unknown device preference '{prefer}'; expected 'cpu', 'auto', "
            "'cuda' or 'mps'."
        )

    spec = DeviceSpec(type="cpu", index=None, requested=prefer, dtype=dtype)

    if prefer == "cpu":
        return spec

    if prefer == "auto":
        if _cuda_available():
            spec.type, spec.index = "cuda", index if index is not None else 0
        elif _mps_available():
            spec.type = "mps"
        else:
            spec.warnings.append(
                "device.prefer='auto': no accelerator detected, using CPU."
            )
        return spec

    available = _cuda_available() if prefer == "cuda" else _mps_available()
    if available:
        spec.type = prefer
        if prefer == "cuda":
            spec.index = index if index is not None else 0
        return spec

    message = (
        f"device.prefer='{prefer}' was requested but no {prefer.upper()} device is "
        "available in this environment."
    )
    if not allow_gpu_fallback:
        raise RuntimeError(message)
    spec.warnings.append(message + " Falling back to CPU.")
    return spec


def configure_threads(
    threads: Optional[int] = None, interop_threads: Optional[int] = None
) -> Dict[str, Optional[int]]:
    """Configure CPU thread pools for reproducible, well-behaved CPU training.

    Args:
        threads: Intra-op thread count (None leaves the library default).
        interop_threads: Inter-op thread count (None leaves the default).

    Returns:
        A dictionary describing the thread settings actually in force.  Values
        are ``None`` when PyTorch is not installed.
    """
    result: Dict[str, Optional[int]] = {"threads": None, "interop_threads": None}
    if not torch_available():
        return result

    torch = _import_torch()
    if threads is not None:
        torch.set_num_threads(int(threads))
        # Keep the common BLAS backends consistent with the torch setting.
        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ.setdefault(var, str(int(threads)))
    if interop_threads is not None:
        try:
            torch.set_num_interop_threads(int(interop_threads))
        except RuntimeError:
            # Torch forbids changing this after the inter-op pool has started.
            pass

    result["threads"] = int(torch.get_num_threads())
    try:
        result["interop_threads"] = int(torch.get_num_interop_threads())
    except Exception:  # pragma: no cover - version-dependent
        result["interop_threads"] = None
    return result


def memory_usage_mb() -> Dict[str, float]:
    """Return current process memory usage in megabytes.

    Returns:
        ``{"rss_mb": ..., "peak_rss_mb": ...}``.  Missing measurements are
        reported as ``-1.0`` rather than silently as ``0.0``, so an unavailable
        reading is never mistaken for a real one.
    """
    usage: Dict[str, float] = {"rss_mb": -1.0, "peak_rss_mb": -1.0}
    try:
        import psutil  # type: ignore
    except ImportError:
        return usage

    try:
        process = psutil.Process(os.getpid())
        info = process.memory_info()
        usage["rss_mb"] = float(info.rss) / (1024.0 ** 2)
        peak = getattr(info, "peak_wset", None)  # Windows
        if peak is None:
            peak = getattr(process.memory_info(), "vms", None)
        if peak is not None:
            usage["peak_rss_mb"] = float(peak) / (1024.0 ** 2)
    except Exception:  # pragma: no cover - platform-dependent
        return usage
    return usage


def device_report() -> Dict[str, Any]:
    """Describe the machine and the numerical stack.

    This report is stored with every run: CPU latency and memory numbers are
    only interpretable together with the hardware that produced them.
    """
    report: Dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "cpu_count_logical": os.cpu_count(),
        "torch_available": torch_available(),
        "torch_version": None,
        "torch_threads": None,
        "cuda_available": False,
        "cuda_device_count": 0,
        "cuda_device_names": [],
        "mps_available": False,
        "total_ram_gb": None,
    }

    if torch_available():
        torch = _import_torch()
        report["torch_version"] = torch.__version__
        try:
            report["torch_threads"] = int(torch.get_num_threads())
        except Exception:  # pragma: no cover
            pass
        report["cuda_available"] = _cuda_available()
        if report["cuda_available"]:
            try:
                count = int(torch.cuda.device_count())
                report["cuda_device_count"] = count
                report["cuda_device_names"] = [
                    torch.cuda.get_device_name(i) for i in range(count)
                ]
            except Exception:  # pragma: no cover - driver-dependent
                pass
        report["mps_available"] = _mps_available()

    try:
        import psutil  # type: ignore

        report["total_ram_gb"] = round(
            psutil.virtual_memory().total / (1024.0 ** 3), 2
        )
    except Exception:
        pass

    return report
