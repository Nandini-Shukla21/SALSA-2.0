"""Tests for the CPU-first device abstraction.

The critical property under test: nothing in this project may quietly require
a GPU, and an unavailable accelerator must degrade to CPU with a recorded
warning rather than crashing or pretending to have succeeded.
"""

import pytest

from salsa.utils.device import (
    DeviceSpec,
    configure_threads,
    device_report,
    memory_usage_mb,
    resolve_device,
    torch_available,
)


def test_default_resolution_is_cpu() -> None:
    """With no arguments the resolver returns the CPU."""
    spec = resolve_device()
    assert spec.type == "cpu"
    assert spec.is_cpu
    assert spec.name == "cpu"
    assert spec.warnings == []


def test_explicit_cpu_never_touches_accelerators() -> None:
    """prefer='cpu' resolves to CPU even on a GPU machine."""
    spec = resolve_device("cpu")
    assert spec.type == "cpu"
    assert spec.requested == "cpu"


def test_auto_falls_back_to_cpu_without_accelerator() -> None:
    """prefer='auto' is safe on a CPU-only machine and says what it did."""
    spec = resolve_device("auto")
    assert spec.type in ("cpu", "cuda", "mps")
    if spec.type == "cpu":
        assert any("auto" in w for w in spec.warnings)


def test_unavailable_accelerator_falls_back_with_a_warning() -> None:
    """A GPU request on a CPU-only machine warns instead of failing."""
    spec = resolve_device("cuda", allow_gpu_fallback=True)
    if spec.type == "cpu":
        assert spec.requested == "cuda"
        assert spec.warnings, "a silent fallback would hide the real device used"


def test_unavailable_accelerator_can_be_made_fatal() -> None:
    """With allow_gpu_fallback=False an unavailable GPU raises."""
    import salsa.utils.device as device_module

    if device_module._cuda_available():  # pragma: no cover - GPU machines
        pytest.skip("CUDA is available on this machine")
    with pytest.raises(RuntimeError, match="CUDA"):
        resolve_device("cuda", allow_gpu_fallback=False)


def test_unknown_preference_is_rejected() -> None:
    """Typos in device.prefer are errors."""
    with pytest.raises(ValueError, match="Unknown device preference"):
        resolve_device("gpu")


def test_device_spec_serialises() -> None:
    """A DeviceSpec can be embedded in run metadata."""
    payload = resolve_device("cpu").to_dict()
    assert payload["type"] == "cpu"
    assert set(payload) == {"type", "index", "name", "requested", "dtype", "warnings"}


def test_cuda_spec_name_includes_index() -> None:
    """cuda specs stringify as 'cuda:<index>'."""
    assert DeviceSpec(type="cuda", index=1).name == "cuda:1"
    assert DeviceSpec(type="mps").name == "mps"


def test_device_report_has_the_fields_needed_to_interpret_timings() -> None:
    """CPU benchmarks are meaningless without the machine description."""
    report = device_report()
    for key in (
        "platform",
        "cpu_count_logical",
        "torch_available",
        "cuda_available",
        "mps_available",
        "total_ram_gb",
    ):
        assert key in report
    assert isinstance(report["torch_available"], bool)
    assert report["torch_available"] == torch_available()


def test_memory_usage_reports_unavailable_readings_honestly() -> None:
    """A missing measurement is -1.0, never a fabricated 0.0."""
    usage = memory_usage_mb()
    assert set(usage) == {"rss_mb", "peak_rss_mb"}
    for value in usage.values():
        assert value == -1.0 or value > 0.0


def test_configure_threads_is_safe_without_torch() -> None:
    """Thread configuration degrades gracefully when torch is absent."""
    result = configure_threads(threads=2)
    assert set(result) == {"threads", "interop_threads"}
    if torch_available():
        assert result["threads"] == 2
    else:
        assert result["threads"] is None


@pytest.mark.skipif(not torch_available(), reason="PyTorch is not installed")
def test_torch_device_and_dtype_resolution() -> None:
    """The spec converts to real torch objects when torch is present."""
    import torch

    spec = resolve_device("cpu", dtype="float32")
    assert spec.torch_device() == torch.device("cpu")
    assert spec.torch_dtype() is torch.float32

    with pytest.raises(ValueError):
        DeviceSpec(dtype="float8").torch_dtype()
