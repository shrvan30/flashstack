"""Shared fixtures and the GPU-marker skip rule.

Tests marked `gpu` need both a CUDA device and the `flashattn_cuda` extension.
CI runs on a CPU-only runner, so they are skipped there rather than failing.
"""

import pytest


def _gpu_unavailable_reason() -> str | None:
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a hard dependency
        return "torch is not installed"
    if not torch.cuda.is_available():
        return "no CUDA device available"
    try:
        import flashattn_cuda  # noqa: F401
    except ImportError:
        return "flashattn_cuda extension is not installed"
    return None


def pytest_collection_modifyitems(config, items):
    reason = _gpu_unavailable_reason()
    if reason is None:
        return
    skip = pytest.mark.skip(reason=reason)
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)
