"""Framework-wide default floating dtype for parameters and buffers.

Layers create their parameters in this dtype unless they are given one explicitly. It is
float32 because GeForce GPUs run FP64 at 1/32 to 1/64 of the FP32 rate; use
``set_default_dtype(np.float64)`` (or a layer's ``dtype=``) for float64 models such as
finite-difference gradient checks.
"""

from __future__ import annotations

import numpy as np

_DEFAULT_DTYPE = np.dtype(np.float32)


def get_default_dtype() -> np.dtype:
    return _DEFAULT_DTYPE


def set_default_dtype(dtype) -> np.dtype:
    """Set the default parameter dtype (float32 or float64) and return the previous one."""
    global _DEFAULT_DTYPE
    resolved = np.dtype(dtype)
    if resolved not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError(f"default dtype must be float32 or float64, got {resolved}")
    previous, _DEFAULT_DTYPE = _DEFAULT_DTYPE, resolved
    return previous


def resolve_dtype(dtype=None) -> np.dtype:
    return _DEFAULT_DTYPE if dtype is None else np.dtype(dtype)
