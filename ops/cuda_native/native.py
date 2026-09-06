"""Python boundary for the native C/C++ CUDA scheduler.

CuPy owns the arrays and allocations. The native DLL owns kernel launches,
the CUDA stream binding, and the cuBLAS calls.
"""

from __future__ import annotations

import ctypes
import importlib.util
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from ...core.device import _lazy_cupy


ROOT = Path(__file__).resolve().parents[3]
DLL_PATH = ROOT / ".codex" / "native-cuda" / "native_cublas.dll"
_nvidia_spec = importlib.util.find_spec("nvidia")
_torch_spec = importlib.util.find_spec("torch")
PYTHON_SITE = Path(next(iter(_nvidia_spec.submodule_search_locations))).parent if _nvidia_spec else Path()
TORCH_SITE = Path(next(iter(_torch_spec.submodule_search_locations))).parent if _torch_spec else Path()
_MISSING = Path("__missing_native_dependency__")
_NVRTC_DIR = TORCH_SITE / "torch" / "lib"
_TOOLKIT_BIN = Path(os.environ.get("CUDA_PATH", "")) / "bin" if os.environ.get("CUDA_PATH") else _MISSING


def _first_existing(*candidates):
    return next((candidate for candidate in candidates if candidate.exists()), _MISSING)


# Prefer Python-provided CUDA 12 DLLs when present, then fall back to the
# installed Toolkit or the DLLs shipped in the local PyTorch wheel.
CUBLAS_DLL = _first_existing(
    *sorted((PYTHON_SITE / "nvidia" / "cublas" / "bin").glob("cublas64_*.dll")),
    _TOOLKIT_BIN / "cublas64_12.dll",
    _NVRTC_DIR / "cublas64_12.dll",
)
NVRTC_DLL = _first_existing(
    _NVRTC_DIR / "nvrtc64_120_0.dll",
    *sorted(_NVRTC_DIR.glob("nvrtc64_*.dll")),
    _TOOLKIT_BIN / "nvrtc64_120_0.dll",
)
_DLL_DIR_HANDLES = []


class NativeCudaError(RuntimeError):
    pass


@dataclass
class NativePoolContext:
    input_shape: tuple
    output_shape: tuple
    kernel_size: tuple
    stride: tuple
    argmax: object
    device_id: int


def _u64(value):
    return ctypes.c_uint64(int(value))


def _i32(value):
    return ctypes.c_int(int(value))


def _configure(lib):
    lib.mf_init.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
    lib.mf_init.restype = ctypes.c_int
    lib.mf_last_error.argtypes = []
    lib.mf_last_error.restype = ctypes.c_char_p
    forward_types = [ctypes.c_uint64] * 5 + [ctypes.c_int] * 13 + [ctypes.c_uint64]
    backward_types = [ctypes.c_uint64] * 8 + [ctypes.c_int] * 13 + [ctypes.c_uint64]
    lib.mf_forward.argtypes = forward_types
    lib.mf_forward.restype = ctypes.c_int
    lib.mf_backward.argtypes = backward_types
    lib.mf_backward.restype = ctypes.c_int
    pool_forward_types = [ctypes.c_uint64] * 3 + [ctypes.c_int] * 10 + [ctypes.c_uint64]
    pool_backward_types = [ctypes.c_uint64] * 3 + [ctypes.c_int] * 10 + [ctypes.c_uint64]
    lib.mf_pool_forward.argtypes = pool_forward_types
    lib.mf_pool_forward.restype = ctypes.c_int
    lib.mf_pool_backward.argtypes = pool_backward_types
    lib.mf_pool_backward.restype = ctypes.c_int
    return lib


@lru_cache(maxsize=1)
def _library():
    if not DLL_PATH.exists():
        raise NativeCudaError(
            f"native CUDA DLL is missing: {DLL_PATH}. "
            "Run tools/build_native_cuda.py first."
        )
    if not CUBLAS_DLL.exists() or not NVRTC_DLL.exists():
        raise NativeCudaError(
            "the native CUDA runtime dependencies are missing; expected "
            f"{CUBLAS_DLL} and {NVRTC_DLL}"
        )
    for directory in (DLL_PATH.parent, CUBLAS_DLL.parent, NVRTC_DLL.parent, Path(r"C:\mingw64\bin")):
        if directory.exists() and hasattr(os, "add_dll_directory"):
            _DLL_DIR_HANDLES.append(os.add_dll_directory(str(directory)))
    lib = _configure(ctypes.CDLL(str(DLL_PATH)))
    status = lib.mf_init(str(CUBLAS_DLL).encode(), str(NVRTC_DLL).encode())
    if status != 0:
        detail = lib.mf_last_error().decode(errors="replace")
        raise NativeCudaError(f"native CUDA initialization failed ({status}): {detail}")
    return lib


def is_available():
    return DLL_PATH.exists() and CUBLAS_DLL.exists() and NVRTC_DLL.exists()


def _array(value, name):
    cp = _lazy_cupy()
    if not isinstance(value, cp.ndarray) or value.dtype != cp.float32:
        raise TypeError(f"{name} must be a CuPy float32 array")
    return cp.ascontiguousarray(value)


def _dims(x, weight, stride, padding):
    if x.ndim != 4 or weight.ndim != 4 or x.shape[1] != weight.shape[1]:
        raise ValueError("x and weight must be NCHW/OIHW arrays with matching channels")
    stride = tuple(int(v) for v in stride)
    padding = tuple(int(v) for v in padding)
    if len(stride) != 2 or len(padding) != 2 or min(stride) <= 0 or min(padding) < 0:
        raise ValueError("stride and padding must contain two valid integers")
    n, ci, h, w = map(int, x.shape)
    co, _, kh, kw = map(int, weight.shape)
    oh = (h + 2 * padding[0] - kh) // stride[0] + 1
    ow = (w + 2 * padding[1] - kw) // stride[1] + 1
    if oh <= 0 or ow <= 0:
        raise ValueError("convolution output shape is not positive")
    return (n, ci, h, w, co, kh, kw, oh, ow, stride[0], stride[1], padding[0], padding[1])


def _check(status, lib):
    if status:
        detail = lib.mf_last_error().decode(errors="replace")
        raise NativeCudaError(f"native CUDA call failed ({status}): {detail}")


def conv2d_forward(x, weight, bias=None, *, stride=(1, 1), padding=(0, 0)):
    cp = _lazy_cupy()
    x, weight = _array(x, "x"), _array(weight, "weight")
    if bias is not None:
        bias = _array(bias, "bias")
        if bias.shape != (weight.shape[0],):
            raise ValueError("bias must have shape (out_channels,)")
    dims = _dims(x, weight, stride, padding)
    n, ci, h, w, co, kh, kw, oh, ow, sh, sw, ph, pw = dims
    m, k = n * oh * ow, ci * kh * kw
    cols = cp.empty((m, k), dtype=cp.float32)
    rows = cp.empty((m, co), dtype=cp.float32)
    lib = _library()
    _check(lib.mf_forward(
        _u64(x.data.ptr), _u64(weight.data.ptr), _u64(bias.data.ptr if bias is not None else 0),
        _u64(cols.data.ptr), _u64(rows.data.ptr),
        *[_i32(v) for v in dims], _u64(cp.cuda.get_current_stream().ptr)), lib)
    y = rows.reshape(n, oh, ow, co).transpose(0, 3, 1, 2)
    return y, cols


def conv2d_backward(x, weight, grad_y, cols, *, stride=(1, 1), padding=(0, 0), need_bias_grad=True):
    cp = _lazy_cupy()
    x, weight, grad_y, cols = (_array(x, "x"), _array(weight, "weight"),
                               _array(grad_y, "grad_y"), _array(cols, "cols"))
    dims = _dims(x, weight, stride, padding)
    n, ci, h, w, co, kh, kw, oh, ow, sh, sw, ph, pw = dims
    if grad_y.shape != (n, co, oh, ow):
        raise ValueError("grad_y shape does not match convolution output")
    m, k = n * oh * ow, ci * kh * kw
    if cols.shape != (m, k):
        raise ValueError("cols shape does not match convolution input")
    dy = _array(grad_y.transpose(0, 2, 3, 1).reshape(m, co), "dy")
    dx = cp.empty_like(x)
    dw = cp.empty_like(weight)
    grad_cols = cp.empty((m, k), dtype=cp.float32)
    db = cp.empty((co,), dtype=cp.float32) if need_bias_grad else None
    lib = _library()
    _check(lib.mf_backward(
        _u64(x.data.ptr), _u64(weight.data.ptr), _u64(dy.data.ptr), _u64(cols.data.ptr),
        _u64(dx.data.ptr), _u64(dw.data.ptr), _u64(grad_cols.data.ptr),
        _u64(db.data.ptr if db is not None else 0),
        *[_i32(v) for v in dims], _u64(cp.cuda.get_current_stream().ptr)), lib)
    return dx, dw, db


def _pool_dims(x, kernel_size, stride):
    if x.ndim != 4:
        raise ValueError("x must be an NCHW array")
    kernel = tuple(int(v) for v in kernel_size)
    stride = tuple(int(v) for v in stride)
    if len(kernel) != 2 or len(stride) != 2 or min(kernel) <= 0 or min(stride) <= 0:
        raise ValueError("kernel_size and stride must contain positive pairs")
    n, c, h, w = map(int, x.shape)
    oh = (h - kernel[0]) // stride[0] + 1
    ow = (w - kernel[1]) // stride[1] + 1
    if oh <= 0 or ow <= 0:
        raise ValueError("pooling window is larger than the input")
    return n, c, h, w, oh, ow, kernel[0], kernel[1], stride[0], stride[1]


def maxpool2d_forward(x, *, kernel_size=(2, 2), stride=(2, 2)):
    cp = _lazy_cupy()
    x = _array(x, "x")
    dims = _pool_dims(x, kernel_size, stride)
    n, c, h, w, oh, ow, kh, kw, sh, sw = dims
    y = cp.empty((n, c, oh, ow), dtype=cp.float32)
    argmax = cp.empty((n, c, oh, ow), dtype=cp.int32)
    lib = _library()
    _check(lib.mf_pool_forward(
        _u64(x.data.ptr), _u64(y.data.ptr), _u64(argmax.data.ptr),
        *[_i32(v) for v in dims], _u64(cp.cuda.get_current_stream().ptr)), lib)
    return y, NativePoolContext(x.shape, y.shape, (kh, kw), (sh, sw), argmax, x.device.id)


def maxpool2d_backward(grad_y, context):
    cp = _lazy_cupy()
    if not isinstance(context, NativePoolContext) or context.device_id != cp.cuda.runtime.getDevice():
        raise ValueError("context must come from maxpool2d_forward on the current device")
    grad_y = _array(grad_y, "grad_y")
    if grad_y.shape != context.output_shape:
        raise ValueError("grad_y/context shape does not match pool output")
    n, c, h, w = map(int, context.input_shape)
    oh, ow = map(int, context.output_shape[2:])
    kh, kw = context.kernel_size
    sh, sw = context.stride
    dx = cp.empty(context.input_shape, dtype=cp.float32)
    dims = (n, c, h, w, oh, ow, kh, kw, sh, sw)
    lib = _library()
    _check(lib.mf_pool_backward(
        _u64(grad_y.data.ptr), _u64(context.argmax.data.ptr), _u64(dx.data.ptr),
        *[_i32(v) for v in dims], _u64(cp.cuda.get_current_stream().ptr)), lib)
    return dx
