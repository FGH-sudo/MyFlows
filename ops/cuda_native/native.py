"""Python boundary for the native C/C++ CUDA scheduler.

CuPy owns the arrays and allocations. The native DLL compiles its kernels with NVRTC for the
current device (caching the binary under build/native-cuda/kernel-cache), owns every kernel
launch on the current CuPy stream, and runs the GEMMs on one of ``GEMM_ENGINES``.
"""

from __future__ import annotations

import ctypes
import importlib.util
import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from ...core.device import _lazy_cupy


ROOT = Path(__file__).resolve().parents[3]
DLL_PATH = ROOT / "build" / "native-cuda" / "native_cublas.dll"
KERNEL_CACHE_DIR = ROOT / "build" / "native-cuda" / "kernel-cache"
_nvidia_spec = importlib.util.find_spec("nvidia")
_torch_spec = importlib.util.find_spec("torch")
PYTHON_SITE = Path(next(iter(_nvidia_spec.submodule_search_locations))).parent if _nvidia_spec else Path()
TORCH_SITE = Path(next(iter(_torch_spec.submodule_search_locations))).parent if _torch_spec else Path()
_MISSING = Path("__missing_native_dependency__")
_TORCH_LIB = TORCH_SITE / "torch" / "lib"
_TOOLKIT_BIN = Path(os.environ.get("CUDA_PATH", "")) / "bin" if os.environ.get("CUDA_PATH") else _MISSING
INT32_LIMIT = 2**31 - 1

_STATUS = {1: "initialization failed", 2: "invalid argument", 3: "CUDA driver error",
           4: "cuBLAS error", 5: "wrong CUDA context"}
_ACTIVATIONS = {None: 0, "relu": 1, "leaky_relu": 2}
MATMUL_VARIANTS = {"basic": 0, "tiled": 1, "shared": 2, "cublas": 3, "register": 4}
# GEMM engine of conv2d_* and gemm: register (the lecture-19 register-tiled kernels in FP32, five tile
# shapes, split-K for outputs too small to fill the GPU) or cublas (cublasSgemm, IEEE FP32 math).
GEMM_ENGINES = {"cublas": 0, "register": 1}
GEMM_TILES = ("128x128", "128x64", "256x32", "64x128", "32x256")  # BM x BN of the register engine
# fp32: sequential FMA; kahan: FMA per tile with TwoSum-compensated tile sums; dot2: compensated
# TwoProduct/TwoSum on every term (see the kernel source in native_cublas.cpp).
MATMUL_ACCUMULATION = {"fp32": 0, "kahan": 1, "dot2": 2}


def _first_existing(*candidates):
    return next((candidate for candidate in candidates if candidate.exists()), _MISSING)


# The nvidia-*-cu12 wheels pinned in pyproject.toml come first so the native path does not depend
# on whichever PyTorch build is installed; the PyTorch wheel and the CUDA Toolkit are fallbacks.
CUBLAS_DLL = _first_existing(
    *sorted((PYTHON_SITE / "nvidia" / "cublas" / "bin").glob("cublas64_*.dll")),
    _TOOLKIT_BIN / "cublas64_12.dll",
    _TORCH_LIB / "cublas64_12.dll",
)
NVRTC_DLL = _first_existing(
    PYTHON_SITE / "nvidia" / "cuda_nvrtc" / "bin" / "nvrtc64_120_0.dll",
    _TORCH_LIB / "nvrtc64_120_0.dll",
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


def _ansi(path):
    # LoadLibraryExA and fopen in the DLL interpret narrow strings in the ANSI code page.
    return str(path).encode("mbcs" if os.name == "nt" else "utf-8")


def _configure(lib):
    u64, i64, i32, f32 = ctypes.c_uint64, ctypes.c_int64, ctypes.c_int, ctypes.c_float
    lib.mf_init.argtypes = [ctypes.c_char_p] * 3
    lib.mf_init.restype = i32
    lib.mf_last_error.argtypes = []
    lib.mf_last_error.restype = ctypes.c_char_p
    lib.mf_info.argtypes = []
    lib.mf_info.restype = ctypes.c_char_p
    lib.mf_set_debug_sync.argtypes = [i32]
    lib.mf_set_debug_sync.restype = None
    lib.mf_conv_forward.argtypes = [u64] * 6 + [i32] * 13 + [i32, f32, i32, u64, i64, u64]
    lib.mf_conv_forward.restype = i32
    lib.mf_conv_backward.argtypes = [u64] * 10 + [i32] * 13 + [i32, i32, f32, i32, u64, i64, u64]
    lib.mf_conv_backward.restype = i32
    lib.mf_pool_forward.argtypes = [u64] * 3 + [i32] * 10 + [u64]
    lib.mf_pool_forward.restype = i32
    lib.mf_pool_backward.argtypes = [u64] * 3 + [i32] * 10 + [u64]
    lib.mf_pool_backward.restype = i32
    lib.mf_matmul.argtypes = [i32] * 3 + [u64] * 4 + [i32] * 3 + [u64]
    lib.mf_matmul.restype = i32
    lib.mf_gemm.argtypes = [i32] * 3 + [u64] * 4 + [i32] * 6 + [u64, i64, u64]
    lib.mf_gemm.restype = i32
    lib.mf_gemm_workspace.argtypes = [i32] * 3
    lib.mf_gemm_workspace.restype = i64
    lib.mf_gemm_plan.argtypes = [i32] * 3 + [ctypes.POINTER(i32)] * 2
    lib.mf_gemm_plan.restype = i32
    return lib


def _error(status, lib):
    detail = lib.mf_last_error().decode(errors="replace")
    return NativeCudaError(f"native CUDA {_STATUS.get(status, 'call failed')} ({status}): {detail}")


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
    for directory in (DLL_PATH.parent, CUBLAS_DLL.parent, NVRTC_DLL.parent):
        if directory.exists() and hasattr(os, "add_dll_directory"):
            _DLL_DIR_HANDLES.append(os.add_dll_directory(str(directory)))
    # NVRTC loads nvrtc-builtins64_<ver>.dll by bare name while compiling, which bypasses the
    # directories above; once a module of that name is loaded the loader reuses it.
    for builtins in sorted(NVRTC_DLL.parent.glob("nvrtc-builtins64_*.dll")):
        _DLL_DIR_HANDLES.append(ctypes.WinDLL(str(builtins)) if os.name == "nt" else ctypes.CDLL(str(builtins)))
    lib = _configure(ctypes.CDLL(str(DLL_PATH)))
    cp = _lazy_cupy()
    # mf_init binds the module and the cuBLAS handle to the current context, so make sure the
    # runtime's primary context of the current device exists on this thread.
    cp.cuda.Device(cp.cuda.runtime.getDevice()).use()
    cp.empty((1,), dtype=cp.float32)
    cache = b""
    if os.environ.get("MYFLOWS_NATIVE_KERNEL_CACHE", "1") != "0":
        KERNEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache = _ansi(KERNEL_CACHE_DIR)
    status = lib.mf_init(_ansi(CUBLAS_DLL), _ansi(NVRTC_DLL), cache)
    if status != 0:
        raise _error(status, lib)
    return lib


def is_available():
    return DLL_PATH.exists() and CUBLAS_DLL.exists() and NVRTC_DLL.exists()


def info():
    """Build facts of the loaded kernels: target architecture, binary kind, NVRTC, cache state."""
    return {**json.loads(_library().mf_info().decode()), "nvrtc_dll": str(NVRTC_DLL), "cublas_dll": str(CUBLAS_DLL)}


def set_debug_sync(enabled=True):
    """Synchronize after every native launch so asynchronous faults name the failing kernel."""
    _library().mf_set_debug_sync(1 if enabled else 0)


def _check(status, lib):
    if status:
        raise _error(status, lib)


def _array(value, name):
    cp = _lazy_cupy()
    if not isinstance(value, cp.ndarray) or value.dtype != cp.float32:
        raise TypeError(f"{name} must be a CuPy float32 array")
    return cp.ascontiguousarray(value)


def _guard_index(**sizes):
    for name, size in sizes.items():
        if int(size) > INT32_LIMIT:
            raise ValueError(f"{name} has {int(size)} elements; native kernels index with int32 "
                             f"(limit {INT32_LIMIT}), so use a smaller batch")


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
    m, k = n * oh * ow, ci * kh * kw
    _guard_index(im2col_columns=m * k, output=m * co, input=n * ci * h * w)
    if n > 65535:
        raise ValueError("native convolution supports at most 65535 samples per call")
    return (n, ci, h, w, co, kh, kw, oh, ow, stride[0], stride[1], padding[0], padding[1])


def _activation(activation, alpha):
    if activation not in _ACTIVATIONS:
        raise ValueError(f"activation must be one of {sorted(a for a in _ACTIVATIONS if a)} or None")
    alpha = float(alpha)
    if activation == "leaky_relu" and not alpha > 0.0:
        raise ValueError("a fused leaky_relu needs alpha > 0")
    return _ACTIVATIONS[activation], ctypes.c_float(alpha)


def _engine(engine):
    if engine not in GEMM_ENGINES:
        raise ValueError(f"engine must be one of {sorted(GEMM_ENGINES)}")
    return GEMM_ENGINES[engine]


def _workspace(lib, engine, *gemms):
    """Split-K workspace of the register engine, sized for the largest of the call's (M, N, K) GEMMs."""
    floats = 0
    if engine == GEMM_ENGINES["register"]:
        for m, n, k in gemms:
            size = lib.mf_gemm_workspace(m, n, k)
            if size < 0:
                raise NativeCudaError(f"native CUDA call failed: {lib.mf_last_error().decode(errors='replace')}")
            floats = max(floats, size)
    if not floats:
        return None, 0
    cp = _lazy_cupy()
    return cp.empty((floats,), dtype=cp.float32), floats


@lru_cache(maxsize=8)
def _sm_count(device_id):
    cp = _lazy_cupy()
    return int(cp.cuda.Device(device_id).attributes["MultiProcessorCount"])


def _spatial_blocks(n, ohw, co, device_id):
    # Enough blocks to fill the GPU (~12 per SM) while each block still walks several spatial
    # tiles, which also keeps the FP64 bias partial buffer at roughly that many rows.
    tiles = (ohw + 31) // 32
    per_sample = -(-12 * _sm_count(device_id) // max(1, n * ((co + 31) // 32)))
    return max(1, min(tiles, per_sample))


def conv2d_forward(x, weight, bias=None, *, stride=(1, 1), padding=(0, 0), activation=None, alpha=0.0,
                   engine="register"):
    """Return the contiguous NCHW output (bias and activation fused) and the im2col columns.
    ``engine`` runs the GEMM (see ``GEMM_ENGINES``)."""
    cp = _lazy_cupy()
    engine = _engine(engine)
    x, weight = _array(x, "x"), _array(weight, "weight")
    if bias is not None:
        bias = _array(bias, "bias")
        if bias.shape != (weight.shape[0],):
            raise ValueError("bias must have shape (out_channels,)")
    dims = _dims(x, weight, stride, padding)
    act, alpha = _activation(activation, alpha)
    n, ci, h, w, co, kh, kw, oh, ow, sh, sw, ph, pw = dims
    m, k = n * oh * ow, ci * kh * kw
    cols = cp.empty((m, k), dtype=cp.float32)
    rows = cp.empty((m, co), dtype=cp.float32)
    y = cp.empty((n, co, oh, ow), dtype=cp.float32)
    lib = _library()
    workspace, floats = _workspace(lib, engine, (m, co, k))
    _check(lib.mf_conv_forward(
        _u64(x.data.ptr), _u64(weight.data.ptr), _u64(bias.data.ptr if bias is not None else 0),
        _u64(cols.data.ptr), _u64(rows.data.ptr), _u64(y.data.ptr),
        *[_i32(v) for v in dims], _i32(act), alpha, _i32(engine),
        _u64(workspace.data.ptr if workspace is not None else 0), ctypes.c_int64(floats),
        _u64(cp.cuda.get_current_stream().ptr)), lib)
    return y, cols


def conv2d_backward(x, weight, grad_y, cols, *, stride=(1, 1), padding=(0, 0), need_bias_grad=True,
                    need_input_grad=True, activation=None, alpha=0.0, output=None, engine="register"):
    """Return (dx, dw, db). dx is None when ``need_input_grad`` is false, db when ``need_bias_grad``
    is false. A fused ``activation`` is differentiated through the forward ``output``. ``engine``
    runs the dW and dX GEMMs (see ``GEMM_ENGINES``)."""
    cp = _lazy_cupy()
    engine = _engine(engine)
    x, weight, grad_y, cols = (_array(x, "x"), _array(weight, "weight"),
                               _array(grad_y, "grad_y"), _array(cols, "cols"))
    dims = _dims(x, weight, stride, padding)
    act, alpha = _activation(activation, alpha)
    n, ci, h, w, co, kh, kw, oh, ow, sh, sw, ph, pw = dims
    if grad_y.shape != (n, co, oh, ow):
        raise ValueError("grad_y shape does not match convolution output")
    m, k = n * oh * ow, ci * kh * kw
    if cols.shape != (m, k):
        raise ValueError("cols shape does not match convolution input")
    gate = None
    if act:
        if output is None:
            raise ValueError("a fused activation needs the forward output")
        gate = _array(output, "output")
        if gate.shape != grad_y.shape:
            raise ValueError("output shape does not match grad_y")
    blocks = _spatial_blocks(n, oh * ow, co, x.device.id)
    dy_rows = cp.empty((m, co), dtype=cp.float32)
    dw = cp.empty_like(weight)
    grad_cols = cp.empty((m, k), dtype=cp.float32) if need_input_grad else None
    dx = cp.empty_like(x) if need_input_grad else None
    partial = cp.empty((n * blocks * co,), dtype=cp.float64) if need_bias_grad else None
    db = cp.empty((co,), dtype=cp.float32) if need_bias_grad else None
    pointer = lambda array: _u64(array.data.ptr if array is not None else 0)
    lib = _library()
    workspace, floats = _workspace(lib, engine, (co, k, m), *([(m, k, co)] if need_input_grad else []))
    _check(lib.mf_conv_backward(
        pointer(weight), pointer(grad_y), pointer(gate), pointer(cols), pointer(dy_rows), pointer(dw),
        pointer(grad_cols), pointer(dx), pointer(partial), pointer(db),
        *[_i32(v) for v in dims], _i32(blocks), _i32(act), alpha, _i32(engine), pointer(workspace),
        ctypes.c_int64(floats), _u64(cp.cuda.get_current_stream().ptr)), lib)
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
    _guard_index(input=n * c * h * w)
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


def matmul(a, b, bias=None, *, variant="shared", tile=32, accumulate="kahan", out=None):
    """Row-major ``a @ b (+ bias)`` with one of the lecture-19 kernels or cuBLAS.

    ``variant``: basic (one thread per output, 1-D grid), tiled (TILE x TILE thread blocks over
    global memory), shared (TILE x TILE shared-memory tiles), register (128 x 128 block tiles,
    8 x 8 outputs per thread in registers, double-buffered shared tiles; ``tile`` is ignored) or
    cublas. ``accumulate`` selects the summation of the hand-written kernels: fp32, kahan or dot2
    (see ``MATMUL_ACCUMULATION``); register gives the same bits as basic for each of them.
    """
    cp = _lazy_cupy()
    if variant not in MATMUL_VARIANTS:
        raise ValueError(f"variant must be one of {sorted(MATMUL_VARIANTS)}")
    if accumulate not in MATMUL_ACCUMULATION:
        raise ValueError(f"accumulate must be one of {sorted(MATMUL_ACCUMULATION)}")
    if int(tile) not in (16, 32):
        raise ValueError("tile must be 16 or 32")
    a, b = _array(a, "a"), _array(b, "b")
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError("matmul needs 2-D operands with matching inner dimensions")
    m, k = map(int, a.shape)
    n = int(b.shape[1])
    if min(m, n, k) <= 0:
        raise ValueError("matmul dimensions must be positive")
    _guard_index(a=m * k, b=k * n, c=m * n)
    if bias is not None:
        bias = _array(bias, "bias").reshape(-1)
        if bias.shape != (n,):
            raise ValueError("bias must have one value per output column")
    if out is None:
        out = cp.empty((m, n), dtype=cp.float32)
    elif not isinstance(out, cp.ndarray) or out.dtype != cp.float32 or out.shape != (m, n) or not out.flags.c_contiguous:
        raise ValueError("out must be a contiguous CuPy float32 array of the result shape")
    lib = _library()
    _check(lib.mf_matmul(
        _i32(MATMUL_VARIANTS[variant]), _i32(tile), _i32(MATMUL_ACCUMULATION[accumulate]),
        _u64(a.data.ptr), _u64(b.data.ptr), _u64(bias.data.ptr if bias is not None else 0), _u64(out.data.ptr),
        _i32(m), _i32(n), _i32(k), _u64(cp.cuda.get_current_stream().ptr)), lib)
    return out


def gemm_plan(m, n, k):
    """Tile (``GEMM_TILES``) and split-K slices the register engine uses for an (m, n) output over k."""
    if min(m, n, k) <= 0:
        raise ValueError("gemm dimensions must be positive")
    lib = _library()
    shape, slices = ctypes.c_int(), ctypes.c_int()
    _check(lib.mf_gemm_plan(_i32(m), _i32(n), _i32(k), ctypes.byref(shape), ctypes.byref(slices)), lib)
    return GEMM_TILES[shape.value], slices.value


def gemm(a, b, bias=None, *, trans_a=False, trans_b=False, engine="register", out=None):
    """Row-major ``op(a) @ op(b) (+ bias)`` on a GEMM engine of the native layers.

    ``op(a)`` is ``a.T`` when ``trans_a`` (``a`` stored K x M), ``op(b)`` is ``b.T`` when
    ``trans_b`` (``b`` stored N x K); not both. ``engine`` is one of ``GEMM_ENGINES``.
    """
    cp = _lazy_cupy()
    engine = _engine(engine)
    if trans_a and trans_b:
        raise ValueError("gemm supports a.T @ b and a @ b.T but not a.T @ b.T")
    a, b = _array(a, "a"), _array(b, "b")
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("gemm needs 2-D operands")
    m, k = (int(a.shape[1]), int(a.shape[0])) if trans_a else (int(a.shape[0]), int(a.shape[1]))
    n, k_b = (int(b.shape[0]), int(b.shape[1])) if trans_b else (int(b.shape[1]), int(b.shape[0]))
    if k != k_b:
        raise ValueError("gemm operands have different inner dimensions")
    if min(m, n, k) <= 0:
        raise ValueError("gemm dimensions must be positive")
    _guard_index(a=m * k, b=k * n, c=m * n)
    if bias is not None:
        bias = _array(bias, "bias").reshape(-1)
        if bias.shape != (n,):
            raise ValueError("bias must have one value per output column")
    if out is None:
        out = cp.empty((m, n), dtype=cp.float32)
    elif not isinstance(out, cp.ndarray) or out.dtype != cp.float32 or out.shape != (m, n) or not out.flags.c_contiguous:
        raise ValueError("out must be a contiguous CuPy float32 array of the result shape")
    lib = _library()
    workspace, floats = _workspace(lib, engine, (m, n, k))
    _check(lib.mf_gemm(
        _i32(engine), _i32(trans_a), _i32(trans_b), _u64(a.data.ptr), _u64(b.data.ptr),
        _u64(bias.data.ptr if bias is not None else 0), _u64(out.data.ptr), _i32(m), _i32(n), _i32(k),
        _i32(a.shape[1]), _i32(b.shape[1]), _i32(n), _u64(workspace.data.ptr if workspace is not None else 0),
        ctypes.c_int64(floats), _u64(cp.cuda.get_current_stream().ptr)), lib)
    return out
