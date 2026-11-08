"""Experimental cuBLAS GEMMs with explicit operand storage and FP32 accumulation/output."""

from functools import cache

import numpy as np

from ..core.device import _lazy_cupy

BACKENDS = ("cuda_tf32", "cuda_fp16", "cuda_bf16")
MIXED = ("cuda_fp16", "cuda_bf16")
_ONE = np.array(1.0, dtype=np.float32)
_ZERO = np.array(0.0, dtype=np.float32)


@cache
def _bf16_pack():
    return _lazy_cupy().ElementwiseKernel(
        "float32 x",
        "uint16 bits",
        r"""
        const unsigned u = __float_as_uint(x);
        // Round to nearest, ties to even. Keep a NaN payload when truncation would yield infinity.
        bits = ((u & 0x7fffffffU) > 0x7f800000U)
            ? (unsigned short)((u >> 16) | 0x40U)
            : (unsigned short)((u + 0x7fffU + ((u >> 16) & 1U)) >> 16);
    """,
        "mf_pack_bf16",
    )


def pack(a, backend):
    cp = _lazy_cupy()
    if not isinstance(a, cp.ndarray) or a.dtype != np.float32 or a.device.id != cp.cuda.runtime.getDevice():
        raise ValueError("experimental GEMMs take FP32 master tensors on the active CUDA device")
    if backend == "cuda_fp16":
        return a.astype(cp.float16)
    if backend == "cuda_bf16":
        return _bf16_pack()(a)
    if backend == "cuda_tf32":
        return a
    raise ValueError(f"unknown precision backend {backend}")


def gemm(a, b, backend, *, trans_a=False, trans_b=False):
    """Packed row-major A/B. Reverse operands for cuBLAS column-major C transpose."""
    from cupy_backends.cuda.api import runtime
    from cupy_backends.cuda.libs import cublas

    cp = _lazy_cupy()
    storage = {
        "cuda_fp16": (np.float16, runtime.CUDA_R_16F),
        "cuda_bf16": (np.uint16, runtime.CUDA_R_16BF),
        "cuda_tf32": (np.float32, runtime.CUDA_R_32F),
    }
    dtype, operand_type = storage[backend]
    if any(
        not isinstance(v, cp.ndarray)
        or v.ndim != 2
        or v.dtype != dtype
        or v.device.id != cp.cuda.runtime.getDevice()
        for v in (a, b)
    ):
        raise ValueError("GEMM operands must be packed matrices on the active CUDA device")
    m, k = a.shape[::-1] if trans_a else a.shape
    kb, n = b.shape[::-1] if trans_b else b.shape
    if k != kb or max(m, n, k) > np.iinfo(np.int32).max:
        raise ValueError("invalid cuBLAS dimensions")
    if not min(m, n, k):
        return cp.zeros((m, n), dtype=cp.float32)
    a, b = cp.ascontiguousarray(a), cp.ascontiguousarray(b)
    out = cp.empty((m, n), dtype=cp.float32)
    handle = cp.cuda.device.get_cublas_handle()
    stream = cp.cuda.get_current_stream().ptr
    previous = cublas.getStream(handle)
    pointer_mode = cublas.getPointerMode(handle)
    if pointer_mode != cublas.CUBLAS_POINTER_MODE_HOST:
        cublas.setPointerMode(handle, cublas.CUBLAS_POINTER_MODE_HOST)
    if previous != stream:
        cublas.setStream(handle, stream)
    compute = cublas.CUBLAS_COMPUTE_32F_FAST_TF32 if backend == "cuda_tf32" else cublas.CUBLAS_COMPUTE_32F
    try:
        cublas.gemmEx(
            handle,
            int(trans_b),
            int(trans_a),
            n,
            m,
            k,
            _ONE.ctypes.data,
            b.data.ptr,
            operand_type,
            b.shape[1],
            a.data.ptr,
            operand_type,
            a.shape[1],
            _ZERO.ctypes.data,
            out.data.ptr,
            runtime.CUDA_R_32F,
            n,
            compute,
            cublas.CUBLAS_GEMM_DEFAULT,
        )
    finally:
        if previous != stream:
            cublas.setStream(handle, previous)
        if pointer_mode != cublas.CUBLAS_POINTER_MODE_HOST:
            cublas.setPointerMode(handle, pointer_mode)
    return out
