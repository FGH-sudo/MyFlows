"""Strict FP32/NCHW wrappers; finite input values are a caller precondition.

All copies, allocations and launches use the current CuPy stream. No implicit
device synchronization or host conversion is performed on the ordinary path.
"""

from dataclasses import dataclass
from numbers import Integral

import numpy as np

from ...core.device import _lazy_cupy
from .loader import get_kernel


def pair(value, name, minimum=1):
    values = (value, value) if isinstance(value, Integral) else value
    if not isinstance(values, (tuple, list)) or len(values) != 2:
        raise TypeError(f"{name} must be an integer or pair of integers")
    if any(isinstance(v, bool) or not isinstance(v, Integral) for v in values):
        raise TypeError(f"{name} must contain integers")
    if any(v < minimum or v > np.iinfo(np.int32).max for v in values):
        raise ValueError(f"{name} is outside supported bounds")
    return tuple(map(int, values))


def array(value, name, ndim, dtype=np.float32):
    cp = _lazy_cupy()
    if not isinstance(value, cp.ndarray):
        raise TypeError(f"{name} must be a CuPy CUDA array")
    if value.device.id != cp.cuda.runtime.getDevice():
        raise ValueError(f"{name} must be on the current CUDA device")
    if value.dtype != dtype:
        raise TypeError(f"{name} must have dtype {np.dtype(dtype)}")
    if value.ndim != ndim or any(d <= 0 for d in value.shape):
        raise ValueError(f"{name} must be a nonempty rank-{ndim} array")
    if value.size > np.iinfo(np.int32).max:
        raise ValueError(f"{name} exceeds the int32 indexing limit")
    return cp.ascontiguousarray(value)


def _output_shape(x_shape, co, kernel, stride, padding):
    n, _, h, w = x_shape
    oh = (h + 2 * padding[0] - kernel[0]) // stride[0] + 1
    ow = (w + 2 * padding[1] - kernel[1]) // stride[1] + 1
    if oh <= 0 or ow <= 0 or n * co * oh * ow > np.iinfo(np.int32).max:
        raise ValueError("output shape is nonpositive or exceeds the int32 indexing limit")
    return n, co, oh, ow


def _launch(filename, name, count, args, block_size):
    if block_size not in (64, 128, 256, 512):
        raise ValueError("block_size must be 64, 128, 256 or 512")
    get_kernel(filename, name)(((count + block_size - 1) // block_size,), (block_size,), args)


def _launch_blocks(filename, name, blocks, args, block_size, shared_mem=0):
    if block_size not in (64, 128, 256, 512):
        raise ValueError("block_size must be 64, 128, 256 or 512")
    get_kernel(filename, name)((int(blocks),), (block_size,), args, shared_mem=shared_mem)


def _launch_gemm(name, grid, args):
    get_kernel("gemm.cu", name)(grid, (16, 16), args)


def _conv_inputs(x, weight, stride, padding):
    x, weight = array(x, "x", 4), array(weight, "weight", 4)
    stride, padding = pair(stride, "stride"), pair(padding, "padding", 0)
    if x.shape[1] != weight.shape[1]:
        raise ValueError("input and weight channels differ")
    shape = _output_shape(x.shape, weight.shape[0], weight.shape[2:], stride, padding)
    dims = tuple(np.int32(v) for v in (*x.shape, weight.shape[0], *weight.shape[2:],
                                      *shape[2:], *stride, *padding))
    return x, weight, shape, dims


def conv2d_forward(x, weight, bias=None, *, stride=(1, 1), padding=(0, 0), block_size=256):
    cp = _lazy_cupy()
    x, weight, shape, dims = _conv_inputs(x, weight, stride, padding)
    if bias is not None:
        bias = array(bias, "bias", 1)
        if bias.shape != (weight.shape[0],):
            raise ValueError("bias shape must equal (out_channels,)")
    y = cp.empty(shape, dtype=cp.float32)
    _launch("conv2d.cu", "conv2d_forward_direct", y.size,
            (x, weight, bias if bias is not None else weight, y, *dims, np.int32(bias is not None)), block_size)
    return y


def conv2d_backward(x, weight, grad_y, *, stride=(1, 1), padding=(0, 0), need_bias_grad=True, block_size=256):
    cp = _lazy_cupy()
    x, weight, shape, dims = _conv_inputs(x, weight, stride, padding)
    grad_y = array(grad_y, "grad_y", 4)
    if grad_y.shape != shape:
        raise ValueError("grad_y shape does not match convolution output")
    dx, dw = cp.empty_like(x), cp.empty_like(weight)
    db = bias_grad_cuda(grad_y, block_size=block_size) if need_bias_grad else None
    _launch("conv2d.cu", "conv2d_backward_input", dx.size, (weight, grad_y, dx, *dims), block_size)
    _launch_blocks("conv2d.cu", "conv2d_backward_weight", dw.size,
                   (x, grad_y, dw, *dims), block_size, shared_mem=block_size * np.dtype(np.float32).itemsize)
    return dx, dw, db


def bias_grad_cuda(grad_y, *, block_size=256):
    """Reduce dY over N/H/W with one parallel block per output channel."""
    cp = _lazy_cupy()
    grad_y = array(grad_y, "grad_y", 4)
    db = cp.empty(grad_y.shape[1], dtype=cp.float32)
    _launch_blocks("conv2d.cu", "conv2d_backward_bias", db.size,
                   (grad_y, db, *(np.int32(v) for v in grad_y.shape)), block_size,
                   shared_mem=block_size * np.dtype(np.float32).itemsize)
    return db


def im2col_cuda(x, kernel_size, *, stride=(1, 1), padding=(0, 0), block_size=256):
    """Create an im2col matrix with the stage-two CUDA C kernel."""
    x = array(x, "x", 4)
    kernel, stride, padding = pair(kernel_size, "kernel_size"), pair(stride, "stride"), pair(padding, "padding", 0)
    shape = _output_shape(x.shape, x.shape[1], kernel, stride, padding)
    dims = tuple(np.int32(v) for v in (*x.shape, x.shape[1], *kernel, *shape[2:], *stride, *padding))
    cols = _lazy_cupy().empty((int(shape[0]) * int(shape[2]) * int(shape[3]),
                               int(x.shape[1]) * int(kernel[0]) * int(kernel[1])), dtype=np.float32)
    _launch("im2col.cu", "conv2d_im2col_forward", cols.size,
            (x, cols, *dims[:4], *dims[5:]), block_size)
    context = {"input_shape": x.shape, "output_shape": shape, "kernel_size": kernel,
               "stride": stride, "padding": padding}
    return cols, context


def col2im_cuda(grad_cols, input_shape, kernel_size, *, stride=(1, 1), padding=(0, 0), block_size=256):
    """Scatter an im2col gradient with the stage-two CUDA C kernel."""
    cp = _lazy_cupy()
    kernel, stride, padding = pair(kernel_size, "kernel_size"), pair(stride, "stride"), pair(padding, "padding", 0)
    if len(input_shape) != 4 or any(int(v) <= 0 for v in input_shape):
        raise ValueError("input_shape must be a nonempty rank-4 shape")
    input_shape = tuple(int(v) for v in input_shape)
    shape = _output_shape(input_shape, input_shape[1], kernel, stride, padding)
    expected = (shape[0] * shape[2] * shape[3], input_shape[1] * kernel[0] * kernel[1])
    grad_cols = array(grad_cols, "grad_cols", 2)
    if grad_cols.shape != expected:
        raise ValueError("grad_cols shape does not match input_shape and convolution parameters")
    dims = tuple(np.int32(v) for v in (*input_shape, input_shape[1], *kernel, *shape[2:], *stride, *padding))
    dx = cp.empty(input_shape, dtype=cp.float32)
    _launch("im2col.cu", "conv2d_col2im_backward", dx.size,
            (grad_cols, dx, *dims[:4], *dims[5:]), block_size)
    return dx


def conv2d_im2col_forward(x, weight, bias=None, *, stride=(1, 1), padding=(0, 0), block_size=256):
    """CUDA im2col followed by the same CuPy GEMM used by the reference path."""
    cp = _lazy_cupy()
    x, weight, shape, _ = _conv_inputs(x, weight, stride, padding)
    if bias is not None:
        bias = array(bias, "bias", 1)
        if bias.shape != (weight.shape[0],):
            raise ValueError("bias shape must equal (out_channels,)")
    cols, _ = im2col_cuda(x, weight.shape[2:], stride=stride, padding=padding, block_size=block_size)
    rows = cols @ weight.reshape(weight.shape[0], -1).T
    y = rows.reshape(int(shape[0]), int(shape[2]), int(shape[3]), int(weight.shape[0])).transpose(0, 3, 1, 2)
    if bias is not None:
        y = y + bias.reshape(1, int(weight.shape[0]), 1, 1)
    return y, cols


def gemm_forward_cuda(cols, weight_rows, *, block_size=256):
    """Compute cols @ weight_rows.T with the tiled CUDA GEMM kernel."""
    cp = _lazy_cupy()
    cols = array(cols, "cols", 2)
    weight_rows = array(weight_rows, "weight_rows", 2)
    m, k = cols.shape
    n, weight_k = weight_rows.shape
    if k != weight_k:
        raise ValueError("GEMM dimensions do not match")
    output = cp.empty((m, n), dtype=cp.float32)
    _launch_gemm("gemm_nt", ((n + 15) // 16, (m + 15) // 16),
                 (cols, weight_rows, output, np.int32(m), np.int32(n), np.int32(k)))
    return output


def gemm_input_cuda(dy_rows, weight_rows, *, block_size=256):
    """Compute dy_rows @ weight_rows with the tiled CUDA GEMM kernel."""
    cp = _lazy_cupy()
    dy_rows = array(dy_rows, "dy_rows", 2)
    weight_rows = array(weight_rows, "weight_rows", 2)
    m, n = dy_rows.shape
    weight_n, k = weight_rows.shape
    if n != weight_n:
        raise ValueError("GEMM dimensions do not match")
    output = cp.empty((m, k), dtype=cp.float32)
    _launch_gemm("gemm_nn", ((k + 15) // 16, (m + 15) // 16),
                 (dy_rows, weight_rows, output, np.int32(m), np.int32(n), np.int32(k)))
    return output


def gemm_weight_cuda(dy_rows, cols, *, block_size=256):
    """Compute dy_rows.T @ cols with the tiled CUDA GEMM kernel."""
    cp = _lazy_cupy()
    dy_rows = array(dy_rows, "dy_rows", 2)
    cols = array(cols, "cols", 2)
    m, n = dy_rows.shape
    cols_m, k = cols.shape
    if m != cols_m:
        raise ValueError("GEMM dimensions do not match")
    output = cp.empty((n, k), dtype=cp.float32)
    _launch_gemm("gemm_tn", ((k + 15) // 16, (n + 15) // 16),
                 (dy_rows, cols, output, np.int32(m), np.int32(n), np.int32(k)))
    return output


def conv2d_im2col_gemm_forward(x, weight, bias=None, *, stride=(1, 1), padding=(0, 0), block_size=256):
    """CUDA im2col followed by the self-written tiled CUDA GEMM."""
    cp = _lazy_cupy()
    x, weight, shape, _ = _conv_inputs(x, weight, stride, padding)
    if bias is not None:
        bias = array(bias, "bias", 1)
        if bias.shape != (weight.shape[0],):
            raise ValueError("bias shape must equal (out_channels,)")
    cols, _ = im2col_cuda(x, weight.shape[2:], stride=stride, padding=padding, block_size=block_size)
    rows = gemm_forward_cuda(cols, weight.reshape(weight.shape[0], -1), block_size=block_size)
    y = rows.reshape(int(shape[0]), int(shape[2]), int(shape[3]), int(weight.shape[0])).transpose(0, 3, 1, 2)
    if bias is not None:
        y = y + bias.reshape(1, int(weight.shape[0]), 1, 1)
    return y, cols


def conv2d_im2col_backward(x, weight, grad_y, cols, *, stride=(1, 1), padding=(0, 0), need_bias_grad=True, block_size=256):
    cp = _lazy_cupy()
    x, weight, shape, _ = _conv_inputs(x, weight, stride, padding)
    grad_y = array(grad_y, "grad_y", 4)
    if grad_y.shape != shape:
        raise ValueError("grad_y shape does not match convolution output")
    expected_cols = (int(shape[0]) * int(shape[2]) * int(shape[3]),
                     int(x.shape[1]) * int(weight.shape[2]) * int(weight.shape[3]))
    if not isinstance(cols, cp.ndarray) or cols.shape != expected_cols or cols.dtype != cp.float32:
        raise ValueError("cols does not match the forward im2col context")
    dy_rows = grad_y.transpose(0, 2, 3, 1).reshape(-1, int(weight.shape[0]))
    weight_rows = weight.reshape(int(weight.shape[0]), -1)
    dw = (dy_rows.T @ cols).reshape(weight.shape)
    grad_cols = dy_rows @ weight_rows
    dx = col2im_cuda(grad_cols, x.shape, weight.shape[2:], stride=stride, padding=padding, block_size=block_size)
    db = bias_grad_cuda(grad_y, block_size=block_size) if need_bias_grad else None
    return dx, dw, db


def conv2d_im2col_gemm_backward(x, weight, grad_y, cols, *, stride=(1, 1), padding=(0, 0),
                                need_bias_grad=True, block_size=256):
    """CUDA col2im plus self-written tiled CUDA GEMM backward path."""
    cp = _lazy_cupy()
    x, weight, shape, _ = _conv_inputs(x, weight, stride, padding)
    grad_y = array(grad_y, "grad_y", 4)
    if grad_y.shape != shape:
        raise ValueError("grad_y shape does not match convolution output")
    expected_cols = (int(shape[0]) * int(shape[2]) * int(shape[3]),
                     int(x.shape[1]) * int(weight.shape[2]) * int(weight.shape[3]))
    if not isinstance(cols, cp.ndarray) or cols.shape != expected_cols or cols.dtype != cp.float32:
        raise ValueError("cols does not match the forward im2col context")
    dy_rows = grad_y.transpose(0, 2, 3, 1).reshape(-1, int(weight.shape[0]))
    weight_rows = weight.reshape(int(weight.shape[0]), -1)
    dw = gemm_weight_cuda(dy_rows, cols, block_size=block_size).reshape(weight.shape)
    grad_cols = gemm_input_cuda(dy_rows, weight_rows, block_size=block_size)
    dx = col2im_cuda(grad_cols, x.shape, weight.shape[2:], stride=stride, padding=padding, block_size=block_size)
    db = bias_grad_cuda(grad_y, block_size=block_size) if need_bias_grad else None
    return dx, dw, db


@dataclass(frozen=True)
class PoolContext:
    input_shape: tuple
    output_shape: tuple
    kernel_size: tuple
    stride: tuple
    argmax: object
    device_id: int


def _pool_dims(context):
    return tuple(np.int32(v) for v in (*context.input_shape, *context.output_shape[2:],
                                      *context.kernel_size, *context.stride))


def maxpool2d_forward(x, *, kernel_size=(2, 2), stride=(2, 2), block_size=256):
    cp = _lazy_cupy()
    x = array(x, "x", 4)
    kernel, stride = pair(kernel_size, "kernel_size"), pair(stride, "stride")
    shape = _output_shape(x.shape, x.shape[1], kernel, stride, (0, 0))
    y, argmax = cp.empty(shape, dtype=cp.float32), cp.empty(shape, dtype=cp.int32)
    context = PoolContext(x.shape, shape, kernel, stride, argmax, x.device.id)
    _launch("maxpool2d.cu", "maxpool2d_forward_direct", y.size,
            (x, y, argmax, *_pool_dims(context)), block_size)
    return y, context


def maxpool2d_backward(grad_y, context, *, block_size=256):
    cp = _lazy_cupy()
    if not isinstance(context, PoolContext) or context.device_id != cp.cuda.runtime.getDevice():
        raise ValueError("context must come from maxpool2d_forward on the current device")
    grad_y = array(grad_y, "grad_y", 4)
    argmax = array(context.argmax, "argmax", 4, np.int32)
    if grad_y.shape != context.output_shape or argmax.shape != context.output_shape:
        raise ValueError("grad_y/context shape does not match pool output")
    dx = cp.empty(context.input_shape, dtype=cp.float32)
    _launch("maxpool2d.cu", "maxpool2d_backward_gather", dx.size,
            (grad_y, argmax, dx, *_pool_dims(context)), block_size)
    return dx
