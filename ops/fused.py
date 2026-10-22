"""Fused operator kernels for the CuPy path, with NumPy fallbacks that follow the same math.

CuPy runs every elementwise operator and reduction as a kernel of its own and materializes each
intermediate, which for the operators below costs several full passes over activation-sized
tensors plus one Python-side launch per step of the formula. The CUDA versions here are
hand-written kernels that CuPy compiles with NVRTC (ElementwiseKernel/RawKernel, cached on disk).

Accuracy rules shared by both backends:

* reductions accumulate in FP64 in a fixed order, so CUDA results are reproducible and within
  about one FP32 rounding of the exact value, and the NumPy fallbacks agree with them to a few ulp;
* the Adam step rounds every operation explicitly (``__f*_rn``) in the same order as the NumPy
  fallback, so a CPU and a GPU step on the same float32 inputs give bitwise identical results.
"""

from __future__ import annotations

from functools import cache

import numpy as np

from ..core.device import _lazy_cupy, get_device


def _on_gpu(*arrays):
    return get_device() == "cuda" and all(isinstance(a, _lazy_cupy().ndarray) for a in arrays)


def _same_float32(*arrays):
    return all(a.dtype == np.float32 for a in arrays)


# ---------------------------------------------------------------------------------------------
# Activations
# ---------------------------------------------------------------------------------------------
@cache
def _relu_backward_kernel():
    return _lazy_cupy().ElementwiseKernel(
        "T x, T g", "T out", "out += x > (T)0 ? g : (T)0", "mf_relu_backward_acc")


@cache
def _leaky_relu_backward_kernel():
    return _lazy_cupy().ElementwiseKernel(
        "T x, T g, T alpha", "T out", "out += x > (T)0 ? g : alpha * g", "mf_leaky_relu_backward_acc")


def relu_backward_accumulate(x, grad, out):
    """``out += grad`` where ``x > 0`` (one pass, dtype of ``out`` preserved)."""
    if _on_gpu(x, grad, out) and x.dtype == grad.dtype == out.dtype:
        _relu_backward_kernel()(x, grad, out)
    else:
        out += np.where(x > 0, grad, 0) if isinstance(out, np.ndarray) else _lazy_cupy().where(x > 0, grad, 0)
    return out


def leaky_relu_backward_accumulate(x, grad, out, alpha):
    """``out += grad`` where ``x > 0`` and ``alpha * grad`` elsewhere."""
    if _on_gpu(x, grad, out) and x.dtype == grad.dtype == out.dtype:
        _leaky_relu_backward_kernel()(x, grad, out.dtype.type(alpha), out)
    else:
        module = np if isinstance(out, np.ndarray) else _lazy_cupy()
        out += module.where(x > 0, grad, float(alpha) * grad)
    return out


# ---------------------------------------------------------------------------------------------
# BatchNorm2d (NCHW): FP64 statistics, one normalization pass, two-pass backward.
# ---------------------------------------------------------------------------------------------
_BN_SOURCE = r"""
// Partial sums over a slice of one channel: blockIdx.x = channel, blockIdx.y = slice of the
// flattened (n, hw) range. Statistics use the shifted data x - x[n=0, c, 0] so that the
// one-pass variance does not cancel catastrophically; everything accumulates in FP64 and the
// block reduction is a fixed tree, so results are reproducible.
template <int MODE>
__device__ void bn_partials(const float* x, const float* dy, const float* mean, double* partial,
                            int N, int C, int HW, int chunk) {
    __shared__ double sa[256];
    __shared__ double sb[256];
    const int c = blockIdx.x, s = blockIdx.y, slices = gridDim.y;
    const int total = N * HW;
    const int j0 = s * chunk, j1 = min(total, j0 + chunk);
    const double center = MODE == 0 ? (double)x[c * HW] : (double)mean[c] + (double)mean[C + c];
    double a = 0.0, b = 0.0;
    for (int j = j0 + threadIdx.x; j < j1; j += blockDim.x) {
        const int n = j / HW, hw = j - n * HW;
        const long long at = ((long long)n * C + c) * HW + hw;
        const double d = (double)x[at] - center;
        if (MODE == 0) { a += d; b = fma(d, d, b); }
        else { const double g = (double)dy[at]; a += g; b = fma(g, d, b); }
    }
    sa[threadIdx.x] = a;
    sb[threadIdx.x] = b;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            sa[threadIdx.x] += sa[threadIdx.x + offset];
            sb[threadIdx.x] += sb[threadIdx.x + offset];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        partial[(c * slices + s) * 2] = sa[0];
        partial[(c * slices + s) * 2 + 1] = sb[0];
    }
}

extern "C" __global__ void mf_bn_stats_partials(const float* x, double* partial, int N, int C, int HW, int chunk) {
    bn_partials<0>(x, nullptr, nullptr, partial, N, C, HW, chunk);
}

extern "C" __global__ void mf_bn_grad_partials(const float* x, const float* dy, const float* mean, double* partial,
                                               int N, int C, int HW, int chunk) {
    bn_partials<1>(x, dy, mean, partial, N, C, HW, chunk);
}

// Per channel: batch mean / biased variance, inverse std and the running-stat update. The mean is
// stored as the unevaluated sum mean[c] + mean[C + c] of two floats (about 48 significant bits), and
// the elementwise kernels form x - mean as (x - hi) - lo, which is exact for x near the mean
// (Sterbenz). A single rounded float mean would cost up to |mean| / std ulp in x_hat.
extern "C" __global__ void mf_bn_stats_finalize(const double* partial, const float* x, float* mean, float* inv_std,
                                                float* running_mean, float* running_var,
                                                int C, int HW, int slices, double count, double momentum, double eps) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    double sum = 0.0, sq = 0.0;
    for (int s = 0; s < slices; ++s) {
        sum += partial[(c * slices + s) * 2];
        sq += partial[(c * slices + s) * 2 + 1];
    }
    const double shifted_mean = sum / count;
    const double var = fmax(sq / count - shifted_mean * shifted_mean, 0.0);
    const double mu = (double)x[c * HW] + shifted_mean;
    const float hi = (float)mu;
    mean[c] = hi;
    mean[C + c] = (float)(mu - (double)hi);
    inv_std[c] = (float)(1.0 / sqrt(var + eps));
    if (running_mean) {
        running_mean[c] = (float)((1.0 - momentum) * (double)running_mean[c] + momentum * mu);
        running_var[c] = (float)((1.0 - momentum) * (double)running_var[c] + momentum * var);
    }
}

// Per channel: accumulate dgamma/dbeta and the coefficients of dx = a * (dy - k1 - (x - mean) * k2).
extern "C" __global__ void mf_bn_grad_finalize(const double* partial, const float* gamma, const float* inv_std,
                                               float* gamma_grad, float* beta_grad, float* coef,
                                               int C, int slices, double count, int training) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    double sdy = 0.0, sdyx = 0.0;
    for (int s = 0; s < slices; ++s) {
        sdy += partial[(c * slices + s) * 2];
        sdyx += partial[(c * slices + s) * 2 + 1];
    }
    const double istd = (double)inv_std[c];
    const double dgamma = istd * sdyx;
    gamma_grad[c] += (float)dgamma;
    beta_grad[c] += (float)sdy;
    coef[c] = (float)((double)gamma[c] * istd);
    coef[C + c] = training ? (float)(sdy / count) : 0.f;
    coef[2 * C + c] = training ? (float)(istd * dgamma / count) : 0.f;
}
"""


@cache
def _bn_module():
    return _lazy_cupy().RawModule(code=_BN_SOURCE, options=("--std=c++17",))


@cache
def _bn_normalize_kernel():
    return _lazy_cupy().ElementwiseKernel(
        "float32 x, float32 hi, float32 lo, float32 inv_std, float32 gamma, float32 beta", "float32 y",
        "y = fmaf(((x - hi) - lo) * inv_std, gamma, beta)", "mf_bn_normalize_hilo")


@cache
def _bn_dx_kernel():
    return _lazy_cupy().ElementwiseKernel(
        "float32 x, float32 dy, float32 hi, float32 lo, float32 a, float32 k1, float32 k2", "float32 dx",
        "dx += a * (dy - k1 - ((x - hi) - lo) * k2)", "mf_bn_dx_acc_hilo")


@cache
def _sm_count(device_id):
    return int(_lazy_cupy().cuda.Device(device_id).attributes["MultiProcessorCount"])


def _bn_grid(n, c, hw):
    # ~8 blocks per SM in total, each slice at least 4096 elements long.
    total = n * hw
    target = 8 * _sm_count(_lazy_cupy().cuda.runtime.getDevice())
    slices = max(1, min(-(-total // 4096), -(-target // max(1, c))))
    return slices, -(-total // slices)


def _channel(array):
    return array.reshape(1, -1, 1, 1)


def batchnorm2d_forward(x, gamma, beta, running_mean, running_var, *, momentum, eps, training):
    """Return ``(y, mean, inv_std)``; updates the running statistics in place when training.

    ``mean`` has shape (2, C): the FP64 mean split into ``mean[0] + mean[1]`` (see
    ``mf_bn_stats_finalize``). Pass it unchanged to :func:`batchnorm2d_backward`.
    """
    n, c, h, w = x.shape
    hw = h * w
    if _on_gpu(x, gamma, beta, running_mean, running_var) and _same_float32(x, gamma, beta, running_mean, running_var):
        cp = _lazy_cupy()
        x = cp.ascontiguousarray(x)
        if n * c * hw > 2**31 - 1:
            raise ValueError("fused BatchNorm indexes with int32; use a smaller batch")
        if training:
            slices, chunk = _bn_grid(n, c, hw)
            partial = cp.empty((c * slices * 2,), dtype=cp.float64)
            mean = cp.empty((2, c), dtype=cp.float32)
            inv_std = cp.empty((c,), dtype=cp.float32)
            module = _bn_module()
            module.get_function("mf_bn_stats_partials")((c, slices), (256,), (x, partial, np.int32(n), np.int32(c), np.int32(hw), np.int32(chunk)))
            module.get_function("mf_bn_stats_finalize")((-(-c // 128),), (128,), (
                partial, x, mean, inv_std, running_mean, running_var, np.int32(c), np.int32(hw),
                np.int32(slices), np.float64(n * hw), np.float64(momentum), np.float64(eps)))
        else:
            mean = cp.stack([running_mean, cp.zeros_like(running_mean)])
            inv_std = (1.0 / cp.sqrt(running_var.astype(cp.float64) + eps)).astype(cp.float32)
        y = _bn_normalize_kernel()(x, _channel(mean[0]), _channel(mean[1]), _channel(inv_std),
                                   _channel(gamma), _channel(beta))
        return y, mean, inv_std

    axes = (0, 2, 3)
    module = np if isinstance(x, np.ndarray) else _lazy_cupy()
    if training:
        mean64 = x.mean(axis=axes, dtype=np.float64)
        centered = x.astype(np.float64) - _channel(mean64)
        var64 = (centered * centered).mean(axis=axes)
        running_mean[...] = (1.0 - momentum) * running_mean.astype(np.float64) + momentum * mean64
        running_var[...] = (1.0 - momentum) * running_var.astype(np.float64) + momentum * var64
    else:
        mean64, var64 = running_mean.astype(np.float64), running_var.astype(np.float64)
    hi = mean64.astype(x.dtype)
    mean = module.stack([hi, (mean64 - hi).astype(x.dtype)])
    inv_std = (1.0 / module.sqrt(var64 + eps)).astype(x.dtype)
    y = (((x - _channel(mean[0])) - _channel(mean[1])) * _channel(inv_std)) * _channel(gamma) + _channel(beta)
    return y, mean, inv_std


def batchnorm2d_backward(x, dy, gamma, mean, inv_std, *, training, x_grad, gamma_grad, beta_grad):
    """Accumulate the BatchNorm2d gradients into ``x_grad``, ``gamma_grad`` and ``beta_grad``."""
    n, c, h, w = x.shape
    hw = h * w
    count = n * hw
    if (_on_gpu(x, dy, gamma, mean, inv_std, x_grad, gamma_grad, beta_grad)
            and _same_float32(x, dy, gamma, mean, inv_std, x_grad, gamma_grad, beta_grad)):
        cp = _lazy_cupy()
        x, dy = cp.ascontiguousarray(x), cp.ascontiguousarray(dy)
        slices, chunk = _bn_grid(n, c, hw)
        partial = cp.empty((c * slices * 2,), dtype=cp.float64)
        coef = cp.empty((3 * c,), dtype=cp.float32)
        module = _bn_module()
        module.get_function("mf_bn_grad_partials")((c, slices), (256,), (
            x, dy, cp.ascontiguousarray(mean), partial, np.int32(n), np.int32(c), np.int32(hw), np.int32(chunk)))
        module.get_function("mf_bn_grad_finalize")((-(-c // 128),), (128,), (
            partial, gamma, inv_std, gamma_grad, beta_grad, coef, np.int32(c), np.int32(slices),
            np.float64(count), np.int32(1 if training else 0)))
        a, k1, k2 = (_channel(part) for part in coef.reshape(3, c))
        _bn_dx_kernel()(x, dy, _channel(mean[0]), _channel(mean[1]), a, k1, k2, x_grad)
        return

    axes = (0, 2, 3)
    centered = x.astype(np.float64) - _channel(mean[0].astype(np.float64) + mean[1].astype(np.float64))
    dy64 = dy.astype(np.float64)
    sdy = dy64.sum(axis=axes)
    istd = inv_std.astype(np.float64)
    dgamma = istd * (dy64 * centered).sum(axis=axes)
    gamma_grad += dgamma.astype(gamma_grad.dtype)
    beta_grad += sdy.astype(beta_grad.dtype)
    a = _channel((gamma.astype(np.float64) * istd).astype(x.dtype))
    if training:
        k1 = _channel((sdy / count).astype(x.dtype))
        k2 = _channel((istd * dgamma / count).astype(x.dtype))
        x_grad += a * (dy - k1 - ((x - _channel(mean[0])) - _channel(mean[1])) * k2)
    else:
        x_grad += a * dy


# ---------------------------------------------------------------------------------------------
# Softmax cross-entropy: FP64 log-sum-exp per row.
# ---------------------------------------------------------------------------------------------
_CE_SOURCE = r"""
// One warp per row. probs = exp(x - logsumexp(x)) rounded once to float; loss_row is
// -sum(target * log p) (or -log p[label]) in FP64. Lane partials combine with a fixed shuffle tree.
extern "C" __global__ void mf_softmax_xent(const float* logits, const long long* labels, const float* targets,
                                           int has_targets, float* probs, double* row_loss, int N, int C) {
    const int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    if (row >= N) return;
    const float* x = logits + (long long)row * C;
    float m = __int_as_float(0xff800000);  // -inf (NVRTC has no INFINITY macro)
    for (int j = lane; j < C; j += 32) m = fmaxf(m, x[j]);
    for (int offset = 16; offset > 0; offset >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, offset));
    double sum = 0.0;
    for (int j = lane; j < C; j += 32) sum += exp((double)x[j] - (double)m);
    for (int offset = 16; offset > 0; offset >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, offset);
    const double lse = (double)m + log(sum);
    double loss = 0.0;
    for (int j = lane; j < C; j += 32) {
        const double logp = (double)x[j] - lse;
        probs[(long long)row * C + j] = (float)exp(logp);
        if (has_targets) loss -= (double)targets[(long long)row * C + j] * logp;
    }
    for (int offset = 16; offset > 0; offset >>= 1) loss += __shfl_xor_sync(0xffffffffu, loss, offset);
    if (lane == 0) {
        // An out-of-range label yields a NaN loss instead of a host-side check (which would sync).
        const long long label = has_targets ? 0 : labels[row];
        row_loss[row] = has_targets ? loss
            : (label < 0 || label >= C) ? __longlong_as_double(0x7ff8000000000000ll) : lse - (double)x[label];
    }
}
"""


@cache
def _ce_kernel():
    return _lazy_cupy().RawKernel(_CE_SOURCE, "mf_softmax_xent", options=("--std=c++17",))


def softmax_cross_entropy(logits, target):
    """Return ``(loss, probs, labels, targets)`` for ``logits`` of shape (N, C).

    ``target`` holds class indices (shape (N,) or (N, 1)) or per-class targets (N, C). Exactly one
    of ``labels``/``targets`` is returned (the other is None) for use by the backward pass.
    """
    n, c = logits.shape
    labels = targets = None
    if target.ndim == 1 or (target.ndim == 2 and target.shape[1] == 1 and c != 1):
        labels = target.reshape(-1).astype(np.int64)
        if labels.shape[0] != n:
            raise ValueError("labels must provide one class index per row")
    else:
        targets = target.astype(logits.dtype, copy=False)
        if targets.shape != logits.shape:
            raise ValueError("targets must have the shape of logits")
    if _on_gpu(logits, target) and logits.dtype == np.float32:
        cp = _lazy_cupy()
        logits = cp.ascontiguousarray(logits)
        if labels is not None:
            labels = cp.ascontiguousarray(labels)
        else:
            targets = cp.ascontiguousarray(targets)
        probs = cp.empty_like(logits)
        row_loss = cp.empty((n,), dtype=cp.float64)
        # RawKernel has no null pointers: the unused operand is a one-element placeholder.
        label_arg = labels if labels is not None else cp.zeros((1,), dtype=cp.int64)
        target_arg = targets if targets is not None else cp.zeros((1,), dtype=cp.float32)
        _ce_kernel()((-(-n // 8),), (256,), (logits, label_arg, target_arg, np.int32(targets is not None),
                                            probs, row_loss, np.int32(n), np.int32(c)))
        return row_loss.mean().astype(logits.dtype), probs, labels, targets

    module = np if isinstance(logits, np.ndarray) else _lazy_cupy()
    x = logits.astype(np.float64)
    shift = x - x.max(axis=1, keepdims=True)
    logp = shift - module.log(module.exp(shift).sum(axis=1, keepdims=True))
    if labels is not None:
        if bool(((labels < 0) | (labels >= c)).any()):
            raise ValueError("class index out of range")
        row_loss = -logp[module.arange(n), labels]
    else:
        row_loss = -(targets.astype(np.float64) * logp).sum(axis=1)
    loss = module.asarray(row_loss.mean(), dtype=logits.dtype)
    return loss, module.exp(logp).astype(logits.dtype), labels, targets


def softmax_cross_entropy_grad(probs, labels, targets):
    """d(mean loss)/d logits = (probs - target) / N."""
    n = probs.shape[0]
    module = np if isinstance(probs, np.ndarray) else _lazy_cupy()
    if labels is not None:
        grad = probs.copy()
        # Clipped so that an invalid label (already reported as a NaN loss) cannot write out of bounds.
        grad[module.arange(n), module.clip(labels, 0, probs.shape[1] - 1)] -= 1
    else:
        grad = probs - targets
    grad /= n
    return grad


# ---------------------------------------------------------------------------------------------
# Adam
# ---------------------------------------------------------------------------------------------
@cache
def _adam_kernel():
    # Each operation rounds explicitly, in the order of the NumPy fallback below.
    return _lazy_cupy().ElementwiseKernel(
        "float32 g, float32 scale, float32 b1, float32 omb1, float32 b2, float32 omb2, "
        "float32 bc1, float32 bc2, float32 lr, float32 eps",
        "float32 p, float32 m, float32 v",
        """
        const float gs = scale == 1.f ? g : __fmul_rn(g, scale);
        m = __fadd_rn(__fmul_rn(b1, m), __fmul_rn(omb1, gs));
        v = __fadd_rn(__fmul_rn(b2, v), __fmul_rn(omb2, __fmul_rn(gs, gs)));
        const float denom = __fadd_rn(__fsqrt_rn(__fdiv_rn(v, bc2)), eps);
        p = __fsub_rn(p, __fdiv_rn(__fmul_rn(lr, __fdiv_rn(m, bc1)), denom));
        """,
        "mf_adam_step")


def adam_update(param, grad, m, v, *, lr, beta1, beta2, eps, step, grad_scale=1.0):
    """One textbook Adam step (zero-initialized moments, bias-corrected) applied in place."""
    dtype = param.dtype.type
    bc1, bc2 = 1.0 - beta1 ** step, 1.0 - beta2 ** step
    scalars = [dtype(value) for value in (grad_scale, beta1, 1.0 - beta1, beta2, 1.0 - beta2, bc1, bc2, lr, eps)]
    if _on_gpu(param, grad, m, v) and _same_float32(param, grad, m, v):
        _adam_kernel()(grad, *scalars, param, m, v)
        return
    scale, b1, omb1, b2, omb2, bc1, bc2, lr, eps = scalars
    gs = grad if scale == 1 else grad * scale
    m *= b1
    m += omb1 * gs
    v *= b2
    v += omb2 * (gs * gs)
    denom = _sqrt(v / bc2)
    denom += eps
    param -= (lr * (m / bc1)) / denom


def _sqrt(array):
    return np.sqrt(array) if isinstance(array, np.ndarray) else _lazy_cupy().sqrt(array)


# ---------------------------------------------------------------------------------------------
# Norms
# ---------------------------------------------------------------------------------------------
@cache
def _sumsq_kernel():
    return _lazy_cupy().ReductionKernel(
        "T x", "float64 y", "(double)x * (double)x", "a + b", "y = a", "0", "mf_sumsq_f64")


def global_norm(arrays):
    """Euclidean norm of all arrays together, accumulated in FP64, as a Python float."""
    arrays = [a for a in arrays if a is not None and a.size]
    if not arrays:
        return 0.0
    if _on_gpu(*arrays):
        cp = _lazy_cupy()
        return float(cp.sqrt(cp.stack([_sumsq_kernel()(a) for a in arrays]).sum()))
    total = 0.0
    for a in arrays:
        flat = np.asarray(a, dtype=np.float64).reshape(-1)
        total += float(np.dot(flat, flat))
    return float(np.sqrt(total))
