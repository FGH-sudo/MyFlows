from ..core.device import xp
from ..core.node import GradientOp
from . import precision

class Add(GradientOp):
    def forward(self, a, b):
        self.a_shape = a.shape
        self.b_shape = b.shape

        if a.ndim == 4 and b.ndim == 1:
            self.a_broadcast_shape = a.shape
            self.b_broadcast_shape = (1, b.shape[0], 1, 1)
            self.value = a + b.reshape(self.b_broadcast_shape)
        else:
            self.a_broadcast_shape = (1,) * (max(a.ndim, b.ndim) - a.ndim) + a.shape
            self.b_broadcast_shape = (1,) * (max(a.ndim, b.ndim) - b.ndim) + b.shape
            self.value = a + b

    def backward(self):
        a, b = self.parents
        
        # 处理 a 的梯度
        if a.requires_grad:
            a.accumulate_grad(self._handle_broadcast(self.grad, self.a_broadcast_shape, self.a_shape))
        # 处理 b 的梯度
        if b.requires_grad:
            b.accumulate_grad(self._handle_broadcast(self.grad, self.b_broadcast_shape, self.b_shape))

    def _handle_broadcast(self, grad, broadcast_shape, original_shape):
        # 将上游梯度沿着被广播的维度求和，再恢复成原始张量形状
        if grad.shape != broadcast_shape:
            axes = tuple(
                axis
                for axis, (grad_dim, shape_dim) in enumerate(zip(grad.shape, broadcast_shape))
                if shape_dim == 1 and grad_dim != 1
            )
            if axes:
                grad = xp.sum(grad, axis=axes, keepdims=True)
        return grad.reshape(original_shape)


class MatMul(GradientOp):
    def forward(self, a, b):
        self.value = xp.dot(a, b)

    def backward(self):
        a, b = self.parents
        # 不需要梯度的输入（如输入数据）跳过对应的 GEMM
        if getattr(a, "requires_grad", True):
            a.accumulate_grad(_dot_grad_a(a.value, b.value, self.grad), owned=True)
        if getattr(b, "requires_grad", True):
            b.accumulate_grad(_dot_grad_b(a.value, b.value, self.grad), owned=True)


LINEAR_BACKENDS = ("auto", "numpy", "cupy", "cuda_native_register", "cuda_native_cublas",
                   "cuda_native_kahan", "cuda_native_dot2", *precision.BACKENDS)


def _linear_backend(backend):
    if backend == "winograd":
        return "auto"  # This model-level choice applies to 3x3 convolutions only.
    if backend not in LINEAR_BACKENDS:
        raise ValueError(f"backend must be one of {', '.join(LINEAR_BACKENDS)}")
    return backend


def _reduce_to(grad, shape):
    """把上游梯度沿被广播的维度求和到 ``shape``；FP64 累加后舍入一次，顺序固定。"""
    lead = grad.ndim - len(shape)
    axes = tuple(range(lead)) + tuple(
        lead + i for i, (gd, bd) in enumerate(zip(grad.shape[lead:], shape)) if bd == 1 and gd != 1)
    total = xp.sum(grad, axis=axes, dtype=xp.float64) if axes else grad
    return total.reshape(shape).astype(grad.dtype, copy=False)


def _dot_grad_a(a, b, grad):
    if a.ndim == 0 or b.ndim == 0:
        return _reduce_to(grad * b, a.shape)
    if b.ndim == 1:
        return xp.expand_dims(grad, -1) * b
    if a.ndim == b.ndim == 2:
        return xp.dot(grad, b.T)
    return xp.tensordot(grad, b, axes=(tuple(range(a.ndim - 1, grad.ndim)),
                                      (*range(b.ndim - 2), b.ndim - 1)))


def _dot_grad_b(a, b, grad):
    if a.ndim == 0 or b.ndim == 0:
        return _reduce_to(grad * a, b.shape)
    if a.ndim == b.ndim == 2:
        return xp.dot(a.T, grad)
    axes = tuple(range(a.ndim - 1))
    result = xp.tensordot(a, grad, axes=(axes, axes))
    return result if b.ndim == 1 else xp.moveaxis(result, 0, -2)


class Linear(GradientOp):
    """融合算子：y = x @ W + b（减少 MatMul+Add 的节点与调度开销）。

    backend：
      - auto / numpy / cupy：``xp.dot`` 加广播 bias（CuPy 上即 cuBLAS）；
      - cuda_native_register / cuda_native_cublas：原生 DLL 的 GEMM 引擎（第 19 讲寄存器分块 kernel 或
        cuBLAS）与 bias epilogue，前向一次调用，反向的两个 GEMM 直接读转置的操作数；
      - cuda_native_kahan / cuda_native_dot2（实验用）：寄存器分块 kernel 的补偿累加（结果与 shared-memory
        kernel 逐位相同），前向与反向的三个 GEMM 都用同一种累加（反向先把转置的操作数拷成连续数组），
        结果最接近 FP64。
    bias 梯度在 FP64 中累加；不需要梯度的输入 x 跳过 dX 的 GEMM。
    """

    def __init__(self, x, w, b, name=None, backend="auto", bias_first=None):
        super().__init__(x, w, b, name=name)
        self.backend = _linear_backend(backend)
        # A fused Add must retain its NCHW channel-bias rule and operand order.
        self.bias_first = bias_first

    def _native_spec(self):
        if self.backend == "auto" or self.backend in ("numpy", "cupy"):
            return None
        from .convolution import _resolve_backend

        _resolve_backend(self.backend, *(p.value for p in self.parents))
        return self.backend.rsplit("_", 1)[1]  # register / cublas / kahan / dot2

    def forward(self, x, w, b):
        self.x_shape = x.shape
        self.w_shape = w.shape
        self.b_shape = b.shape
        bias_done = False
        if self.backend in precision.BACKENDS:
            rows = x.reshape(-1, x.shape[-1])
            packed_x = precision.pack(rows, self.backend)
            packed_w = precision.pack(w, self.backend)
            self._packed_x = packed_x if self.parents[1].requires_grad else None
            self._packed_w = packed_w if self.parents[0].requires_grad else None
            shape = (*x.shape[:-1], w.shape[-1])
            out = precision.gemm(packed_x, packed_w, self.backend).reshape(shape)
        else:
            spec = self._native_spec()
            if spec is None or x.ndim != 2:
                out = xp.dot(x, w)
            else:
                from .cuda_native import native

                bias = b if b.shape in ((w.shape[1],), (1, w.shape[1])) else None
                if spec in native.GEMM_ENGINES:
                    out = native.gemm(x, w, bias, engine=spec)
                else:
                    out = native.matmul(x, w, bias, variant="register", accumulate=spec)
                bias_done = bias is not None
        self._dot_shape = self._dot_broadcast_shape = out.shape
        self._bias_broadcast_shape = b.shape
        if self.bias_first is False and out.ndim == 4 and b.ndim == 1:
            self._bias_broadcast_shape = (1, b.shape[0], 1, 1)
        elif self.bias_first is True and b.ndim == 4 and out.ndim == 1:
            self._dot_broadcast_shape = (1, out.shape[0], 1, 1)
        self.value = out if bias_done else out.reshape(self._dot_broadcast_shape) + b.reshape(self._bias_broadcast_shape)

    def _matmul(self, a, b, trans_a=False, trans_b=False):
        # op(a) @ op(b)，op 为可选的转置
        spec = self._native_spec()
        if spec is None:
            return xp.dot(a.T if trans_a else a, b.T if trans_b else b)
        from .cuda_native import native

        if spec in native.GEMM_ENGINES:
            return native.gemm(a, b, trans_a=trans_a, trans_b=trans_b, engine=spec)
        return native.matmul(xp.ascontiguousarray(a.T if trans_a else a),
                             xp.ascontiguousarray(b.T if trans_b else b), variant="register", accumulate=spec)

    def backward(self):
        x, w, b = self.parents
        grad = _reduce_to(self.grad, self._dot_broadcast_shape).reshape(self._dot_shape)
        if self.backend in precision.BACKENDS:
            if x.requires_grad or w.requires_grad:
                shape = (*self.x_shape[:-1], self.w_shape[-1])
                dy = precision.pack(grad.reshape(-1, shape[-1]), self.backend)
                if x.requires_grad:
                    x.accumulate_grad(precision.gemm(dy, self._packed_w, self.backend, trans_b=True).reshape(self.x_shape), owned=True)
                if w.requires_grad:
                    w.accumulate_grad(precision.gemm(self._packed_x, dy, self.backend, trans_a=True), owned=True)
            if b.requires_grad:
                b.accumulate_grad(_reduce_to(self.grad, self._bias_broadcast_shape).reshape(self.b_shape))
            return
        if x.value.ndim == w.value.ndim == 2:
            if getattr(x, "requires_grad", True):
                x.accumulate_grad(self._matmul(grad, w.value, trans_b=True), owned=True)
            if w.requires_grad:
                w.accumulate_grad(self._matmul(x.value, grad, trans_a=True), owned=True)
        else:
            if getattr(x, "requires_grad", True):
                x.accumulate_grad(_dot_grad_a(x.value, w.value, grad), owned=True)
            if w.requires_grad:
                w.accumulate_grad(_dot_grad_b(x.value, w.value, grad), owned=True)
        if b.requires_grad:
            b.accumulate_grad(_reduce_to(self.grad, self._bias_broadcast_shape).reshape(self.b_shape))
