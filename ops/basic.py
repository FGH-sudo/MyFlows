from ..core.device import xp
from ..core.node import Node

class Add(Node):
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
        a.grad += self._handle_broadcast(self.grad, self.a_broadcast_shape, self.a_shape)
        # 处理 b 的梯度
        b.grad += self._handle_broadcast(self.grad, self.b_broadcast_shape, self.b_shape)

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


class MatMul(Node):
    def forward(self, a, b):
        self.value = xp.dot(a, b)

    def backward(self):
        a, b = self.parents
        # 不需要梯度的输入（如输入数据）跳过对应的 GEMM
        if getattr(a, "requires_grad", True):
            a.grad += xp.dot(self.grad, b.value.T)
        if getattr(b, "requires_grad", True):
            b.grad += xp.dot(a.value.T, self.grad)


LINEAR_BACKENDS = ("auto", "numpy", "cupy", "cuda_native_register", "cuda_native_cublas",
                   "cuda_native_kahan", "cuda_native_dot2")


def _linear_backend(backend):
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


class Linear(Node):
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

    def __init__(self, x, w, b, name=None, backend="auto"):
        super().__init__(x, w, b, name=name)
        self.backend = _linear_backend(backend)

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
        spec = self._native_spec()
        if spec is None or x.ndim != 2:
            self.value = xp.dot(x, w) + b
            return
        from .cuda_native import native

        bias = b if b.size == w.shape[1] else None
        if spec in native.GEMM_ENGINES:
            out = native.gemm(x, w, bias, engine=spec)
        else:
            out = native.matmul(x, w, bias, variant="register", accumulate=spec)
        self.value = out if bias is not None else out + b

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
        grad = self.grad
        if grad.ndim == 2:
            if getattr(x, "requires_grad", True):
                x.grad += self._matmul(grad, w.value, trans_b=True)
            w.grad += self._matmul(x.value, grad, trans_a=True)
        else:
            if getattr(x, "requires_grad", True):
                x.grad += xp.dot(grad, w.value.T)
            w.grad += xp.tensordot(x.value, grad, axes=(tuple(range(x.value.ndim - 1)),) * 2)
        b.grad += _reduce_to(grad, b.value.shape)
