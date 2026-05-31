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
            
        a.grad += xp.dot(self.grad, b.value.T)
        b.grad += xp.dot(a.value.T, self.grad)


class Linear(Node):
    """融合算子：y = x @ W + b（减少 MatMul+Add 的节点与调度开销）。"""

    def forward(self, x, w, b):
        self.x_shape = x.shape
        self.w_shape = w.shape
        self.b_shape = b.shape
        self.value = xp.dot(x, w) + b

    def backward(self):
        x, w, b = self.parents
        x.grad += xp.dot(self.grad, w.value.T)
        w.grad += xp.dot(x.value.T, self.grad)

        # bias 的梯度：将上游梯度按被广播维度求和，恢复成原始 b 形状
        grad = self.grad
        # 尽量对齐到 b 的 ndim，处理 (out,) / (1,out) 这类常见 bias
        while grad.ndim > b.value.ndim:
            grad = xp.sum(grad, axis=0)
        if grad.shape != b.value.shape:
            axes = tuple(i for i, (gd, bd) in enumerate(zip(grad.shape, b.value.shape)) if bd == 1 and gd != 1)
            if axes:
                grad = xp.sum(grad, axis=axes, keepdims=True)
        b.grad += grad.reshape(b.value.shape)

