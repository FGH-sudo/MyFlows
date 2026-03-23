import numpy as np
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
                grad = np.sum(grad, axis=axes, keepdims=True)
        return grad.reshape(original_shape)


class MatMul(Node):
    def forward(self, a, b):
        self.value = np.dot(a, b)

    def backward(self):
        a, b = self.parents
            
        a.grad += np.dot(self.grad, b.value.T)
        b.grad += np.dot(a.value.T, self.grad)

