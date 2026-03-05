import numpy as np
from ..core.node import Node

class Add(Node):
    def forward(self, a, b):
        if a.ndim == 4 and b.ndim == 1:
            self.value = a + b[None, :, None, None]
        else:
            self.value = a + b

    def backward(self):
        a, b = self.parents
        
        # 处理 a 的梯度
        a.grad += self._handle_broadcast(self.grad, a.value.shape)
        # 处理 b 的梯度
        b.grad += self._handle_broadcast(self.grad, b.value.shape)

    def _handle_broadcast(self, grad, shape):
        # 广播处理逻辑，确保梯度维度对齐
        if grad.shape == shape: return grad
        axes = []
        for i, (g_d, s_d) in enumerate(zip(grad.shape[::-1], shape[::-1])):
            if g_d != s_d: axes.append(len(grad.shape) - 1 - i)
        # 补齐前面缺失的维度
        for i in range(len(grad.shape) - len(shape)):
            axes.append(i)
        return np.sum(grad, axis=tuple(axes)).reshape(shape)


class MatMul(Node):
    def forward(self, a, b):
        self.value = np.dot(a, b)

    def backward(self):
        a, b = self.parents
            
        a.grad += np.dot(self.grad, b.value.T)
        b.grad += np.dot(a.value.T, self.grad)


