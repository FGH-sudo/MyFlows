import numpy as np
from ..core.node import Node

class Logistic(Node):
    def forward(self, x):
        safe_x = np.clip(x, -500, 500) 
        self.value = 1.0 / (1.0 + np.exp(-safe_x))

    def backward(self):
        x = self.parents[0]
        if x.grad is None:
            x.grad = np.zeros_like(x.value)
        x.grad += self.grad * self.value * (1 - self.value)


class Softmax(Node):
    @staticmethod
    def softmax(a):
        # 采用平移法保证数值稳定
        shift_a = a - np.max(a, axis=-1, keepdims=True)
        exps = np.exp(shift_a)
        return exps / np.sum(exps, axis=-1, keepdims=True)

    def forward(self, x):
        # 节点的前向传播直接调用静态方法
        self.value = Softmax.softmax(x)

    def backward(self):
        # 训练时应使用 CrossEntropy 节点
        raise NotImplementedError("训练时请使用 CrossEntropy 节点，不要直接对 Softmax 节点执行 backward")


class Tanh(Node):
    def forward(self, x):
        # 使用 np.tanh 提高数值稳定性
        self.value = np.tanh(x)

    def backward(self):
        x_node = self.parents[0]
        if x_node.grad is None:
            x_node.grad = np.zeros_like(x_node.value)
        
        # tanh的导数公式说明：1 - tanh(x)^2
        
        x_node.grad += self.grad * (1 - self.value**2)


class ReLU(Node):
    def forward(self, x):
        # ReLU函数说明: max(0, x)
        self.value = np.maximum(0, x)

    def backward(self):
        x_node = self.parents[0]
        if x_node.grad is None:
            x_node.grad = np.zeros_like(x_node.value)
        
        # ReLU 的导数说明：x > 0 时为 1，否则为 0
        grad_mask = (x_node.value > 0).astype(float)
        x_node.grad += self.grad * grad_mask


class LeakyReLU(Node):
    def __init__(self, x, alpha=0.01, name=None):
        self.alpha = alpha  # 负半轴的斜率，通常设为 0.01
        super().__init__(x, name=name)

    def forward(self, x):
        # LeakyReLU的函数说明: x if x > 0 else alpha * x
        self.value = np.where(x > 0, x, self.alpha * x)

    def backward(self):
        x_node = self.parents[0]
        if x_node.grad is None:
            x_node.grad = np.zeros_like(x_node.value)
        
        # LeakyReLU 的导数说明：x > 0 时为 1，其余则取值为 alpha
        grad_mask = np.where(x_node.value > 0, 1.0, self.alpha)
        x_node.grad += self.grad * grad_mask