import numpy as np
from .tensor import Tensor

class Node:
    def __init__(self, *parents, trainable=False, name=None):
        self.parents = list(parents)
        self.children = []
        for p in parents:
            p.children.append(self)
        self._value = None
        self._grad = None
        self.trainable = trainable
        self.name = name if name else self.__class__.__name__

    @property
    def value(self):
        return None if self._value is None else self._value.data

    @value.setter
    def value(self, value):
        self._value = None if value is None else Tensor.ensure(value)

    @property
    def tensor(self):
        return self._value

    @tensor.setter
    def tensor(self, value):
        self._value = None if value is None else Tensor.ensure(value)

    @property
    def grad(self):
        return None if self._grad is None else self._grad.data

    @grad.setter
    def grad(self, grad):
        self._grad = None if grad is None else Tensor.ensure(grad)

    @property
    def grad_tensor(self):
        return self._grad

    @grad_tensor.setter
    def grad_tensor(self, grad):
        self._grad = None if grad is None else Tensor.ensure(grad)

    def forward(self, *inputs):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError

    def clear_grad(self):
        # 完成前向传播，确定梯度形状
        if self._value is not None:
            # 将梯度初始化，与value相关
            self._grad = Tensor.zeros_like(self._value)
        else:
            # 如果还没有计算 value 保持 None
            self._grad = None


class Variable(Node):
    def __init__(self, value=None, trainable=False, name=None, constant=False):
        super().__init__(trainable=trainable, name=name)
        self.value = value
        #: 若为 True，表示运行期不参与训练更新，且可被图优化折叠（常量折叠）
        self.constant = bool(constant)

    def forward(self): 
        pass

    def backward(self):
        pass
