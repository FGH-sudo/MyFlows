import numpy as np

class Node:
    def __init__(self, *parents, trainable=False, name=None):
        self.parents = list(parents)
        self.children = []
        for p in parents:
            p.children.append(self)
        self.value = None
        self.grad = None
        self.trainable = trainable
        self.name = name if name else self.__class__.__name__

    def forward(self, *inputs):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError

    def clear_grad(self):
        # 完成前向传播，确定梯度形状
        if self.value is not None:
            # 将梯度初始化，与value相关
            self.grad = np.zeros_like(self.value)
        else:
            # 如果还没有计算 value 保持 None
            self.grad = None


class Variable(Node):
    def __init__(self, value=None, trainable=False, name=None):
        super().__init__(trainable=trainable, name=name)
        self.value = value

    def forward(self): 
        pass

    def backward(self):
        pass