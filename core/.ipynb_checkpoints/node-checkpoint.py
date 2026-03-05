import numpy as np
class Variable:
        def __init__(self, dim, init=True, trainable=True):
            self.dim = dim
            self.trainable = trainable
            self.init = init
            self.value = None
            self.grad = None
            
            # 初始化变量值
            if self.init:
                # 使用小的随机数初始化权重
                self.value = np.random.randn(*dim) * 0.01
            
            # 初始化梯度
            self.grad = np.zeros(dim)
        
        def set_value(self, value):
            self.value = value
        
        def forward(self):
            # Variable的forward操作就是保持其值不变
            pass
        
        def backward(self):
            # Variable的backward操作就是保持其梯度不变
            pass
        
        def update(self, learning_rate):
            if self.trainable:
                self.value -= learning_rate * self.grad
                # 清空梯度
                self.grad = np.zeros(self.dim)