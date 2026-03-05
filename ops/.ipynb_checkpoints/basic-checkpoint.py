import numpy as np

class Add:
    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.value = None
        self.grad = None
    
    def forward(self):
        # 先调用依赖操作的forward方法
        self.a.forward()
        self.b.forward()
        
        if self.a.value is None or self.b.value is None:
            raise ValueError("Add操作的输入值为None，请先设置输入变量的值")
        self.value = self.a.value + self.b.value

    def get_jacobi(self, parent):
        if parent is self.a:
            return np.eye(self.x.dimension())
        elif parent is self.b:
            return np.eye(self.y.dimension())

    def backward(self):
        # 简单的加法反向传播，梯度直接传递
        if self.a.trainable:
            self.a.grad += self.grad
        if self.b.trainable:
            self.b.grad += self.grad
            
    def backward(self, result):
        # 反向传播，计算结果节点对本节点的雅可比矩阵
        if self.jacobi is None:
            if self is result:
                self.jacobi = np.mat(np.eye(self.dimension()))
            else:
                self.jacobi = np.mat(np.zeros((result.dimension(), self.dimension())))
                for child in self.get_children():
                    if child.value is not None:
                        self.jacobi += child.backward(result) * child.get_jacobi(self)
        return self.jacobi


class MatMul:
    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.value = None
        self.grad = None
    
    def forward(self):
        # 先调用依赖操作的forward方法
        self.a.forward()
        self.b.forward()
        
        if self.a.value is None or self.b.value is None:
            raise ValueError("MatMul操作的输入值为None，请先设置输入变量的值")
        self.value = np.dot(self.a.value, self.b.value)

    def backward(self):
        # 矩阵乘法的反向传播
        if self.a.trainable:
            self.a.grad += np.dot(self.grad, self.b.value.T)
        if self.b.trainable:
            self.b.grad += np.dot(self.a.value.T, self.grad)

class Step:
    def __init__(self, x):
        self.x = x
        self.value = None
        self.grad = None
    
    def forward(self):
        # 先调用依赖操作的forward方法
        self.x.forward()
        
        if self.x.value is None:
            raise ValueError("Step操作的输入值为None，请先设置输入变量的值")
        # Step函数，大于0的为1，否则为0
        self.value = np.where(self.x.value > 0, 1.0, 0.0)
    
    def backward(self):
        # Step函数在大部分点的导数为0，这里先简化处理
        self.x.grad += 0
    
       