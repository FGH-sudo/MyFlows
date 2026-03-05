import numpy as np
class loss:
    """Loss functions collection"""
    
    class PerceptionLoss:
        """Perceptron loss function"""
        def __init__(self, y_pred):
            self.y_pred = y_pred
            self.value = None
            self.grad = None
        
        def forward(self):
            # 先调用依赖操作的forward方法
            self.y_pred.forward()
            if self.y_pred.value is None:
                raise ValueError("PerceptionLoss的输入值为None，请先设置输入变量的值")
            
            # 感知器损失函数
            # 对于正确分类的样本，损失为0；对于错误分类的样本，损失为-y_pred
            self.value = np.where(self.y_pred.value < 0, -self.y_pred.value, 0)
        
        def backward(self):
            # 如果grad还没有被初始化，就先初始化成全0矩阵
            if self.y_pred.grad is None:
                self.y_pred.grad = np.zeros_like(self.y_pred.value)
        
            # 感知器损失函数的梯度
            self.y_pred.grad += np.where(self.y_pred.value < 0, -1, 0)