import numpy as np
from ..core.node import Node
from .activation import Softmax

class PerceptionLoss(Node):
    """
    两个父节点：y_pred (预测) 和 y_true (标签)
    """
    def __init__(self, y_pred, y_true, name="PerceptionLoss"):
        super().__init__(y_pred, y_true, name=name)

    def forward(self, y_pred_val, y_true_val):
        # 存储值以便反向传播
        self.y_pred_val = y_pred_val
        self.y_true_val = y_true_val
        
       
        losses = np.where(self.y_true_val * self.y_pred_val < 0, -self.y_true_val * self.y_pred_val, 0)
        
        # 返回平均损失
        self.value = np.mean(losses)

    def backward(self):
        # 损失对 y_pred 的梯度
        grad_signal = np.where(self.y_true_val * self.y_pred_val < 0, -self.y_true_val, 0)
        
        # 梯度平均
        y_pred.grad += self.grad * (grad_signal / len(self.y_true_val))


class LogLoss(Node):
    def __init__(self, y_pred, y_true, name="LogLoss"):
        # 保持与基类一致
        super().__init__(y_pred, y_true, name=name)
        self.y_pred = y_pred
        self.y_true = y_true

    def forward(self, y_pred, y_true):
        # 前向传播公式说明 x = y_true * y_pred
        self.x = self.y_true.value * self.y_pred.value
        
       
        res = np.zeros_like(self.x)
        mask = self.x > 0
        res[mask] = np.log1p(np.exp(-self.x[mask]))
        res[~mask] = -self.x[~mask] + np.log1p(np.exp(self.x[~mask]))
        
        self.value = np.mean(res)

    def backward(self):
        N = len(self.y_true.value)
        
        # 梯度公式说明: grad = -y_true / (1 + e^x)
       
        exp_x = np.exp(np.clip(self.x, -500, 500)) 
        grad_x = -1.0 / (1.0 + exp_x)
        
        # 最终梯度
        final_grad = (self.grad * grad_x * self.y_true.value) / N
        
        self.y_pred.grad += final_grad


class CrossEntropy(Node):
    def __init__(self, logits, y_true, name="CrossEntropy"):
        # 这里的 logits 是前层(如Add)的输出，y_true 是标签变量
        super().__init__(logits, y_true, name=name)
        self.logits = logits
        self.y_true = y_true

    def forward(self, logits_val, y_true_val):
        # 借用 Softmax 的静态方法计算概率
        self.probs = Softmax.softmax(logits_val)
        
        # 处理标签维度 (兼容整数索引或 One-hot)
        if y_true_val.ndim == 1 or y_true_val.shape[1] == 1:
            num_classes = logits_val.shape[1]
            self.y_true_final = np.eye(num_classes)[y_true_val.reshape(-1).astype(int)]
        else:
            self.y_true_final = y_true_val

        # 计算 Loss 值
        self.value = -np.mean(np.sum(self.y_true_final * np.log(self.probs + 1e-10), axis=-1))

    def backward(self):
        #计算 Loss 对 Logits 的梯度：(P - Y) / N
        N = self.logits.value.shape[0]
        
        # 这里的 self.grad 由Graph.backward 传来，为 1.0
        # 计算合成梯度
        final_grad = (self.probs - self.y_true_final) / N
        
        # 将梯度回传给父节点 (logits)
        if self.logits.grad is None:
            self.logits.clear_grad() #
        self.logits.grad += self.grad * final_grad

        