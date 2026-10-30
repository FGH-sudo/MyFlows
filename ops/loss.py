import numpy as np

from ..core.device import xp
from ..core.node import Node
from .fused import softmax_cross_entropy, softmax_cross_entropy_grad

class PerceptionLoss(Node):
    """
    两个父节点：y_pred (预测) 和 y_true (标签)
    """
    def __init__(self, y_pred, y_true, name="PerceptionLoss"):
        super().__init__(y_pred, y_true, name=name)
        self.y_pred = y_pred
        self.y_true = y_true

    def forward(self, y_pred_val, y_true_val):
        self.y_pred, self.y_true = self.parents
        # 存储值以便反向传播
        self.y_pred_val = y_pred_val
        self.y_true_val = y_true_val

        losses = xp.where(self.y_true_val * self.y_pred_val < 0, -self.y_true_val * self.y_pred_val, 0)

        # 返回平均损失
        self.value = xp.mean(losses)

    def backward(self):
        # 损失对 y_pred 的梯度
        grad_signal = xp.where(self.y_true_val * self.y_pred_val < 0, -self.y_true_val, 0)
        n = max(int(self.y_true_val.size), 1)
        if self.y_pred.grad is None:
            self.y_pred.clear_grad()
        self.y_pred.grad += self.grad * (grad_signal / n)


class LogLoss(Node):
    def __init__(self, y_pred, y_true, name="LogLoss"):
        # 保持与基类一致
        super().__init__(y_pred, y_true, name=name)
        self.y_pred = y_pred
        self.y_true = y_true

    def forward(self, y_pred, y_true):
        self.y_pred, self.y_true = self.parents
        # 前向传播公式说明 x = y_true * y_pred
        self.x = y_true * y_pred

        # log(1 + e^{-x}) = logaddexp(0, -x)：两侧都数值稳定，且没有布尔索引（CuPy 上会同步）；
        # 均值在 FP64 中累加后转回输入 dtype
        self.value = xp.mean(xp.logaddexp(0, -self.x), dtype=np.float64).astype(self.x.dtype)

    def backward(self):
        # 前向对全部元素求均值，因此按元素总数归一化（多列标签时与 len() 不同）
        N = self.x.size
        
        # 梯度公式说明: grad = -y_true / (1 + e^x)
       
        exp_x = xp.exp(xp.clip(self.x, -500, 500)) 
        grad_x = -1.0 / (1.0 + exp_x)
        
        # 最终梯度
        final_grad = (self.grad * grad_x * self.y_true.value) / N
        
        self.y_pred.grad += final_grad


class CrossEntropy(Node):
    """Softmax 交叉熵。logits 为 (N, C)；标签可为类别下标 (N,)/(N, 1) 或 (N, C) 的 one-hot/软标签。

    按 log-sum-exp 直接求 log softmax（FP64 累加），不再计算 log(softmax + 1e-10)；
    GPU 上 softmax、log-sum-exp 与逐行损失在一个融合 kernel 内完成。
    """

    def __init__(self, logits, y_true, name="CrossEntropy"):
        # 这里的 logits 是前层(如Add)的输出，y_true 是标签变量
        super().__init__(logits, y_true, name=name)
        self.logits = logits
        self.y_true = y_true

    def forward(self, logits_val, y_true_val):
        self.logits, self.y_true = self.parents
        self.value, self.probs, self._labels, self._targets = softmax_cross_entropy(logits_val, y_true_val)

    def backward(self):
        # 计算 Loss 对 Logits 的梯度：(P - Y) / N；self.grad 由 Graph.backward 传来，为 1.0
        final_grad = softmax_cross_entropy_grad(self.probs, self._labels, self._targets)
        if self.logits.grad is None:
            self.logits.clear_grad()
        self.logits.grad += self.grad * final_grad


class MSELoss(Node):
    """均方误差，用于回归(如 user/angle、user/throttle)。"""
    def __init__(self, y_pred, y_true, name="MSELoss"):
        super().__init__(y_pred, y_true, name=name)

    def forward(self, y_pred_val, y_true_val):
        self.y_pred_val = y_pred_val
        self.y_true_val = y_true_val
        # 损失留在设备上（0 维数组，FP64 累加后转回预测的 dtype）：float() 会在前向中途
        # 同步 GPU，让 CPU 等前向跑完才能发射反向 kernel；读取时再转成 Python 标量
        diff = y_pred_val - y_true_val
        self.value = xp.mean(diff * diff, dtype=np.float64).astype(diff.dtype)

    def backward(self):
        y_pred, y_true = self.parents
        n = max(y_pred.value.size, 1)
        g = 2.0 * (y_pred.value - y_true.value) / n
        y_pred.grad += self.grad * g
