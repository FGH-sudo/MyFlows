import math

import numpy as np

from ..core.device import xp
from ..core.node import GradientOp
from .basic import _reduce_to
from .fused import softmax_cross_entropy, softmax_cross_entropy_grad

class PerceptionLoss(GradientOp):
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
        if not self.y_pred.requires_grad:
            return
        grad_signal = xp.where(self.y_true_val * self.y_pred_val < 0, -self.y_true_val, 0)
        n = max(int(grad_signal.size), 1)
        self.y_pred.accumulate_grad(_reduce_to(self.grad * (grad_signal / n), self.y_pred_val.shape), owned=True)


class LogLoss(GradientOp):
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
        if not self.y_pred.requires_grad:
            return
        N = self.x.size
        
        # 梯度公式说明: grad = -y_true / (1 + e^x)
       
        grad_x = -xp.exp(-xp.logaddexp(0, self.x))
        
        # 最终梯度
        final_grad = (self.grad * grad_x * self.y_true.value) / N
        
        self.y_pred.accumulate_grad(_reduce_to(final_grad, self.y_pred.value.shape), owned=True)


class CrossEntropy(GradientOp):
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
        # 归一化标签为 (P - Y) / N；一般类别权重为 (P * sum(Y) - Y) / N。
        if not self.logits.requires_grad:
            return
        final_grad = softmax_cross_entropy_grad(self.probs, self._labels, self._targets)
        self.logits.accumulate_grad(self.grad * final_grad, owned=True)


class MSELoss(GradientOp):
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
        if not y_pred.requires_grad:
            return
        diff = self.y_pred_val - self.y_true_val
        n = max(diff.size, 1)
        g = self.grad * (2.0 * diff / n)
        y_pred.accumulate_grad(_reduce_to(g, y_pred.value.shape), owned=True)


class HuberLoss(GradientOp):
    """Smooth Huber (L1/L2) 回归损失，对极端异常标签更稳健。"""

    def __init__(self, y_pred, y_true, delta: float = 1.0, name="HuberLoss"):
        delta = float(delta)
        if not math.isfinite(delta) or delta <= 0.0:
            raise ValueError(f"HuberLoss delta 必须为有限正数，收到 {delta!r}")
        super().__init__(y_pred, y_true, name=name)
        self.delta = delta

    def forward(self, y_pred_val, y_true_val):
        self.y_pred_val = y_pred_val
        self.y_true_val = y_true_val
        diff = y_pred_val - y_true_val
        abs_diff = xp.abs(diff)
        d = self.delta
        elem = xp.where(abs_diff <= d, 0.5 * diff * diff, d * (abs_diff - 0.5 * d))
        self.value = xp.mean(elem, dtype=np.float64).astype(diff.dtype)

    def backward(self):
        y_pred, _y_true = self.parents
        if not y_pred.requires_grad:
            return
        diff = self.y_pred_val - self.y_true_val
        elem_grad = xp.clip(diff, -self.delta, self.delta)
        n = max(int(diff.size), 1)
        g = (self.grad * (elem_grad / n)).astype(diff.dtype, copy=False)
        y_pred.accumulate_grad(_reduce_to(g, y_pred.value.shape), owned=True)


class WeightedControlLoss(GradientOp):
    """面向 [angle, throttle] 驾驶控制的加权回归损失（支持 MSE / Huber，及弯道与通道权重）。"""

    def __init__(
        self,
        y_pred,
        y_true,
        *,
        base: str = "mse",
        delta: float = 0.2,
        angle_weight: float = 1.0,
        throttle_weight: float = 1.0,
        turn_weight: float = 0.0,
        name="WeightedControlLoss",
    ):
        if base not in ("mse", "huber"):
            raise ValueError(f"base 只能是 'mse' 或 'huber'，收到 {base!r}")
        delta = float(delta)
        angle_weight = float(angle_weight)
        throttle_weight = float(throttle_weight)
        turn_weight = float(turn_weight)
        if not math.isfinite(delta) or delta <= 0.0:
            raise ValueError(f"delta 必须为有限正数，收到 {delta!r}")
        for label_name, val in (
            ("angle_weight", angle_weight),
            ("throttle_weight", throttle_weight),
            ("turn_weight", turn_weight),
        ):
            if not math.isfinite(val) or val < 0.0:
                raise ValueError(f"{label_name} 必须为有限非负数，收到 {val!r}")
        if angle_weight + throttle_weight <= 0.0:
            raise ValueError("angle_weight 与 throttle_weight 不能同时为 0")
        super().__init__(y_pred, y_true, name=name)
        self.base = base
        self.delta = delta
        self.angle_weight = angle_weight
        self.throttle_weight = throttle_weight
        self.turn_weight = turn_weight

    def forward(self, y_pred_val, y_true_val):
        if y_pred_val.ndim < 1 or y_pred_val.shape[-1] != 2:
            raise ValueError(f"WeightedControlLoss 需要最后一维为 2 ([angle, throttle])，收到 {y_pred_val.shape}")
        if y_true_val.shape != y_pred_val.shape:
            raise ValueError(f"y_true 形状 {y_true_val.shape} 与 y_pred 形状 {y_pred_val.shape} 不匹配")
        self.y_pred_val = y_pred_val
        self.y_true_val = y_true_val
        diff = y_pred_val - y_true_val
        if self.base == "mse":
            elem = diff * diff
        else:
            abs_diff = xp.abs(diff)
            d = self.delta
            elem = xp.where(abs_diff <= d, 0.5 * diff * diff, d * (abs_diff - 0.5 * d))
        w0 = (self.angle_weight * (1.0 + self.turn_weight * xp.abs(y_true_val[..., 0]))).astype(
            diff.dtype, copy=False
        )
        w1 = xp.full_like(w0, self.throttle_weight, dtype=diff.dtype)
        weights = xp.stack([w0, w1], axis=-1)
        w_sum = xp.sum(weights, dtype=np.float64)
        self._weights = weights
        self._w_sum = w_sum
        self.value = (xp.sum(weights * elem, dtype=np.float64) / w_sum).astype(diff.dtype)

    def backward(self):
        y_pred, _y_true = self.parents
        if not y_pred.requires_grad:
            return
        diff = self.y_pred_val - self.y_true_val
        delem = 2.0 * diff if self.base == "mse" else xp.clip(diff, -self.delta, self.delta)
        inv_w_sum = (1.0 / self._w_sum).astype(diff.dtype)
        g = (self.grad * (self._weights * delem * inv_w_sum)).astype(diff.dtype, copy=False)
        y_pred.accumulate_grad(_reduce_to(g, y_pred.value.shape), owned=True)
