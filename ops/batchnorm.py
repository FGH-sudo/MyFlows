import numpy as np

from ..core.node import GradientOp
from .fused import batchnorm2d_backward, batchnorm2d_forward


class BatchNorm2d_Op(GradientOp):
    """2D 批归一化算子。

    输入:  x (N, C, H, W), gamma (C,), beta (C,)
    行为:
      - training=True: 以 mini-batch 的 (N,H,W) 均值/方差进行归一化，
        并用 momentum 更新 running_mean / running_var (仅作为推断用，不参与梯度)。
      - training=False: 使用 running_mean / running_var。

    running stats 以外部可变数组传入，Op 不持有可训练参数的状态。
    统计量按 FP64 累加；GPU 上前向为一次统计 + 一次归一化，反向为一次规约 + 一次 dx，
    x_hat 在反向中由 x 重算，不再为每层保存一份激活大小的中间结果。
    """

    def __init__(
        self,
        x,
        gamma,
        beta,
        running_mean,
        running_var,
        momentum=0.1,
        eps=1e-5,
        training=True,
        unbiased_running_var=False,
    ):
        super().__init__(x, gamma, beta)
        self.unbiased_running_var = bool(unbiased_running_var)
        self.running_mean = running_mean
        self.running_var = running_var
        self.momentum = float(momentum)
        self.eps = float(eps)
        self.training = bool(training)

    def forward(self, x_val, gamma_val, beta_val):
        N, C, H, W = x_val.shape
        if gamma_val.shape != (C,) or beta_val.shape != (C,):
            raise ValueError("gamma/beta shape must be (C,)")
        # backward 必须使用与本次前向相同的模式与统计量
        self._training_used = self.training
        self.value, self._mean, self._inv_std = batchnorm2d_forward(
            x_val, gamma_val, beta_val, self.running_mean, self.running_var,
            momentum=self.momentum, eps=self.eps, training=self.training,
            unbiased_running_var=self.unbiased_running_var)
        if not self.requires_grad:
            self._mean = self._inv_std = None

    def backward(self):
        x_node, gamma_node, beta_node = self.parents
        outputs = [node.grad_buffer() if node.requires_grad else (None, False) for node in self.parents]
        batchnorm2d_backward(
            x_node.value, self.grad, gamma_node.value, self._mean, self._inv_std,
            training=self._training_used, x_grad=outputs[0][0], gamma_grad=outputs[1][0],
            beta_grad=outputs[2][0], accumulate=tuple(add for _, add in outputs))


class BatchNormActivation:
    """Execution unit for BN, optional same-shape residual Add, and ReLU. Logical nodes stay addressable."""

    def __init__(self, bn, add, output, residual):
        self.bn, self.add, self.output, self.residual = bn, add, output, residual
        self.hidden = (bn, add) if add is not None else (bn,)
        self.parents = [*bn.parents] + ([residual] if residual is not None else [])
        self.name = "BatchNormAddReLU" if add is not None else "BatchNormReLU"

    def forward(self, x, gamma, beta, residual=None):
        bn = self.bn
        if gamma.shape != (x.shape[1],) or beta.shape != gamma.shape:
            raise ValueError("gamma/beta shape must be (C,)")
        self.fallback = residual is not None and (residual.shape != x.shape or residual.dtype != x.dtype)
        if self.fallback:
            # Add also supports broadcasting. Keep its original shape and reduction semantics in this case.
            bn.forward(x, gamma, beta)
            self.add.forward(*(p.value for p in self.add.parents))
            self.output.forward(self.add.value)
            return
        self.training = bn.training
        self.output.value, self.mean, self.inv_std = batchnorm2d_forward(
            x, gamma, beta, bn.running_mean, bn.running_var, momentum=bn.momentum, eps=bn.eps,
            training=self.training, unbiased_running_var=bn.unbiased_running_var, relu=True, residual=residual)
        for node in self.hidden:
            node.value = None
        if not bn.requires_grad:
            self.mean = self.inv_std = None

    def backward(self):
        bn = self.bn
        if self.fallback:
            self.output.backward()
            self.add.backward()
            if bn.requires_grad:
                bn.backward()
            for node in self.hidden:
                node.grad = None
            return
        grads = [p.grad_buffer() if p.requires_grad else (None, False) for p in self.parents]
        residual, add_residual = grads[3] if self.residual is not None else (None, False)
        batchnorm2d_backward(bn.parents[0].value, self.output.grad, bn.parents[1].value, self.mean, self.inv_std,
                             training=self.training, x_grad=grads[0][0], gamma_grad=grads[1][0], beta_grad=grads[2][0],
                             accumulate=tuple(add for _, add in grads[:3]), gate=self.output.value,
                             residual_grad=residual, add_residual=add_residual)


class GlobalAvgPool2d_Op(GradientOp):
    """全局平均池化：(N, C, H, W) -> (N, C)。"""

    def forward(self, x_val):
        self._in_shape = x_val.shape
        # FP64 累加后舍入一次（与其他规约一致），输出保持输入 dtype
        self.value = x_val.mean(axis=(2, 3), dtype=np.float64).astype(x_val.dtype, copy=False)

    def backward(self):
        x_node = self.parents[0]
        N, C, H, W = self._in_shape
        grad = self.grad.reshape(N, C, 1, 1) / (H * W)
        # 广播累加，不先物化一份 (N, C, H, W) 的副本
        x_node.accumulate_grad(grad)
