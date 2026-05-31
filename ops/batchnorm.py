from ..core.device import xp
from ..core.node import Node


class BatchNorm2d_Op(Node):
    """2D 批归一化算子。

    输入:  x (N, C, H, W), gamma (C,), beta (C,)
    行为:
      - training=True: 以 mini-batch 的 (N,H,W) 均值/方差进行归一化，
        并用 momentum 更新 running_mean / running_var (仅作为推断用，不参与梯度)。
      - training=False: 使用 running_mean / running_var。

    running stats 以外部可变数组传入，Op 不持有可训练参数的状态。
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
    ):
        super().__init__(x, gamma, beta)
        self.running_mean = running_mean
        self.running_var = running_var
        self.momentum = float(momentum)
        self.eps = float(eps)
        self.training = bool(training)

    def forward(self, x_val, gamma_val, beta_val):
        N, C, H, W = x_val.shape
        if gamma_val.shape != (C,) or beta_val.shape != (C,):
            raise ValueError("gamma/beta shape must be (C,)")

        if self.training:
            # 在 (N,H,W) 上求均值与方差，得到 (C,)
            mean = x_val.mean(axis=(0, 2, 3))
            var = x_val.var(axis=(0, 2, 3))
            # 更新 running stats（就地修改外部 buffer）
            self.running_mean[:] = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var[:] = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            mean = self.running_mean
            var = self.running_var

        inv_std = 1.0 / xp.sqrt(var + self.eps)
        self._mean = mean
        self._inv_std = inv_std
        self._x_hat = (x_val - mean.reshape(1, C, 1, 1)) * inv_std.reshape(1, C, 1, 1)
        self.value = gamma_val.reshape(1, C, 1, 1) * self._x_hat + beta_val.reshape(1, C, 1, 1)

    def backward(self):
        x_node, gamma_node, beta_node = self.parents
        if x_node.grad is None:
            x_node.clear_grad()
        if gamma_node.grad is None:
            gamma_node.clear_grad()
        if beta_node.grad is None:
            beta_node.clear_grad()

        grad_y = self.grad
        N, C, H, W = grad_y.shape
        M = N * H * W

        # dgamma / dbeta
        gamma_node.grad += xp.sum(grad_y * self._x_hat, axis=(0, 2, 3))
        beta_node.grad += xp.sum(grad_y, axis=(0, 2, 3))

        if not self.training:
            # 推断期：running stats 不依赖 x
            gamma_val = gamma_node.value.reshape(1, C, 1, 1)
            x_node.grad += grad_y * gamma_val * self._inv_std.reshape(1, C, 1, 1)
            return

        # 训练期 BN 反向（标准公式）
        gamma_val = gamma_node.value.reshape(1, C, 1, 1)
        inv_std = self._inv_std.reshape(1, C, 1, 1)
        dx_hat = grad_y * gamma_val
        sum_dx_hat = xp.sum(dx_hat, axis=(0, 2, 3), keepdims=True)
        sum_dx_hat_xhat = xp.sum(dx_hat * self._x_hat, axis=(0, 2, 3), keepdims=True)
        dx = (1.0 / M) * inv_std * (M * dx_hat - sum_dx_hat - self._x_hat * sum_dx_hat_xhat)
        x_node.grad += dx


class GlobalAvgPool2d_Op(Node):
    """全局平均池化：(N, C, H, W) -> (N, C)。"""

    def forward(self, x_val):
        self._in_shape = x_val.shape
        self.value = x_val.mean(axis=(2, 3))

    def backward(self):
        x_node = self.parents[0]
        if x_node.grad is None:
            x_node.clear_grad()
        N, C, H, W = self._in_shape
        grad = self.grad.reshape(N, C, 1, 1) / (H * W)
        x_node.grad += xp.broadcast_to(grad, self._in_shape).copy()
