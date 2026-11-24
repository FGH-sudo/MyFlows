import numpy as np

from ..core.device import asnumpy, xp
from ..core.node import GradientOp
from .fused import batchnorm2d_backward, batchnorm2d_forward


def _split_f64(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """将 float64 精确拆为高低两段 float32 传过 collective，再累加回 float64（保留约 48 bit 尾数精度）。"""
    f64 = np.asarray(arr, dtype=np.float64)
    hi = f64.astype(np.float32)
    lo = (f64 - hi.astype(np.float64)).astype(np.float32)
    return hi, lo


def _merge_f64(hi, lo) -> np.ndarray:
    return np.asarray(hi, dtype=np.float64) + np.asarray(lo, dtype=np.float64)


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


class SyncBatchNorm2d_Op(BatchNorm2d_Op):
    """跨进程同步批归一化算子（FP64 split collective 前向统计 + 精确反向 dx 归约）。

    继承自 :class:`BatchNorm2d_Op`：推理态可被图优化折叠进 Conv，训练态通过 ``type(bn) is BatchNorm2d_Op``
    守卫跳过 BN+ReLU 融合，保证前反向集合通信完整执行。
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
        *,
        allreduce_fn=None,
        sample_weighted: bool = True,
        sync_tag: str = "bn",
    ):
        super().__init__(
            x,
            gamma,
            beta,
            running_mean,
            running_var,
            momentum=momentum,
            eps=eps,
            training=training,
            unbiased_running_var=unbiased_running_var,
        )
        self.allreduce_fn = allreduce_fn
        self.sample_weighted = bool(sample_weighted)
        self.sync_tag = str(sync_tag)
        self._global_count = None

    def forward(self, x_val, gamma_val, beta_val):
        if not self.training or self.allreduce_fn is None:
            self._global_count = None
            super().forward(x_val, gamma_val, beta_val)
            return

        N, C, H, W = x_val.shape
        if gamma_val.shape != (C,) or beta_val.shape != (C,):
            raise ValueError("gamma/beta shape must be (C,)")
        self._training_used = True
        local_count = float(N * H * W)
        s1_local = asnumpy(x_val.sum(axis=(0, 2, 3), dtype=xp.float64))
        hi, lo = _split_f64(s1_local)
        reduced = self.allreduce_fn(
            f"{self.sync_tag}:fwd_mean",
            {"hi": hi, "lo": lo, "count": np.array([local_count], dtype=np.float32)},
        )
        global_count = float(np.asarray(reduced["count"], dtype=np.float64).reshape(-1)[0])
        if global_count <= 0:
            raise ValueError(f"SyncBatchNorm2d global sample count must be positive, got {global_count}")
        mean = _merge_f64(reduced["hi"], reduced["lo"]) / global_count

        mean_dev = xp.asarray(mean, dtype=xp.float64).reshape(1, C, 1, 1)
        sq_local = asnumpy(((x_val.astype(xp.float64) - mean_dev) ** 2).sum(axis=(0, 2, 3)))
        sq_hi, sq_lo = _split_f64(sq_local)
        reduced_v = self.allreduce_fn(f"{self.sync_tag}:fwd_var", {"hi": sq_hi, "lo": sq_lo})
        var = np.maximum(_merge_f64(reduced_v["hi"], reduced_v["lo"]) / global_count, 0.0)
        inv_std = 1.0 / np.sqrt(var + self.eps)

        var_stat = var * (global_count / (global_count - 1.0)) if (self.unbiased_running_var and global_count > 1.0) else var
        prev_rm = asnumpy(self.running_mean).astype(np.float64)
        prev_rv = asnumpy(self.running_var).astype(np.float64)
        self.running_mean[...] = xp.asarray(
            (1.0 - self.momentum) * prev_rm + self.momentum * mean,
            dtype=self.running_mean.dtype,
        )
        self.running_var[...] = xp.asarray(
            (1.0 - self.momentum) * prev_rv + self.momentum * var_stat,
            dtype=self.running_var.dtype,
        )

        inv_dev = xp.asarray(inv_std, dtype=xp.float64).reshape(1, C, 1, 1)
        gamma64 = gamma_val.astype(xp.float64).reshape(1, C, 1, 1)
        beta64 = beta_val.astype(xp.float64).reshape(1, C, 1, 1)
        self.value = ((x_val.astype(xp.float64) - mean_dev) * inv_dev * gamma64 + beta64).astype(
            x_val.dtype, copy=False
        )
        if self.requires_grad:
            self._mean = mean
            self._inv_std = inv_std
            self._global_count = global_count
        else:
            self._mean = self._inv_std = self._global_count = None

    def backward(self):
        if not self._training_used or self.allreduce_fn is None:
            super().backward()
            return
        x_node, gamma_node, beta_node = self.parents
        x_val = x_node.value
        dy = self.grad
        N, C, _H, _W = x_val.shape
        mean64 = xp.asarray(self._mean, dtype=xp.float64).reshape(1, C, 1, 1)
        inv64 = xp.asarray(self._inv_std, dtype=xp.float64).reshape(1, C, 1, 1)
        x_hat = (x_val.astype(xp.float64) - mean64) * inv64
        dy64 = dy.astype(xp.float64)
        dbeta_local = dy64.sum(axis=(0, 2, 3))
        dgamma_local = (dy64 * x_hat).sum(axis=(0, 2, 3))

        if gamma_node.requires_grad:
            gamma_node.accumulate_grad(dgamma_local.astype(gamma_node.value.dtype, copy=False))
        if beta_node.requires_grad:
            beta_node.accumulate_grad(dbeta_local.astype(beta_node.value.dtype, copy=False))
        if x_node.requires_grad:
            scale = float(N) if self.sample_weighted else 1.0
            g1_hi, g1_lo = _split_f64(asnumpy(dbeta_local) * scale)
            g2_hi, g2_lo = _split_f64(asnumpy(dgamma_local) * scale)
            reduced_g = self.allreduce_fn(
                f"{self.sync_tag}:bwd",
                {"g1_hi": g1_hi, "g1_lo": g1_lo, "g2_hi": g2_hi, "g2_lo": g2_lo},
            )
            g1_global = _merge_f64(reduced_g["g1_hi"], reduced_g["g1_lo"])
            g2_global = _merge_f64(reduced_g["g2_hi"], reduced_g["g2_lo"])
            denom = scale * float(self._global_count)
            term1 = xp.asarray(g1_global / denom, dtype=xp.float64).reshape(1, C, 1, 1)
            term2 = xp.asarray(g2_global / denom, dtype=xp.float64).reshape(1, C, 1, 1)
            gamma64 = gamma_node.value.astype(xp.float64).reshape(1, C, 1, 1)
            dx = (gamma64 * inv64) * (dy64 - term1 - x_hat * term2)
            x_node.accumulate_grad(dx.astype(x_val.dtype, copy=False))


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
