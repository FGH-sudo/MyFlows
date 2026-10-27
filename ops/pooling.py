"""自适应平均池化与输入归一化：ImageNet 预训练模型（AlexNet、迁移训练的输入）需要的两个算子。"""

import math

import numpy as np

from ..core.device import xp
from ..core.node import Node


def adaptive_bins(size, out_size):
    """torch.nn.AdaptiveAvgPool2d 的分箱：第 i 个输出取 [floor(i*size/out), ceil((i+1)*size/out))。"""
    return [(math.floor(i * size / out_size), math.ceil((i + 1) * size / out_size)) for i in range(out_size)]


def _averaging_matrix(size, out_size, dtype):
    matrix = np.zeros((out_size, size), dtype=np.float64)
    for row, (start, end) in enumerate(adaptive_bins(size, out_size)):
        matrix[row, start:end] = 1.0 / (end - start)
    return xp.asarray(matrix.astype(dtype))


class AdaptiveAvgPool2d_Op(Node):
    """(N, C, H, W) -> (N, C, oh, ow)。

    每个输出是一个矩形窗口的均值，窗口按行、列分别分箱，所以 y = Ph · x · Pwᵀ（Ph、Pw 是每行和为 1 的平均矩阵），
    反向 dx = Phᵀ · dy · Pw。输入已是目标尺寸时（AlexNet 224 输入的 6x6）直接透传。
    """

    def __init__(self, x, output_size=(6, 6), name=None):
        super().__init__(x, name=name)
        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        self.output_size = (int(output_size[0]), int(output_size[1]))
        if min(self.output_size) <= 0:
            raise ValueError(f"output_size must be positive, got {output_size!r}")
        self._matrices = None

    def forward(self, x_val):
        _, _, h, w = x_val.shape
        oh, ow = self.output_size
        self._identity = (h, w) == (oh, ow)
        if self._identity:
            self.value = x_val
            return
        key = (h, w, x_val.dtype)
        if self._matrices is None or self._matrices[0] != key:
            self._matrices = (key, _averaging_matrix(h, oh, x_val.dtype), _averaging_matrix(w, ow, x_val.dtype))
        _, ph, pw = self._matrices
        self.value = ph @ x_val @ pw.T

    def backward(self):
        x_node = self.parents[0]
        if x_node.grad is None:
            x_node.clear_grad()
        if self._identity:
            x_node.grad += self.grad
            return
        _, ph, pw = self._matrices
        x_node.grad += ph.T @ self.grad @ pw


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ChannelNormalize_Op(Node):
    """y = (x - mean[c]) / std[c]：torchvision 预训练权重要求的输入归一化（输入为 [0, 1] 的 RGB）。"""

    def __init__(self, x, mean=IMAGENET_MEAN, std=IMAGENET_STD, name=None):
        super().__init__(x, name=name)
        self.mean = tuple(float(v) for v in mean)
        self.std = tuple(float(v) for v in std)
        if any(v <= 0 for v in self.std):
            raise ValueError("std must be positive")

    def _channel(self, values, dtype):
        return xp.asarray(np.asarray(values, dtype=dtype).reshape(1, -1, 1, 1))

    def forward(self, x_val):
        if x_val.shape[1] != len(self.mean):
            raise ValueError(f"expected {len(self.mean)} channels, got {x_val.shape[1]}")
        self.value = (x_val - self._channel(self.mean, x_val.dtype)) / self._channel(self.std, x_val.dtype)

    def backward(self):
        x_node = self.parents[0]
        if not getattr(x_node, "requires_grad", True):
            return
        if x_node.grad is None:
            x_node.clear_grad()
        x_node.grad += self.grad / self._channel(self.std, self.grad.dtype)
