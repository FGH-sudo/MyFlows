# -*- coding: utf-8 -*-
"""VGG 系列分类网络（动态图风格，复用 Conv2D / MaxPool2d / Dense）。"""

from __future__ import annotations

import numpy as np

from ..core.node import Variable
from ..ops.activation import ReLU
from ..ops.basic import Linear
from .layer import Conv2D, Dense, Flatten, Layer, MaxPool2d


# VGG11: 每段 [conv×n, MaxPool]；通道 64,128,256,512,512
_VGG11_CFG = [
    (2, 64),
    (2, 128),
    (2, 256),
    (2, 512),
    (2, 512),
]


def _spatial_after_pools(h: int, w: int, num_pools: int = 5) -> tuple[int, int]:
    for _ in range(num_pools):
        h, w = h // 2, w // 2
    return max(1, h), max(1, w)


def _make_vgg_features(in_channels: int, cfg: list[tuple[int, int]]):
    layers: list[Layer] = []
    ch = in_channels
    for num_convs, out_ch in cfg:
        for _ in range(num_convs):
            layers.append(
                Conv2D(
                    ch,
                    out_ch,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    activation=ReLU,
                    name=None,
                )
            )
            ch = out_ch
        layers.append(MaxPool2d(2, 2))
    return layers, ch


def vgg_fc_input_dim(image_h: int, image_w: int, last_channels: int = 512, num_pools: int = 5) -> int:
    fh, fw = _spatial_after_pools(image_h, image_w, num_pools)
    return last_channels * fh * fw


class VGG11(Layer):
    """
    VGG-11 分类器，适配 Donkey 小分辨率输入。

    参数:
      in_channels: 输入通道（RGB=3）
      num_classes: 分类数
      small_input: True 时假设输入约 120×160，经 5 次池化后特征图较小；
                   全连接输入维由首次 forward 前根据 Variable 形状推算（此处用固定估算）。
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 5,
        *,
        image_h: int = 120,
        image_w: int = 160,
        name: str | None = None,
    ):
        super().__init__(name=name)
        self.num_classes = int(num_classes)
        self.image_h = int(image_h)
        self.image_w = int(image_w)

        self.feature_layers, self._last_conv_ch = _make_vgg_features(in_channels, _VGG11_CFG)
        self.flatten = Flatten(name=f"{name}_flatten" if name else "flatten")

        self._fc_in = vgg_fc_input_dim(self.image_h, self.image_w, self._last_conv_ch)
        self.fc1 = Dense(self._fc_in, 4096, activation=ReLU, name=f"{name}_fc1" if name else "fc1")
        self.fc2 = Dense(4096, 4096, activation=ReLU, name=f"{name}_fc2" if name else "fc2")
        self.fc3 = Dense(4096, num_classes, name=f"{name}_fc3" if name else "fc3")

        self.params: list = []
        self.sub_layers: list[Layer] = []
        for layer in self.feature_layers:
            self.params.extend(layer.params)
            self.sub_layers.append(layer)
        self.params.extend(self.flatten.params)
        for fc in (self.fc1, self.fc2, self.fc3):
            self.params.extend(fc.params)
            self.sub_layers.extend([self.fc1, self.fc2, self.fc3])

    def forward(self, input_node):
        x = input_node
        for layer in self.feature_layers:
            x = layer(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

    def train(self, mode: bool = True) -> None:
        pass

    def eval(self) -> None:
        pass


def angle_to_class(angle: float, num_classes: int = 5) -> int:
    """将连续转向角离散为分类标签。"""
    a = float(angle)
    if num_classes == 3:
        if a < -0.05:
            return 0
        if a > 0.05:
            return 2
        return 1
    if num_classes == 5:
        edges = [-0.2, -0.05, 0.05, 0.2]
        for i, edge in enumerate(edges):
            if a < edge:
                return i
        return 4
    # 均匀分箱
    lo, hi = -1.0, 1.0
    idx = int((a - lo) / (hi - lo) * num_classes)
    return int(np.clip(idx, 0, num_classes - 1))
