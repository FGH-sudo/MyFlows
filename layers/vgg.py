# -*- coding: utf-8 -*-
"""VGG 系列回归网络（动态图风格，复用 Conv2D / MaxPool2d / Dense）。"""

from __future__ import annotations

from ..ops.activation import ReLU
from .layer import Conv2D, Dense, Dropout, Flatten, Layer, MaxPool2d


# 论文 VGG-11：每段 [conv×n, MaxPool]，卷积层数 1,1,2,2,2 共 8 层
_VGG11_CFG = [
    (1, 64),
    (1, 128),
    (2, 256),
    (2, 512),
    (2, 512),
]

# VGG-13：每段 2 个卷积，共 10 层。历史误命名为 VGG11，现有 checkpoint 基于此配置。
_VGG13_CFG = [
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


def _make_vgg_features(in_channels: int, cfg: list[tuple[int, int]], initializer=None):
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
                    initializer=initializer,
                )
            )
            ch = out_ch
        layers.append(MaxPool2d(2, 2))
    return layers, ch


def vgg_fc_input_dim(image_h: int, image_w: int, last_channels: int = 512, num_pools: int = 5) -> int:
    fh, fw = _spatial_after_pools(image_h, image_w, num_pools)
    return last_channels * fh * fw


class _VGGRegressor(Layer):
    """VGG 回归网络公共实现，适配 Donkey 小分辨率输入。"""

    _cfg: list[tuple[int, int]] = _VGG13_CFG

    def __init__(
        self,
        in_channels: int = 3,
        output_dim: int = 2,
        *,
        image_h: int = 120,
        image_w: int = 160,
        name: str | None = None,
        dropout: float = 0.0,
        initializer=None,
        cfg: list[tuple[int, int]] | None = None,
    ):
        super().__init__(name=name)
        self.output_dim = int(output_dim)
        self.image_h = int(image_h)
        self.image_w = int(image_w)
        feature_cfg = list(cfg if cfg is not None else self._cfg)

        self.feature_layers, self._last_conv_ch = _make_vgg_features(
            in_channels, feature_cfg, initializer=initializer
        )
        self.flatten = Flatten(name=f"{name}_flatten" if name else "flatten")

        self._fc_in = vgg_fc_input_dim(self.image_h, self.image_w, self._last_conv_ch)
        self.fc1 = Dense(self._fc_in, 4096, activation=ReLU, name=f"{name}_fc1" if name else "fc1", initializer=initializer)
        self.drop1 = Dropout(dropout, name=f"{name}_drop1" if name else "drop1") if float(dropout or 0.0) > 0.0 else None
        self.fc2 = Dense(4096, 4096, activation=ReLU, name=f"{name}_fc2" if name else "fc2", initializer=initializer)
        self.drop2 = Dropout(dropout, name=f"{name}_drop2" if name else "drop2") if float(dropout or 0.0) > 0.0 else None
        self.fc3 = Dense(4096, self.output_dim, name=f"{name}_fc3" if name else "fc3", initializer=initializer)

        self.params: list = []
        self.sub_layers: list[Layer] = []
        for layer in self.feature_layers:
            self.params.extend(layer.params)
            self.sub_layers.append(layer)
        self.params.extend(self.flatten.params)
        self.sub_layers.append(self.flatten)
        for layer in (self.fc1, self.drop1, self.fc2, self.drop2, self.fc3):
            if layer is None:
                continue
            self.params.extend(layer.params)
            self.sub_layers.append(layer)

    def forward(self, input_node):
        self._last_feature_nodes = {}
        x = input_node
        stage = 0
        for layer in self.feature_layers:
            x = layer(x)
            if isinstance(layer, MaxPool2d):
                stage += 1
                self._last_feature_nodes[f"stage{stage}"] = x
        x = self.flatten(x)
        x = self.fc1(x)
        if self.drop1 is not None:
            x = self.drop1(x)
        x = self.fc2(x)
        if self.drop2 is not None:
            x = self.drop2(x)
        x = self.fc3(x)
        return x

    def train(self, mode: bool = True) -> None:
        for layer in self.sub_layers:
            if hasattr(layer, "train"):
                layer.train(mode)

    def eval(self) -> None:
        self.train(False)


class VGG13(_VGGRegressor):
    """VGG-13 回归网络（10 个 3x3 卷积）。历史 checkpoint 与训练脚本使用此配置。"""

    _cfg = _VGG13_CFG


class VGG11Standard(_VGGRegressor):
    """论文口径的 VGG-11（8 个 3x3 卷积：1,1,2,2,2）。与现有 DonkeyCar checkpoint 不兼容。"""

    _cfg = _VGG11_CFG


# 兼容别名：旧代码与 JSON+NPZ checkpoint 均按 10 卷积 VGG-13 构建。
VGG11 = VGG13
