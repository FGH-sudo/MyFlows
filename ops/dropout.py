# -*- coding: utf-8 -*-
"""Dropout operator."""

from __future__ import annotations

import numpy as np

from ..core.device import xp
from ..core.node import Node


class Dropout_Op(Node):
    def __init__(self, x, p: float = 0.5, training: bool = True, seed: int | None = None, name=None):
        self.p = float(p)
        if self.p < 0.0 or self.p >= 1.0:
            raise ValueError("dropout p must be in [0, 1)")
        self.training = bool(training)
        self.seed = seed
        self._mask = None
        super().__init__(x, name=name)

    def _random(self, shape):
        if self.seed is not None:
            return xp.asarray(np.random.default_rng(int(self.seed)).random(shape))
        return xp.random.random(shape)

    def forward(self, x):
        if not self.training or self.p == 0.0:
            self._mask = xp.ones_like(x)
            self.value = x
            return
        keep = 1.0 - self.p
        self._mask = (self._random(x.shape) < keep).astype(x.dtype) / keep
        self.value = x * self._mask

    def backward(self):
        x_node = self.parents[0]
        if x_node.grad is None:
            x_node.grad = xp.zeros_like(x_node.value)
        x_node.grad += self.grad * self._mask
