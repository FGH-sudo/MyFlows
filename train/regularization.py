# -*- coding: utf-8 -*-
"""Regularization helpers applied to accumulated optimizer gradients."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from ..core.device import xp


@dataclass
class RegularizationConfig:
    l1: float = 0.0
    l2: float = 0.0
    regularize_bias_bn: bool = False

    @property
    def enabled(self) -> bool:
        return float(self.l1) != 0.0 or float(self.l2) != 0.0


def _regularize_param(name: str, regularize_bias_bn: bool) -> bool:
    if regularize_bias_bn:
        return True
    lowered = (name or "").lower()
    excluded = ("bias", "_b", "beta", "gamma", "bn")
    return not any(part in lowered for part in excluded)


def apply_regularization(optimizer, params: Iterable, config: RegularizationConfig) -> int:
    """Add L1/L2 gradient terms to ``optimizer.acc_gradient`` in place."""

    if not config or not config.enabled:
        return 0
    applied = 0
    for param in params:
        if not getattr(param, "trainable", False):
            continue
        if not _regularize_param(getattr(param, "name", ""), bool(config.regularize_bias_bn)):
            continue
        if param not in optimizer.acc_gradient or optimizer.acc_gradient[param] is None:
            continue
        value = xp.asarray(param.value)
        extra = xp.zeros_like(value)
        if float(config.l2) != 0.0:
            extra = extra + float(config.l2) * value
        if float(config.l1) != 0.0:
            extra = extra + float(config.l1) * xp.sign(value)
        optimizer.acc_gradient[param] = optimizer.acc_gradient[param] + extra
        applied += 1
    return applied
