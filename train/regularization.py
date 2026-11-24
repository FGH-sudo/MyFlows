# -*- coding: utf-8 -*-
"""Regularization helpers applied to accumulated optimizer gradients."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable

from ..core.device import xp

_BN_TOKEN_RE = re.compile(r"(?:^|[._/])(?:bn\d*|batchnorm\d*|beta|gamma)(?:$|[._/])", re.IGNORECASE)
_BIAS_TOKEN_RE = re.compile(r"(?:^|[._/])bias(?:$|[._/])", re.IGNORECASE)
# Case-sensitive suffix so lowercase parameter biases (fc_b, dense_b) are skipped while LoRA *_B are regularized.
_LOWER_BIAS_SUFFIX_RE = re.compile(r"(?:^|[._/])b$")


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
    raw = name or ""
    if _BN_TOKEN_RE.search(raw) or _BIAS_TOKEN_RE.search(raw) or _LOWER_BIAS_SUFFIX_RE.search(raw):
        return False
    return True


def apply_regularization(optimizer, params: Iterable, config: RegularizationConfig) -> int:
    """Add L1/L2 gradient terms to ``optimizer.acc_gradient`` in place."""

    if not config or not config.enabled:
        return 0
    applied = 0
    scale = float(max(1, int(getattr(optimizer, "acc_no", 1) or 1)))
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
        optimizer.acc_gradient[param] = optimizer.acc_gradient[param] + extra * scale
        applied += 1
    return applied
