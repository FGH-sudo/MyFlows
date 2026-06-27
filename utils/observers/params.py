# -*- coding: utf-8 -*-
"""参数、梯度和更新比例观测指标。"""

from __future__ import annotations

import numpy as np

from MyFlows.core.device import asnumpy
from MyFlows.utils.observers.formatting import safe_name


def _to_numpy(value) -> np.ndarray:
    if value is None:
        return np.asarray([], dtype=np.float64)
    return np.asarray(asnumpy(value))


def named_parameters(model) -> list[tuple[str, object]]:
    params = list(getattr(model, "params", []) or [])
    seen: dict[str, int] = {}
    named: list[tuple[str, object]] = []
    for index, param in enumerate(params):
        base = safe_name(getattr(param, "name", None), f"param_{index}")
        count = seen.get(base, 0)
        seen[base] = count + 1
        name = base if count == 0 else f"{base}_{count}"
        named.append((name, param))
    return named


def array_norm(value) -> float:
    arr = _to_numpy(value).astype(np.float64, copy=False).reshape(-1)
    if arr.size == 0:
        return 0.0
    return float(np.linalg.norm(arr))


def parameter_norms(model, *, max_items: int = 32) -> dict[str, float]:
    out: dict[str, float] = {}
    for name, param in named_parameters(model)[: int(max_items)]:
        out[name] = array_norm(getattr(param, "value", None))
    return out


def gradient_norms(model, *, max_items: int = 32) -> dict[str, float]:
    out: dict[str, float] = {}
    for name, param in named_parameters(model)[: int(max_items)]:
        out[name] = array_norm(getattr(param, "grad", None))
    return out


def global_gradient_norm(model) -> float:
    total = 0.0
    for _, param in named_parameters(model):
        grad = _to_numpy(getattr(param, "grad", None)).astype(np.float64, copy=False).reshape(-1)
        if grad.size:
            total += float(np.dot(grad, grad))
    return float(np.sqrt(total))


def update_ratios(model, learning_rate: float, *, eps: float = 1e-12, max_items: int = 32) -> dict[str, float]:
    out: dict[str, float] = {}
    lr = float(learning_rate)
    for name, param in named_parameters(model)[: int(max_items)]:
        param_norm = array_norm(getattr(param, "value", None))
        grad_norm = array_norm(getattr(param, "grad", None))
        out[name] = float(lr * grad_norm / (param_norm + eps))
    return out


def to_numpy(value) -> np.ndarray:
    return _to_numpy(value)
