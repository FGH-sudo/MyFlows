# -*- coding: utf-8 -*-
"""TensorBoard 观测指标写入辅助。"""

from __future__ import annotations

from MyFlows.utils.observers.params import named_parameters, to_numpy


def log_parameter_histograms(tb, model, step: int, *, max_items: int = 24) -> None:
    if not getattr(tb, "active", False):
        return
    for name, param in named_parameters(model)[: int(max_items)]:
        value = to_numpy(getattr(param, "value", None))
        if value.size:
            tb.log_histogram(f"params/{name}", value, step)


def log_gradient_histograms(tb, model, step: int, *, max_items: int = 24) -> None:
    if not getattr(tb, "active", False):
        return
    for name, param in named_parameters(model)[: int(max_items)]:
        grad = to_numpy(getattr(param, "grad", None))
        if grad.size:
            tb.log_histogram(f"gradients/{name}", grad, step)
