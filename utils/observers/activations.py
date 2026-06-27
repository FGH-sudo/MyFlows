# -*- coding: utf-8 -*-
"""激活统计和特征图可视化观测指标。"""

from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np

from MyFlows.utils.observers.formatting import safe_name
from MyFlows.utils.observers.params import to_numpy


def activation_stats(nodes: Mapping[str, object] | Iterable[tuple[str, object]], *, eps: float = 1e-8) -> dict[str, dict[str, float]]:
    items = nodes.items() if isinstance(nodes, Mapping) else nodes
    out: dict[str, dict[str, float]] = {}
    for name, node in items:
        value = to_numpy(getattr(node, "value", None)).astype(np.float64, copy=False)
        if value.size == 0:
            continue
        out[safe_name(name, "activation")] = {
            "mean": float(np.mean(value)),
            "std": float(np.std(value)),
            "min": float(np.min(value)),
            "max": float(np.max(value)),
            "sparsity": float(np.mean(np.abs(value) < float(eps))),
        }
    return out


def feature_grid_images(node, *, max_channels: int = 16) -> list[np.ndarray]:
    value = to_numpy(getattr(node, "value", None)).astype(np.float32, copy=False)
    if value.ndim != 4 or value.shape[0] == 0:
        return []
    feature = value[0]
    images: list[np.ndarray] = []
    for channel in range(min(int(max_channels), feature.shape[0])):
        fmap = feature[channel]
        fmap_min = float(np.min(fmap))
        fmap_max = float(np.max(fmap))
        denom = fmap_max - fmap_min
        if denom <= 1e-12:
            norm = np.zeros_like(fmap, dtype=np.float32)
        else:
            norm = ((fmap - fmap_min) / denom).astype(np.float32)
        images.append(norm[np.newaxis, ...])
    return images
