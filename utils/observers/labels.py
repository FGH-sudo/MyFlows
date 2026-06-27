# -*- coding: utf-8 -*-
"""训练标签分布观测指标。"""

from __future__ import annotations

import numpy as np


def label_stats_regression(y_batch: np.ndarray) -> dict[str, float]:
    y = np.asarray(y_batch, dtype=np.float64)
    angle = y[:, 0]
    throttle = y[:, 1] if y.shape[1] > 1 else np.zeros_like(angle)
    return {
        "angle_mean": float(np.mean(angle)),
        "angle_std": float(np.std(angle)),
        "angle_min": float(np.min(angle)),
        "angle_max": float(np.max(angle)),
        "throttle_mean": float(np.mean(throttle)),
        "throttle_std": float(np.std(throttle)),
    }


def label_stats_classification(y_batch: np.ndarray, num_classes: int) -> dict[str, float]:
    labels = np.asarray(y_batch).reshape(-1).astype(int)
    counts = np.bincount(labels, minlength=int(num_classes)).astype(np.float64)
    total = max(float(np.sum(counts)), 1.0)
    out = {f"class_{i}_ratio": float(count / total) for i, count in enumerate(counts)}
    out["class_entropy"] = float(-np.sum((counts / total) * np.log((counts / total) + 1e-12)))
    return out
