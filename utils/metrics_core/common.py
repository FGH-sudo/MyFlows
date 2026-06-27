# -*- coding: utf-8 -*-
"""指标计算公共数组规范化。"""

from __future__ import annotations

import numpy as np


def as_1d(arr: np.ndarray) -> np.ndarray:
    return np.asarray(arr, dtype=np.float64).reshape(-1)


def as_2d(arr: np.ndarray) -> np.ndarray:
    value = np.asarray(arr, dtype=np.float64)
    if value.ndim == 1:
        return value.reshape(-1, 1)
    return value
