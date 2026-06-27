# -*- coding: utf-8 -*-
"""通用回归指标。"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from MyFlows.utils.metrics_core.common import as_1d, as_2d


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt = as_1d(y_true)
    yp = as_1d(y_pred)
    return float(np.mean((yt - yp) ** 2))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt = as_1d(y_true)
    yp = as_1d(y_pred)
    return float(np.mean(np.abs(yt - yp)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mse(y_true, y_pred)))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt = as_1d(y_true)
    yp = as_1d(y_pred)
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
    if ss_tot <= 0.0:
        return 1.0 if ss_res <= 0.0 else 0.0
    return 1.0 - ss_res / ss_tot


def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    name: str = "target",
) -> dict[str, float]:
    return {
        f"{name}_mse": mse(y_true, y_pred),
        f"{name}_mae": mae(y_true, y_pred),
        f"{name}_rmse": rmse(y_true, y_pred),
        f"{name}_r2": r2_score(y_true, y_pred),
    }


def multivariate_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    target_names: Sequence[str] | None = None,
) -> dict[str, float]:
    yt = as_2d(y_true)
    yp = as_2d(y_pred)
    if yt.shape != yp.shape:
        raise ValueError(f"shape mismatch: y_true {yt.shape} vs y_pred {yp.shape}")
    n_out = yt.shape[1]
    if target_names is None:
        target_names = [f"dim{i}" for i in range(n_out)]
    if len(target_names) != n_out:
        raise ValueError("target_names length must match output dimension")

    out: dict[str, float] = {}
    for i, name in enumerate(target_names):
        out.update(regression_metrics(yt[:, i], yp[:, i], name=name))
    out["overall_mse"] = float(np.mean((yt - yp) ** 2))
    return out
