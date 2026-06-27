# -*- coding: utf-8 -*-
"""DonkeyCar 回归任务指标。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

from MyFlows.utils.metrics_core.common import as_1d, as_2d
from MyFlows.utils.metrics_core.regression import multivariate_regression_metrics


def angle_sign_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    near_zero_eps: float = 1e-3,
    zero_pred_eps: float = 0.05,
) -> dict[str, float]:
    yt = as_1d(y_true)
    yp = as_1d(y_pred)
    sign_correct = 0
    sign_total = 0
    near_zero_total = 0
    near_zero_correct = 0

    for ai_true, ai_pred in zip(yt, yp):
        sign_total += 1
        if abs(float(ai_true)) < near_zero_eps:
            near_zero_total += 1
            if abs(float(ai_pred)) < zero_pred_eps:
                near_zero_correct += 1
                sign_correct += 1
            continue
        sign_true = 1.0 if ai_true > 0 else -1.0
        sign_pred = 1.0 if ai_pred > 0 else -1.0
        if sign_true == sign_pred:
            sign_correct += 1

    return {
        "angle_sign_accuracy": sign_correct / max(1, sign_total),
        "angle_near_zero_accuracy": near_zero_correct / max(1, near_zero_total),
        "angle_sign_total": float(sign_total),
        "angle_near_zero_total": float(near_zero_total),
    }


@dataclass
class DonkeyRegressionEvaluator:
    """在评估循环中累积 batch，最后一次性汇总 Donkey 控制回归指标。"""

    target_names: tuple[str, str] = ("angle", "throttle")
    near_zero_eps: float = 1e-3
    zero_pred_eps: float = 0.05
    _y_true: list[np.ndarray] = field(default_factory=list)
    _y_pred: list[np.ndarray] = field(default_factory=list)
    _batch_count: int = 0

    def update(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        yt = as_2d(y_true)
        yp = as_2d(y_pred)
        if yt.shape != yp.shape:
            raise ValueError(f"shape mismatch: {yt.shape} vs {yp.shape}")
        self._y_true.append(yt)
        self._y_pred.append(yp)
        self._batch_count += 1

    def compute(self) -> dict[str, float]:
        if not self._y_true:
            raise RuntimeError("no batches accumulated; call update() first")
        yt = np.vstack(self._y_true)
        yp = np.vstack(self._y_pred)
        out = multivariate_regression_metrics(yt, yp, target_names=self.target_names)
        out.update(
            angle_sign_accuracy(
                yt[:, 0],
                yp[:, 0],
                near_zero_eps=self.near_zero_eps,
                zero_pred_eps=self.zero_pred_eps,
            )
        )
        out["mean_abs_angle_true"] = float(np.mean(np.abs(yt[:, 0])))
        out["mean_abs_angle_pred"] = float(np.mean(np.abs(yp[:, 0])))
        out["num_samples"] = float(yt.shape[0])
        out["num_batches"] = float(self._batch_count)
        return out

    def format_summary(self, metrics: Mapping[str, float] | None = None) -> str:
        m = dict(metrics if metrics is not None else self.compute())
        lines = [
            "=== Eval Summary ===",
            f"angle_mse={m['angle_mse']:.6f}",
            f"throttle_mse={m['throttle_mse']:.6f}",
            f"angle_mae={m['angle_mae']:.6f}",
            f"throttle_mae={m['throttle_mae']:.6f}",
            f"overall_mse={m['overall_mse']:.6f}",
            (
                f"angle_sign_accuracy={m['angle_sign_accuracy']:.4f}  "
                f"(near-zero 正确率={m['angle_near_zero_accuracy']:.4f})"
            ),
            (
                f"mean_abs_angle: true={m['mean_abs_angle_true']:.4f} "
                f"pred={m['mean_abs_angle_pred']:.4f}"
            ),
            f"samples={int(m['num_samples'])} batches={int(m['num_batches'])}",
        ]
        return "\n".join(lines)
