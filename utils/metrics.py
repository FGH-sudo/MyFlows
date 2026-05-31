# -*- coding: utf-8 -*-
"""统一回归与分类评估指标。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np


def _as_1d(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float64).reshape(-1)
    return a


def _as_2d(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim == 1:
        return a.reshape(-1, 1)
    return a


# ---------------------------------------------------------------------------
# 回归
# ---------------------------------------------------------------------------


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt = _as_1d(y_true)
    yp = _as_1d(y_pred)
    return float(np.mean((yt - yp) ** 2))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt = _as_1d(y_true)
    yp = _as_1d(y_pred)
    return float(np.mean(np.abs(yt - yp)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mse(y_true, y_pred)))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt = _as_1d(y_true)
    yp = _as_1d(y_pred)
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
    """单通道回归指标摘要。"""
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
    """多输出回归：对每个维度分别计算 MSE/MAE/RMSE/R2，并给出整体 MSE。"""
    yt = _as_2d(y_true)
    yp = _as_2d(y_pred)
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


def angle_sign_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    near_zero_eps: float = 1e-3,
    zero_pred_eps: float = 0.05,
) -> dict[str, float]:
    """
    转向角符号正确率（Donkey 回归任务常用）。

    真值接近 0 时，若预测绝对值也足够小则计为正确。
    """
    yt = _as_1d(y_true)
    yp = _as_1d(y_pred)
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


# ---------------------------------------------------------------------------
# 分类
# ---------------------------------------------------------------------------


def _predicted_labels(y_pred: np.ndarray, num_classes: int | None = None) -> np.ndarray:
    yp = np.asarray(y_pred)
    if yp.ndim == 2 and yp.shape[1] > 1:
        return np.argmax(yp, axis=1).astype(np.int64)

    labels = _as_1d(yp)
    if np.issubdtype(labels.dtype, np.integer):
        return labels.astype(np.int64)

    rounded = np.rint(labels)
    if np.allclose(labels, rounded, atol=1e-6, rtol=0.0):
        ints = rounded.astype(np.int64)
        if num_classes is None or (np.all(ints >= 0) and np.all(ints < num_classes)):
            return ints

    if num_classes is not None and num_classes > 2:
        raise ValueError("multiclass predictions must be integer labels or (N,C) logits")
    return (labels >= 0.5).astype(np.int64)


def _true_labels(y_true: np.ndarray) -> np.ndarray:
    yt = np.asarray(y_true)
    if yt.ndim == 2 and yt.shape[1] > 1:
        return np.argmax(yt, axis=1).astype(np.int64)
    return _as_1d(yt).astype(np.int64)


def accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    num_classes: int | None = None,
) -> float:
    yt = _true_labels(y_true)
    yp = _predicted_labels(y_pred, num_classes=num_classes)
    return float(np.mean(yt == yp))


def confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    num_classes: int | None = None,
) -> np.ndarray:
    yt = _true_labels(y_true)
    yp = _predicted_labels(y_pred, num_classes=num_classes)
    if num_classes is None:
        num_classes = int(max(np.max(yt), np.max(yp))) + 1
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(yt, yp):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1
    return cm


def precision_recall_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    num_classes: int | None = None,
    average: str = "macro",
    zero_division: float = 0.0,
) -> dict[str, float]:
    """
    多分类 precision / recall / F1。

    average: 'macro' | 'micro' | 'weighted' | 'none'（返回每类列表时用 dict 嵌套）
    """
    if average not in ("macro", "micro", "weighted", "none"):
        raise ValueError(f"unsupported average={average!r}")

    yt = _true_labels(y_true)
    yp = _predicted_labels(y_pred, num_classes=num_classes)
    cm = confusion_matrix(yt, yp, num_classes=num_classes)
    n_cls = cm.shape[0]

    per_prec: list[float] = []
    per_rec: list[float] = []
    per_f1: list[float] = []
    support = cm.sum(axis=1).astype(np.float64)

    for c in range(n_cls):
        tp = float(cm[c, c])
        fp = float(cm[:, c].sum() - tp)
        fn = float(cm[c, :].sum() - tp)
        prec = tp / (tp + fp) if (tp + fp) > 0 else zero_division
        rec = tp / (tp + fn) if (tp + fn) > 0 else zero_division
        f1 = (
            2 * prec * rec / (prec + rec)
            if (prec + rec) > 0
            else zero_division
        )
        per_prec.append(prec)
        per_rec.append(rec)
        per_f1.append(f1)

    if average == "none":
        return {
            "precision_per_class": per_prec,
            "recall_per_class": per_rec,
            "f1_per_class": per_f1,
            "support": support.tolist(),
        }

    if average == "micro":
        tp = float(np.trace(cm))
        total = float(cm.sum())
        prec = rec = f1 = tp / total if total > 0 else zero_division
        return {"precision": prec, "recall": rec, "f1": f1}

    if average == "weighted":
        weights = support / max(1.0, support.sum())
        return {
            "precision": float(np.dot(per_prec, weights)),
            "recall": float(np.dot(per_rec, weights)),
            "f1": float(np.dot(per_f1, weights)),
        }

    # macro
    return {
        "precision": float(np.mean(per_prec)),
        "recall": float(np.mean(per_rec)),
        "f1": float(np.mean(per_f1)),
    }


def classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    num_classes: int | None = None,
) -> dict[str, Any]:
    """分类评估摘要（accuracy + macro PRF1 + 混淆矩阵）。"""
    cm = confusion_matrix(y_true, y_pred, num_classes=num_classes)
    prf = precision_recall_f1(
        y_true, y_pred, num_classes=cm.shape[0], average="macro"
    )
    prf_micro = precision_recall_f1(
        y_true, y_pred, num_classes=cm.shape[0], average="micro"
    )
    return {
        "accuracy": accuracy(y_true, y_pred, num_classes=cm.shape[0]),
        "precision_macro": prf["precision"],
        "recall_macro": prf["recall"],
        "f1_macro": prf["f1"],
        "precision_micro": prf_micro["precision"],
        "recall_micro": prf_micro["recall"],
        "f1_micro": prf_micro["f1"],
        "confusion_matrix": cm.tolist(),
        "num_classes": int(cm.shape[0]),
    }


# ---------------------------------------------------------------------------
# Donkey 回归评估器（流式累积 batch）
# ---------------------------------------------------------------------------


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
        yt = _as_2d(y_true)
        yp = _as_2d(y_pred)
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
        out = multivariate_regression_metrics(
            yt, yp, target_names=self.target_names
        )
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
