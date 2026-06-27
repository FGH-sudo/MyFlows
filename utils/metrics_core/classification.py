# -*- coding: utf-8 -*-
"""通用分类指标。"""

from __future__ import annotations

from typing import Any

import numpy as np

from MyFlows.utils.metrics_core.common import as_1d


def _predicted_labels(y_pred: np.ndarray, num_classes: int | None = None) -> np.ndarray:
    yp = np.asarray(y_pred)
    if yp.ndim == 2 and yp.shape[1] > 1:
        return np.argmax(yp, axis=1).astype(np.int64)

    labels = as_1d(yp)
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
    return as_1d(yt).astype(np.int64)


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
    for true_label, pred_label in zip(yt, yp):
        if 0 <= true_label < num_classes and 0 <= pred_label < num_classes:
            cm[true_label, pred_label] += 1
    return cm


def precision_recall_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    num_classes: int | None = None,
    average: str = "macro",
    zero_division: float = 0.0,
) -> dict[str, float]:
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

    for class_id in range(n_cls):
        tp = float(cm[class_id, class_id])
        fp = float(cm[:, class_id].sum() - tp)
        fn = float(cm[class_id, :].sum() - tp)
        prec = tp / (tp + fp) if (tp + fp) > 0 else zero_division
        rec = tp / (tp + fn) if (tp + fn) > 0 else zero_division
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else zero_division
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
    cm = confusion_matrix(y_true, y_pred, num_classes=num_classes)
    prf = precision_recall_f1(y_true, y_pred, num_classes=cm.shape[0], average="macro")
    prf_micro = precision_recall_f1(y_true, y_pred, num_classes=cm.shape[0], average="micro")
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
