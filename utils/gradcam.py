# -*- coding: utf-8 -*-
"""Grad-CAM utilities for MyFlows models."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from MyFlows.core.device import asnumpy, xp
from MyFlows.core.graph import Graph
from MyFlows.core.node import Node


class TargetScore(Node):
    """Scalar node selecting one model output for explanation."""

    def __init__(self, logits, target_index: int = 0, sample_index: int = 0, name: str = "TargetScore"):
        super().__init__(logits, name=name)
        self.target_index = int(target_index)
        self.sample_index = int(sample_index)

    def forward(self, logits_val):
        self._logits_shape = logits_val.shape
        self.value = logits_val[self.sample_index, self.target_index]

    def backward(self):
        logits = self.parents[0]
        if logits.grad is None:
            logits.clear_grad()
        grad = xp.zeros_like(logits.value)
        grad[self.sample_index, self.target_index] = self.grad
        logits.grad += grad


@dataclass
class GradCamResult:
    heatmap: np.ndarray
    overlay_rgb: np.ndarray
    target_index: int
    score: float


def normalize_heatmap(heatmap: np.ndarray) -> np.ndarray:
    hm = np.asarray(heatmap, dtype=np.float32)
    hm = np.maximum(hm, 0.0)
    hm_min = float(np.min(hm)) if hm.size else 0.0
    hm_max = float(np.max(hm)) if hm.size else 0.0
    denom = hm_max - hm_min
    if denom <= 1e-12:
        return np.zeros_like(hm, dtype=np.float32)
    return ((hm - hm_min) / denom).astype(np.float32)


def overlay_heatmap(rgb: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    image = np.asarray(rgb, dtype=np.uint8)
    hm = normalize_heatmap(heatmap)
    if hm.shape[:2] != image.shape[:2]:
        hm = cv2.resize(hm, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
    color_bgr = cv2.applyColorMap(np.uint8(255 * hm), cv2.COLORMAP_JET)
    color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
    overlay = (1.0 - float(alpha)) * image.astype(np.float32) + float(alpha) * color_rgb.astype(np.float32)
    return np.clip(overlay, 0, 255).astype(np.uint8)


def chw_float_image(rgb: np.ndarray) -> np.ndarray:
    arr = np.asarray(rgb, dtype=np.float32)
    if arr.max() > 1.0:
        arr = arr / 255.0
    return np.transpose(arr, (2, 0, 1))


def heatmap_chw(heatmap: np.ndarray, image_h: int, image_w: int) -> np.ndarray:
    hm = normalize_heatmap(heatmap)
    if hm.shape != (int(image_h), int(image_w)):
        hm = cv2.resize(hm, (int(image_w), int(image_h)), interpolation=cv2.INTER_LINEAR)
    color_bgr = cv2.applyColorMap(np.uint8(255 * hm), cv2.COLORMAP_JET)
    color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
    return chw_float_image(color_rgb)


def compute_gradcam(
    logits,
    feature_node,
    *,
    target_index: int,
    sample_index: int = 0,
) -> tuple[np.ndarray, float]:
    target = TargetScore(logits, target_index=target_index, sample_index=sample_index)
    graph = Graph(target)
    graph.forward()
    score = float(asnumpy(target.value))
    graph.backward()

    activations = np.asarray(asnumpy(feature_node.value), dtype=np.float32)
    gradients = np.asarray(asnumpy(feature_node.grad), dtype=np.float32)
    if activations.ndim != 4 or gradients.ndim != 4:
        raise ValueError("Grad-CAM target feature must be a 4D NCHW tensor")
    acts = activations[sample_index]
    grads = gradients[sample_index]
    weights = np.mean(grads, axis=(1, 2))
    cam = np.sum(weights[:, None, None] * acts, axis=0)
    return normalize_heatmap(cam), score


def gradcam_for_image(
    rgb: np.ndarray,
    x_var,
    logits,
    feature_node,
    *,
    target_index: int,
    image_w: int,
    image_h: int,
    dtype=np.float32,
) -> GradCamResult:
    resized = cv2.resize(np.asarray(rgb), (int(image_w), int(image_h)), interpolation=cv2.INTER_LINEAR)
    x = resized.astype(dtype) / 255.0
    x_var.value = np.transpose(x, (2, 0, 1))[np.newaxis, ...]
    heatmap, score = compute_gradcam(logits, feature_node, target_index=target_index)
    heatmap_resized = cv2.resize(heatmap, (resized.shape[1], resized.shape[0]), interpolation=cv2.INTER_LINEAR)
    overlay = overlay_heatmap(resized, heatmap_resized)
    return GradCamResult(
        heatmap=heatmap_resized,
        overlay_rgb=overlay,
        target_index=int(target_index),
        score=float(score),
    )
