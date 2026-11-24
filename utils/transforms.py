# -*- coding: utf-8 -*-
"""图像数据增强（纯 NumPy + OpenCV），支持 HWC [0,1] 浮点图像。"""

from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


def _require_cv2() -> None:
    if cv2 is None:
        raise ImportError("transforms 需要 opencv-python (cv2)")


def _clip01(img: np.ndarray) -> np.ndarray:
    return np.clip(img, 0.0, 1.0).astype(np.float32, copy=False)


def chw_to_hwc(x: np.ndarray) -> np.ndarray:
    if x.ndim == 3 and x.shape[0] in (1, 3):
        return np.transpose(x, (1, 2, 0))
    return x


def hwc_to_chw(x: np.ndarray) -> np.ndarray:
    if x.ndim == 3 and x.shape[-1] in (1, 3):
        return np.transpose(x, (2, 0, 1))
    return x


class Transform(ABC):
    """单样本变换：输入 HWC 图像与可选标签，返回 (image, label)。"""

    @abstractmethod
    def __call__(self, image: np.ndarray, label: Any = None) -> tuple[np.ndarray, Any]:
        raise NotImplementedError


class ComposeTransform(Transform):
    def __init__(self, transforms: list[Transform]):
        self.transforms = list(transforms)

    def __call__(self, image: np.ndarray, label: Any = None) -> tuple[np.ndarray, Any]:
        out_img, out_label = image, label
        for t in self.transforms:
            out_img, out_label = t(out_img, out_label)
        return out_img, out_label


class RandomCrop(Transform):
    """随机裁剪后 resize 回原尺寸。"""

    def __init__(self, scale: tuple[float, float] = (0.8, 1.0), seed: int | None = None):
        self.scale = scale
        self._rng = random.Random(seed)

    def __call__(self, image: np.ndarray, label: Any = None) -> tuple[np.ndarray, Any]:
        _require_cv2()
        h, w = image.shape[:2]
        ratio = self._rng.uniform(self.scale[0], self.scale[1])
        ch, cw = max(1, int(h * ratio)), max(1, int(w * ratio))
        y0 = self._rng.randint(0, max(0, h - ch))
        x0 = self._rng.randint(0, max(0, w - cw))
        crop = image[y0 : y0 + ch, x0 : x0 + cw]
        out = cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)
        return _clip01(out), label


class RandomRotation(Transform):
    def __init__(self, degrees: float = 15.0, seed: int | None = None):
        self.degrees = float(degrees)
        self._rng = random.Random(seed)

    def __call__(self, image: np.ndarray, label: Any = None) -> tuple[np.ndarray, Any]:
        _require_cv2()
        h, w = image.shape[:2]
        angle = self._rng.uniform(-self.degrees, self.degrees)
        m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        out = cv2.warpAffine(
            image,
            m,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        return _clip01(out), label


class ColorJitter(Transform):
    """亮度 / 对比度 / 饱和度随机扰动。"""

    def __init__(
        self,
        brightness: float = 0.2,
        contrast: float = 0.2,
        saturation: float = 0.2,
        seed: int | None = None,
    ):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self._rng = random.Random(seed)

    def __call__(self, image: np.ndarray, label: Any = None) -> tuple[np.ndarray, Any]:
        _require_cv2()
        out = image.astype(np.float32, copy=True)
        # brightness
        b = 1.0 + self._rng.uniform(-self.brightness, self.brightness)
        out = out * b
        # contrast
        c = 1.0 + self._rng.uniform(-self.contrast, self.contrast)
        mean = out.mean(axis=(0, 1), keepdims=True)
        out = (out - mean) * c + mean
        # saturation
        s = 1.0 + self._rng.uniform(-self.saturation, self.saturation)
        gray = np.mean(out, axis=2, keepdims=True)
        out = gray + s * (out - gray)
        return _clip01(out), label


class MixUp(Transform):
    """与 batch 内另一样本混合。label 为向量时线性混合。"""

    def __init__(self, alpha: float = 0.4, seed: int | None = None):
        self.alpha = float(alpha)
        self._rng = random.Random(seed)
        self._partner_img: np.ndarray | None = None
        self._partner_label: Any = None

    def set_partner(self, image: np.ndarray, label: Any) -> None:
        self._partner_img = np.asarray(image, dtype=np.float32)
        self._partner_label = label

    def __call__(self, image: np.ndarray, label: Any = None) -> tuple[np.ndarray, Any]:
        if self._partner_img is None:
            return image, label
        lam = float(self._rng.betavariate(self.alpha, self.alpha))
        mixed = lam * image + (1.0 - lam) * self._partner_img
        mixed = _clip01(mixed)
        if label is None or self._partner_label is None:
            return mixed, label
        y_a = np.asarray(label, dtype=np.float64)
        y_b = np.asarray(self._partner_label, dtype=np.float64)
        if y_a.shape == y_b.shape:
            return mixed, lam * y_a + (1.0 - lam) * y_b
        return mixed, label


class CutMix(Transform):
    """随机区域粘贴 partner 图像块；回归/向量标签按面积比混合。"""

    def __init__(self, alpha: float = 1.0, seed: int | None = None):
        self.alpha = float(alpha)
        self._rng = random.Random(seed)
        self._partner_img: np.ndarray | None = None
        self._partner_label: Any = None

    def set_partner(self, image: np.ndarray, label: Any) -> None:
        self._partner_img = np.asarray(image, dtype=np.float32)
        self._partner_label = label

    def __call__(self, image: np.ndarray, label: Any = None) -> tuple[np.ndarray, Any]:
        if self._partner_img is None:
            return image, label
        h, w = image.shape[:2]
        lam = float(self._rng.betavariate(self.alpha, self.alpha))
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = int(w * cut_rat)
        cut_h = int(h * cut_rat)
        cx = self._rng.randint(0, w)
        cy = self._rng.randint(0, h)
        x1 = int(np.clip(cx - cut_w // 2, 0, w))
        y1 = int(np.clip(cy - cut_h // 2, 0, h))
        x2 = int(np.clip(cx + cut_w // 2, 0, w))
        y2 = int(np.clip(cy + cut_h // 2, 0, h))
        if x2 <= x1 or y2 <= y1:
            return image, label
        out = image.copy()
        out[y1:y2, x1:x2] = self._partner_img[y1:y2, x1:x2]
        lam_adj = 1.0 - (x2 - x1) * (y2 - y1) / (w * h)
        if label is None or self._partner_label is None:
            return _clip01(out), label
        y_a = np.asarray(label, dtype=np.float64)
        y_b = np.asarray(self._partner_label, dtype=np.float64)
        if y_a.shape == y_b.shape:
            return _clip01(out), lam_adj * y_a + (1.0 - lam_adj) * y_b
        return _clip01(out), label


def default_train_transforms(seed: int | None = None) -> ComposeTransform:
    """训练常用增强流水线（不含 MixUp/CutMix，需在 batch 级配对）。"""
    return ComposeTransform([
        RandomCrop(seed=seed),
        RandomRotation(seed=seed),
        ColorJitter(seed=seed),
    ])


class RandomHorizontalShift(Transform):
    """几何一致的水平/轻微垂直平移，并按平移幅度补偿转向角（右移图像 -> 路标右移 -> 增加右转角）。"""

    def __init__(
        self,
        max_shift_frac: float = 0.08,
        steer_per_frac: float = 1.2,
        max_vertical_frac: float = 0.04,
        seed: int | None = None,
    ) -> None:
        max_shift_frac = float(max_shift_frac)
        steer_per_frac = float(steer_per_frac)
        max_vertical_frac = float(max_vertical_frac)
        for name, val in (
            ("max_shift_frac", max_shift_frac),
            ("steer_per_frac", steer_per_frac),
            ("max_vertical_frac", max_vertical_frac),
        ):
            if not math.isfinite(val) or val < 0.0:
                raise ValueError(f"{name} 必须为有限非负数，收到 {val!r}")
        self.max_shift_frac = max_shift_frac
        self.steer_per_frac = steer_per_frac
        self.max_vertical_frac = max_vertical_frac
        self._rng = random.Random(seed)

    def __call__(self, image: np.ndarray, label: Any = None) -> tuple[np.ndarray, Any]:
        _require_cv2()
        h, w = image.shape[:2]
        frac_x = self._rng.uniform(-self.max_shift_frac, self.max_shift_frac)
        frac_y = self._rng.uniform(-self.max_vertical_frac, self.max_vertical_frac)
        dx = frac_x * float(w)
        dy = frac_y * float(h)
        m = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
        src = np.asarray(image, dtype=np.float32)
        out = cv2.warpAffine(
            src,
            m,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        out_label = label
        delta = frac_x * self.steer_per_frac
        if label is not None:
            if isinstance(label, np.ndarray):
                out_label = label.copy()
                if out_label.ndim == 0:
                    out_label = np.asarray(
                        np.clip(float(out_label) + delta, -1.0, 1.0),
                        dtype=label.dtype,
                    )
                else:
                    flat = out_label.reshape(-1)
                    if flat.size >= 1:
                        flat[0] = np.clip(float(flat[0]) + delta, -1.0, 1.0)
            elif isinstance(label, (list, tuple)):
                out_vals = list(label)
                if out_vals:
                    out_vals[0] = float(np.clip(float(out_vals[0]) + delta, -1.0, 1.0))
                out_label = tuple(out_vals) if isinstance(label, tuple) else out_vals
            elif isinstance(label, (int, float, np.floating, np.integer)):
                out_label = float(np.clip(float(label) + delta, -1.0, 1.0))
        return _clip01(out), out_label


class RandomShadow(Transform):
    """沿随机半平面施加局部阴影遮暗，不改变控制标签。"""

    def __init__(
        self,
        prob: float = 0.5,
        min_alpha: float = 0.45,
        max_alpha: float = 0.85,
        seed: int | None = None,
    ) -> None:
        prob = float(prob)
        min_alpha = float(min_alpha)
        max_alpha = float(max_alpha)
        if not math.isfinite(prob) or not (0.0 <= prob <= 1.0):
            raise ValueError(f"prob 必须在 [0, 1] 内，收到 {prob!r}")
        if not math.isfinite(min_alpha) or not math.isfinite(max_alpha) or not (0.0 < min_alpha <= max_alpha <= 1.0):
            raise ValueError(f"min_alpha/max_alpha 需满足 0 < min_alpha <= max_alpha <= 1，收到 {(min_alpha, max_alpha)!r}")
        self.prob = prob
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
        self._rng = random.Random(seed)

    def __call__(self, image: np.ndarray, label: Any = None) -> tuple[np.ndarray, Any]:
        if self._rng.random() >= self.prob:
            return _clip01(image), label
        h, w = image.shape[:2]
        x1 = self._rng.uniform(0.0, float(w))
        x2 = self._rng.uniform(0.0, float(w))
        side = self._rng.random() < 0.5
        alpha = self._rng.uniform(self.min_alpha, self.max_alpha)
        ys = np.arange(h, dtype=np.float32)[:, None]
        xs = np.arange(w, dtype=np.float32)[None, :]
        cross = (x2 - x1) * ys - float(h) * (xs - x1)
        mask = (cross >= 0.0) if side else (cross < 0.0)
        out = image.astype(np.float32, copy=True)
        out[mask] *= float(alpha)
        return _clip01(out), label


def driving_train_transforms(seed: int | None = None) -> ComposeTransform:
    """自动驾驶训练增强流水线（水平平移+转向补偿 + 随机阴影 + 色彩抖动）。"""
    s0 = seed
    s1 = None if seed is None else int(seed) + 101
    s2 = None if seed is None else int(seed) + 202
    return ComposeTransform([
        RandomHorizontalShift(max_shift_frac=0.08, steer_per_frac=1.2, max_vertical_frac=0.04, seed=s0),
        RandomShadow(prob=0.5, min_alpha=0.45, max_alpha=0.85, seed=s1),
        ColorJitter(brightness=0.25, contrast=0.2, saturation=0.15, seed=s2),
    ])


def apply_batch_pairwise_mix(
    images_hwc: list[np.ndarray],
    labels: list[Any],
    *,
    use_mixup: bool = True,
    use_cutmix: bool = True,
    alpha_mixup: float = 0.4,
    alpha_cutmix: float = 1.0,
    seed: int | None = None,
) -> tuple[list[np.ndarray], list[Any]]:
    """对 batch 内样本随机施加 MixUp 或 CutMix（互斥择一）。"""
    rng = random.Random(seed)
    n = len(images_hwc)
    if n < 2:
        return images_hwc, labels
    order = list(range(n))
    rng.shuffle(order)
    out_imgs: list[np.ndarray] = []
    out_labels: list[Any] = []
    for i in range(n):
        img, lab = images_hwc[i], labels[i]
        partner_idx = order[i]
        if partner_idx == i:
            partner_idx = (i + 1) % n
        partner_img = images_hwc[partner_idx]
        partner_lab = labels[partner_idx]
        choice = rng.random()
        mix_seed = (seed + i * 17) if seed is not None else None
        if use_mixup and (not use_cutmix or choice < 0.5):
            t = MixUp(alpha=alpha_mixup, seed=mix_seed)
            t.set_partner(partner_img, partner_lab)
            img, lab = t(img, lab)
        elif use_cutmix:
            t = CutMix(alpha=alpha_cutmix, seed=mix_seed)
            t.set_partner(partner_img, partner_lab)
            img, lab = t(img, lab)
        out_imgs.append(img)
        out_labels.append(lab)
    return out_imgs, out_labels


def augment_chw_batch(
    x_batch: np.ndarray,
    y_batch: np.ndarray,
    transform: ComposeTransform | None,
    *,
    mixup: bool = False,
    cutmix: bool = False,
    mix_seed: int | None = None,
) -> np.ndarray:
    """
    对 NCHW batch 做增强，返回同形状 NCHW。
    y_batch 可能被 MixUp/CutMix 修改（调用方应使用返回值）。
    mix_seed: MixUp/CutMix 配对与混合比例的种子；None 时每次随机。
    """
    b, c, h, w = x_batch.shape
    imgs_hwc = [chw_to_hwc(x_batch[i]) for i in range(b)]
    labels = [y_batch[i].copy() for i in range(b)]
    if transform is not None:
        for i in range(b):
            imgs_hwc[i], labels[i] = transform(imgs_hwc[i], labels[i])
    if mixup or cutmix:
        imgs_hwc, labels = apply_batch_pairwise_mix(
            imgs_hwc,
            labels,
            use_mixup=mixup,
            use_cutmix=cutmix,
            seed=mix_seed,
        )
    for i in range(b):
        y_batch[i] = labels[i]
        x_batch[i] = hwc_to_chw(imgs_hwc[i])
    return x_batch
