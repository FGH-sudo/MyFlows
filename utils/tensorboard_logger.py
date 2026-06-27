# -*- coding: utf-8 -*-
"""TensorBoard 训练日志封装（仅写事件，训练仍在 NumPy/MyFlows 图内）。"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


class TensorBoardLogger:
    """封装 SummaryWriter，在缺少依赖时静默禁用。"""

    def __init__(self, log_dir: str | Path, *, enabled: bool = True):
        self.log_dir = Path(log_dir)
        self.enabled = bool(enabled)
        self._writer = None
        if not self.enabled:
            return
        try:
            from torch.utils.tensorboard import SummaryWriter

            self.log_dir.mkdir(parents=True, exist_ok=True)
            self._writer = SummaryWriter(log_dir=str(self.log_dir))
        except ImportError as exc:
            print(f"[TensorBoard] 未安装 tensorboard/torch，已禁用: {exc}")
            self.enabled = False

    @property
    def active(self) -> bool:
        return self.enabled and self._writer is not None

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        if not self.active:
            return
        self._writer.add_scalar(tag, float(value), int(step))

    def log_scalars(self, main_tag: str, tag_scalar_dict: Mapping[str, float], step: int) -> None:
        if not self.active:
            return
        self._writer.add_scalars(main_tag, dict(tag_scalar_dict), int(step))

    def log_scalar_dict(self, prefix: str, values: Mapping[str, float], step: int) -> None:
        if not self.active:
            return
        clean_prefix = str(prefix).strip("/")
        for key, value in values.items():
            self.log_scalar(f"{clean_prefix}/{key}", float(value), step)

    def log_histogram(self, tag: str, values, step: int) -> None:
        if not self.active:
            return
        arr = np.asarray(values)
        if arr.size == 0:
            return
        self._writer.add_histogram(tag, arr, int(step))

    def log_text(self, tag: str, text: str, step: int = 0) -> None:
        if not self.active:
            return
        self._writer.add_text(tag, str(text), int(step))

    def log_figure(self, tag: str, figure, step: int, *, close: bool = False) -> None:
        if not self.active:
            return
        self._writer.add_figure(tag, figure, int(step), close=close)

    def log_image(
        self,
        tag: str,
        image_chw: np.ndarray,
        step: int,
        *,
        dataformats: str = "CHW",
    ) -> None:
        """记录单张图像，image_chw 形状 (C,H,W)，值域建议 [0,1]。"""
        if not self.active:
            return
        arr = np.asarray(image_chw, dtype=np.float32)
        if arr.ndim != 3:
            raise ValueError(f"expected CHW image, got shape {arr.shape}")
        self._writer.add_image(tag, arr, int(step), dataformats=dataformats)

    def log_images_grid(
        self,
        tag: str,
        images: Sequence[np.ndarray],
        step: int,
        *,
        nrow: int = 4,
    ) -> None:
        """将多张 CHW 图像拼成网格写入 TensorBoard。"""
        if not self.active or not images:
            return
        try:
            import torch
            from torchvision.utils import make_grid
        except ImportError:
            if len(images) == 1:
                self.log_image(tag, images[0], step)
            return
        tensors = [torch.from_numpy(np.asarray(im, dtype=np.float32)) for im in images]
        grid = make_grid(tensors, nrow=min(nrow, len(tensors)))
        self._writer.add_image(tag, grid, int(step))

    def flush(self) -> None:
        if self.active:
            self._writer.flush()

    def close(self) -> None:
        if self.active:
            self._writer.close()
            self._writer = None

    def __enter__(self) -> TensorBoardLogger:
        return self

    def __exit__(self, *args) -> None:
        self.close()


def has_tensorboard_events(log_dir: str | Path) -> bool:
    """检查 logdir 下是否已生成 tfevents 文件。"""
    p = Path(log_dir)
    if not p.is_dir():
        return False
    return any(p.rglob("events.out.tfevents.*"))
