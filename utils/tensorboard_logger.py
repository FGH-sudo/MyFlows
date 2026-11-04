# -*- coding: utf-8 -*-
"""TensorBoard 训练日志封装（仅写事件，训练仍在 NumPy/MyFlows 图内）。"""

from __future__ import annotations

import io
import time
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


class TensorBoardLogger:
    """Write TensorBoard protobuf events without loading a second training framework."""

    def __init__(self, log_dir: str | Path, *, enabled: bool = True):
        self.log_dir = Path(log_dir)
        self.enabled = bool(enabled)
        self._writer = None
        if not self.enabled:
            return
        try:
            from tensorboard.summary.writer.event_file_writer import EventFileWriter

            self.log_dir.mkdir(parents=True, exist_ok=True)
            self._writer = EventFileWriter(str(self.log_dir))
        except ImportError as exc:
            print(f"[TensorBoard] 未安装 tensorboard，已禁用: {exc}")
            self.enabled = False

    @property
    def active(self) -> bool:
        return self.enabled and self._writer is not None

    def _event(self, value, step: int, wall_time: float | None = None) -> None:
        from tensorboard.compat.proto.event_pb2 import Event
        from tensorboard.compat.proto.summary_pb2 import Summary

        self._writer.add_event(Event(wall_time=time.time() if wall_time is None else float(wall_time),
                                     step=int(step), summary=Summary(value=[value])))

    def log_scalar(self, tag: str, value: float, step: int, *, wall_time: float | None = None) -> None:
        if not self.active:
            return
        from tensorboard.compat.proto.summary_pb2 import Summary

        self._event(Summary.Value(tag=tag, simple_value=float(value)), step, wall_time)

    def log_scalars(self, main_tag: str, tag_scalar_dict: Mapping[str, float], step: int) -> None:
        if not self.active:
            return
        self.log_scalar_dict(main_tag, tag_scalar_dict, step)

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
        from tensorboard.compat.proto.summary_pb2 import HistogramProto, Summary

        values = np.asarray(arr, dtype=np.float64).ravel()
        values = values[np.isfinite(values)]
        if not values.size:
            return
        counts, edges = np.histogram(values, bins=min(64, max(1, int(np.sqrt(values.size)))))
        hist = HistogramProto(min=float(values.min()), max=float(values.max()), num=values.size,
                              sum=float(values.sum()), sum_squares=float(np.dot(values, values)),
                              bucket_limit=edges[1:].tolist(), bucket=counts.astype(float).tolist())
        self._event(Summary.Value(tag=tag, histo=hist), step)

    def log_text(self, tag: str, text: str, step: int = 0) -> None:
        if not self.active:
            return
        from tensorboard.compat.proto.summary_pb2 import Summary, SummaryMetadata
        from tensorboard.compat.proto.tensor_pb2 import TensorProto
        from tensorboard.compat.proto.tensor_shape_pb2 import TensorShapeProto
        from tensorboard.compat.proto.types_pb2 import DT_STRING

        tensor = TensorProto(dtype=DT_STRING, string_val=[str(text).encode("utf-8")],
                             tensor_shape=TensorShapeProto(dim=[TensorShapeProto.Dim(size=1)]))
        metadata = SummaryMetadata(plugin_data=SummaryMetadata.PluginData(plugin_name="text"))
        self._event(Summary.Value(tag=f"{tag}/text_summary", tensor=tensor, metadata=metadata), step)

    def log_figure(self, tag: str, figure, step: int, *, close: bool = False) -> None:
        if not self.active:
            return
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        canvas = FigureCanvasAgg(figure)
        canvas.draw()
        self.log_image(tag, np.asarray(canvas.buffer_rgba())[..., :3].copy(), step, dataformats="HWC")
        if close:
            import matplotlib.pyplot as plt

            plt.close(figure)

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
        arr = np.asarray(image_chw)
        if arr.ndim != 3:
            raise ValueError(f"expected CHW image, got shape {arr.shape}")
        if dataformats == "CHW":
            arr = arr.transpose(1, 2, 0)
        elif dataformats != "HWC":
            raise ValueError("dataformats must be CHW or HWC")
        if arr.shape[-1] not in (1, 3, 4):
            raise ValueError("images must have 1, 3 or 4 channels")
        if arr.dtype != np.uint8:
            arr = (np.clip(np.nan_to_num(arr), 0., 1.) * 255).astype(np.uint8)
        from PIL import Image
        from tensorboard.compat.proto.summary_pb2 import Summary

        encoded = io.BytesIO()
        Image.fromarray(arr[..., 0] if arr.shape[-1] == 1 else arr).save(encoded, format="PNG")
        value = Summary.Image(height=arr.shape[0], width=arr.shape[1], colorspace=arr.shape[2],
                              encoded_image_string=encoded.getvalue())
        self._event(Summary.Value(tag=tag, image=value), step)

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
        arrays = [np.asarray(im, dtype=np.float32) for im in images]
        if any(arr.ndim != 3 or arr.shape != arrays[0].shape for arr in arrays):
            raise ValueError("image grid requires equal CHW image shapes")
        columns = min(max(1, int(nrow)), len(arrays))
        rows = (len(arrays) + columns - 1) // columns
        channels, height, width = arrays[0].shape
        padding = 2
        grid = np.zeros((channels, rows * (height + padding) + padding,
                         columns * (width + padding) + padding), dtype=np.float32)
        for index, arr in enumerate(arrays):
            y, x = (index // columns) * (height + padding) + padding, (index % columns) * (width + padding) + padding
            grid[:, y:y + height, x:x + width] = arr
        self.log_image(tag, grid, step)

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
