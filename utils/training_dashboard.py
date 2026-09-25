# -*- coding: utf-8 -*-
"""训练期 TensorBoard dashboard 编排器。"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Mapping

import numpy as np

from MyFlows.utils.observers import (
    activation_stats,
    feature_grid_images,
    gradient_norms,
    label_stats_classification,
    label_stats_regression,
    log_gradient_histograms,
    log_parameter_histograms,
    parameter_norms,
    training_config_text,
    update_ratios,
)
from MyFlows.utils.tensorboard_logger import TensorBoardLogger


class TrainingDashboard:
    """封装训练脚本中的 TensorBoard tag 约定和写入时机。"""

    def __init__(
        self,
        log_dir: str | Path,
        *,
        enabled: bool = True,
        grad_interval: int = 100,
        param_interval: int = 200,
        activation_interval: int = 200,
        image_interval: int = 200,
        log_interval: int = 10,
        max_hist_params: int = 24,
        feature_channels: int = 16,
        jsonl_path: str | Path | None = None,
    ):
        self.log_dir = Path(log_dir)
        self.jsonl_path = Path(jsonl_path) if jsonl_path else None
        self._jsonl = None
        if self.jsonl_path is not None:
            self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            self._jsonl = self.jsonl_path.open("a", encoding="utf-8", buffering=1)
        self.tb = TensorBoardLogger(self.log_dir, enabled=enabled)
        self.grad_interval = int(grad_interval)
        self.param_interval = int(param_interval)
        self.activation_interval = int(activation_interval)
        self.image_interval = int(image_interval)
        self.log_interval = int(log_interval)
        self.max_hist_params = int(max_hist_params)
        self.feature_channels = int(feature_channels)

    @property
    def active(self) -> bool:
        return self.tb.active

    def _write_jsonl(self, row: dict) -> None:
        if self._jsonl is None:
            return
        row = {"monotonic_s": time.perf_counter(), "unix_s": time.time(), **row}
        self._jsonl.write(json.dumps(row, ensure_ascii=False) + "\n")

    def log_run_config(self, args, extra: Mapping[str, object] | None = None, step: int = 0) -> None:
        if self.active:
            self.tb.log_text("config/training_args", training_config_text(args, extra), step)

    def log_dataset_summary(self, labels: np.ndarray, *, task: str, num_classes: int | None = None, step: int = 0) -> None:
        if not self.active:
            return
        if task == "classification":
            if num_classes is None:
                raise ValueError("num_classes is required for classification summary")
            self.tb.log_scalar_dict("data/classes", label_stats_classification(labels, int(num_classes)), step)
            self.tb.log_histogram("data/class_histogram", labels, step)
            return
        self.tb.log_scalar_dict("data", label_stats_regression(labels), step)
        self.tb.log_histogram("data/angle_histogram", labels[:, 0], step)
        if labels.shape[1] > 1:
            self.tb.log_histogram("data/throttle_histogram", labels[:, 1], step)

    def log_augmentation(self, step: int, original: np.ndarray, augmented: np.ndarray, *, batch_size: int) -> None:
        if self.active and self.image_interval > 0 and step % self.image_interval == 0 and int(batch_size) >= 1:
            self.tb.log_images_grid("augment/original_vs_augmented", [original[0], augmented[0]], step, nrow=2)

    def log_gradients(self, step: int, model) -> None:
        if self.active and self.grad_interval > 0 and step % self.grad_interval == 0:
            self.tb.log_scalar_dict("gradients/norm", gradient_norms(model), step)
            log_gradient_histograms(self.tb, model, step, max_items=self.max_hist_params)

    def log_parameters(self, step: int, model, learning_rate: float) -> None:
        if self.active and self.param_interval > 0 and step % self.param_interval == 0:
            self.tb.log_scalar_dict("params/norm", parameter_norms(model), step)
            self.tb.log_scalar_dict("params/update_ratio", update_ratios(model, learning_rate), step)
            log_parameter_histograms(self.tb, model, step, max_items=self.max_hist_params)

    def log_train_step(
        self,
        step: int,
        *,
        loss: float,
        running_loss: float | None = None,
        accuracy: float | None = None,
        data_load_ms: float,
        train_step_ms: float,
        step_time_ms: float,
        batch_size: int,
        labels: np.ndarray,
        task: str,
        num_classes: int | None = None,
        force: bool = False,
        epoch: int | None = None,
    ) -> None:
        samples_per_sec = float(batch_size) / max(step_time_ms / 1000.0, 1e-12)
        self._write_jsonl({
            "kind": "step", "epoch": epoch, "step": int(step), "loss": float(loss),
            "running_loss": None if running_loss is None else float(running_loss),
            "accuracy": None if accuracy is None else float(accuracy),
            "data_load_ms": float(data_load_ms), "train_step_ms": float(train_step_ms),
            "step_time_ms": float(step_time_ms), "batch_size": int(batch_size), "samples_per_sec": samples_per_sec,
        })
        if not self.active:
            return
        if not force and (self.log_interval <= 0 or step % self.log_interval != 0):
            return
        self.tb.log_scalar("train/loss_step", loss, step)
        if running_loss is not None:
            self.tb.log_scalar("train/loss_running_mean", running_loss, step)
        if accuracy is not None:
            self.tb.log_scalar("train/accuracy_step", accuracy, step)
        self.tb.log_scalar("train/data_load_ms", data_load_ms, step)
        self.tb.log_scalar("train/train_step_ms", train_step_ms, step)
        self.tb.log_scalar("train/step_time_ms", step_time_ms, step)
        self.tb.log_scalar("train/samples_per_sec", samples_per_sec, step)
        if task == "classification":
            if num_classes is None:
                raise ValueError("num_classes is required for classification batch stats")
            self.tb.log_scalar_dict("data/batch_classes", label_stats_classification(labels, int(num_classes)), step)
        else:
            self.tb.log_scalar_dict("data/batch", label_stats_regression(labels), step)

    def log_activations(self, step: int, feature_nodes: Mapping[str, object], *, preferred_last: str | None = None) -> None:
        if not self.active or self.activation_interval <= 0 or step % self.activation_interval != 0:
            return
        for name, stats in activation_stats(feature_nodes).items():
            self.tb.log_scalar_dict(f"activations/{name}", stats, step)
        last_name = preferred_last if preferred_last and preferred_last in feature_nodes else next(reversed(feature_nodes), None)
        if last_name:
            images = feature_grid_images(feature_nodes[last_name], max_channels=self.feature_channels)
            if images:
                self.tb.log_images_grid(f"activations/{last_name}/feature_grid", images, step, nrow=4)

    def log_epoch(
        self,
        epoch: int,
        *,
        loss: float,
        learning_rate: float,
        epoch_time_s: float,
        accuracy: float | None = None,
        validation_loss: float | None = None,
        validation_accuracy: float | None = None,
    ) -> None:
        self._write_jsonl({
            "kind": "epoch", "epoch": int(epoch), "loss": float(loss),
            "accuracy": None if accuracy is None else float(accuracy),
            "val_loss": None if validation_loss is None else float(validation_loss),
            "val_accuracy": None if validation_accuracy is None else float(validation_accuracy),
            "learning_rate": float(learning_rate), "epoch_time_s": float(epoch_time_s),
        })
        if not self.active:
            return
        self.tb.log_scalar("train/loss_epoch", loss, epoch)
        if accuracy is not None:
            self.tb.log_scalar("train/accuracy_epoch", accuracy, epoch)
        if validation_loss is not None:
            self.tb.log_scalar("val/loss_epoch", validation_loss, epoch)
        if validation_accuracy is not None:
            self.tb.log_scalar("val/accuracy_epoch", validation_accuracy, epoch)
        self.tb.log_scalar("train/lr", learning_rate, epoch)
        self.tb.log_scalar("train/epoch_time_s", epoch_time_s, epoch)

    def log_checkpoint(self, tag: str, text: str, step: int) -> None:
        if self.active:
            self.tb.log_text(f"checkpoint/{tag}", text, step)

    def flush(self) -> None:
        self.tb.flush()
        if self._jsonl is not None:
            self._jsonl.flush()

    def close(self) -> None:
        self.tb.close()
        if self._jsonl is not None:
            self._jsonl.close()
            self._jsonl = None
