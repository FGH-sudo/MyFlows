# -*- coding: utf-8 -*-
"""训练过程 loss 曲线记录与可视化。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


@dataclass
class TrainingHistory:
    """记录 step 级与 epoch 级 loss，用于绘图与持久化。"""

    step_losses: list[float] = field(default_factory=list)
    step_indices: list[int] = field(default_factory=list)
    epoch_losses: list[float] = field(default_factory=list)
    epoch_indices: list[int] = field(default_factory=list)

    def record_step(self, global_step: int, loss: float) -> None:
        self.step_indices.append(int(global_step))
        self.step_losses.append(float(loss))

    def record_epoch(self, epoch: int, mean_loss: float) -> None:
        self.epoch_indices.append(int(epoch))
        self.epoch_losses.append(float(mean_loss))

    def to_dict(self) -> dict:
        return {
            "step_indices": list(self.step_indices),
            "step_losses": list(self.step_losses),
            "epoch_indices": list(self.epoch_indices),
            "epoch_losses": list(self.epoch_losses),
        }

    @classmethod
    def from_dict(cls, data: dict) -> TrainingHistory:
        return cls(
            step_indices=list(data.get("step_indices", [])),
            step_losses=list(data.get("step_losses", [])),
            epoch_indices=list(data.get("epoch_indices", [])),
            epoch_losses=list(data.get("epoch_losses", [])),
        )

    def save_json(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return p


def plot_training_curves(
    history: TrainingHistory,
    save_dir: str | Path,
    *,
    prefix: str = "train",
    dpi: int = 120,
) -> list[Path]:
    """
    将 step / epoch loss 曲线保存为 PNG。

    返回已写入的文件路径列表。
    """
    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    if history.step_losses:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(history.step_indices, history.step_losses, linewidth=1.0, color="#2563eb")
        ax.set_xlabel("Global step")
        ax.set_ylabel("Loss")
        ax.set_title("Training loss (per step)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        p = out_dir / f"{prefix}_loss_steps.png"
        fig.savefig(p, dpi=dpi)
        plt.close(fig)
        saved.append(p)

    if history.epoch_losses:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(
            history.epoch_indices,
            history.epoch_losses,
            marker="o",
            linewidth=1.5,
            color="#059669",
        )
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Mean loss")
        ax.set_title("Training loss (per epoch)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        p = out_dir / f"{prefix}_loss_epochs.png"
        fig.savefig(p, dpi=dpi)
        plt.close(fig)
        saved.append(p)

    if history.step_losses and history.epoch_losses:
        fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=False)
        axes[0].plot(history.step_indices, history.step_losses, linewidth=1.0, color="#2563eb")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Per-step loss")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(
            history.epoch_indices,
            history.epoch_losses,
            marker="o",
            linewidth=1.5,
            color="#059669",
        )
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Mean loss")
        axes[1].set_title("Per-epoch mean loss")
        axes[1].grid(True, alpha=0.3)

        fig.tight_layout()
        p = out_dir / f"{prefix}_loss_combined.png"
        fig.savefig(p, dpi=dpi)
        plt.close(fig)
        saved.append(p)

    return saved


def plot_metric_series(
    values: Sequence[float],
    save_path: str | Path,
    *,
    x: Sequence[int] | None = None,
    title: str = "Metric",
    ylabel: str = "Value",
    xlabel: str = "Step",
    dpi: int = 120,
) -> Path:
    """通用单序列折线图（评估或其它指标）。"""
    p = Path(save_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    xs = list(x) if x is not None else list(range(len(values)))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(xs, list(values), linewidth=1.2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(p, dpi=dpi)
    plt.close(fig)
    return p
