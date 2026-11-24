# -*- coding: utf-8 -*-
"""学习率调度器。"""

from __future__ import annotations

import math


class CosineWarmupScheduler:
    """线性预热 + 余弦退火学习率调度器。"""

    def __init__(
        self,
        base_lr: float,
        total_steps: int,
        *,
        warmup_steps: int = 0,
        min_lr: float = 0.0,
    ) -> None:
        base_lr = float(base_lr)
        total_steps = int(total_steps)
        warmup_steps = int(warmup_steps)
        min_lr = float(min_lr)
        if not math.isfinite(base_lr) or base_lr <= 0.0:
            raise ValueError(f"base_lr 必须为有限正数，收到 {base_lr!r}")
        if total_steps < 1:
            raise ValueError(f"total_steps 必须 >= 1，收到 {total_steps!r}")
        if warmup_steps < 0:
            raise ValueError(f"warmup_steps 不能为负数，收到 {warmup_steps!r}")
        if not math.isfinite(min_lr) or min_lr < 0.0 or min_lr > base_lr:
            raise ValueError(f"min_lr 必须满足 0 <= min_lr <= base_lr ({base_lr})，收到 {min_lr!r}")

        self.base_lr = base_lr
        self.total_steps = total_steps
        self.warmup_steps = min(warmup_steps, total_steps)
        self.min_lr = min_lr

    def lr_at(self, step: int) -> float:
        step = max(0, int(step))
        if self.warmup_steps > 0 and step < self.warmup_steps:
            return float(self.base_lr * (step + 1) / self.warmup_steps)
        if self.warmup_steps == 0:
            if self.total_steps <= 1:
                return float(self.base_lr)
            progress = min(1.0, step / float(self.total_steps - 1))
        else:
            decay_span = self.total_steps - self.warmup_steps
            if decay_span <= 0:
                return float(self.base_lr)
            progress = min(1.0, (step - self.warmup_steps + 1) / float(decay_span))
        return float(self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1.0 + math.cos(math.pi * progress)))

    def step(self, optimizer, global_step: int) -> float:
        lr = self.lr_at(global_step)
        optimizer.learning_rate = lr
        return lr

    def state_dict(self) -> dict:
        return {
            "type": "cosine_warmup",
            "base_lr": float(self.base_lr),
            "total_steps": int(self.total_steps),
            "warmup_steps": int(self.warmup_steps),
            "min_lr": float(self.min_lr),
        }

    @classmethod
    def from_state_dict(cls, data: dict) -> "CosineWarmupScheduler":
        if not isinstance(data, dict):
            raise ValueError("scheduler state_dict 必须是 dict")
        kind = data.get("type", "cosine_warmup")
        if kind != "cosine_warmup":
            raise ValueError(f"未知的调度器类型: {kind!r}")
        return cls(
            base_lr=float(data["base_lr"]),
            total_steps=int(data["total_steps"]),
            warmup_steps=int(data.get("warmup_steps", 0)),
            min_lr=float(data.get("min_lr", 0.0)),
        )
