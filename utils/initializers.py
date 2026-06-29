# -*- coding: utf-8 -*-
"""Parameter initializer helpers."""

from __future__ import annotations

from typing import Callable

import numpy as np


Initializer = Callable[[tuple[int, ...]], np.ndarray]


def compute_fans(shape: tuple[int, ...]) -> tuple[int, int]:
    if len(shape) < 2:
        return 1, 1
    if len(shape) == 2:
        return int(shape[0]), int(shape[1])
    receptive = int(np.prod(shape[2:])) if len(shape) > 2 else 1
    fan_in = int(shape[1] * receptive)
    fan_out = int(shape[0] * receptive)
    return max(1, fan_in), max(1, fan_out)


def make_initializer(
    name: str | Callable | None = None,
    *,
    seed: int | None = None,
    value: float = 0.0,
    mean: float = 0.0,
    std: float = 0.01,
) -> Initializer:
    if callable(name):
        return name
    key = str(name or "kaiming_normal").strip().lower()
    rng = np.random.default_rng(seed)

    def init(shape) -> np.ndarray:
        resolved = tuple(int(v) for v in shape)
        fan_in, fan_out = compute_fans(resolved)
        if key in ("default", "he", "he_normal", "kaiming", "kaiming_normal"):
            return rng.normal(0.0, np.sqrt(2.0 / fan_in), size=resolved)
        if key == "kaiming_uniform":
            limit = np.sqrt(6.0 / fan_in)
            return rng.uniform(-limit, limit, size=resolved)
        if key == "xavier_normal":
            return rng.normal(0.0, np.sqrt(2.0 / (fan_in + fan_out)), size=resolved)
        if key == "xavier_uniform":
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            return rng.uniform(-limit, limit, size=resolved)
        if key == "normal":
            return rng.normal(float(mean), float(std), size=resolved)
        if key == "constant":
            return np.full(resolved, float(value), dtype=np.float64)
        raise ValueError(f"unknown initializer: {name!r}")

    return init
