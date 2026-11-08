"""Dynamic loss scaling with a checkpointable growth counter and atomic overflow skips."""

from functools import cache
import math

import numpy as np

from ..core.device import _lazy_cupy, get_device


@cache
def _unscale_kernel():
    return _lazy_cupy().ElementwiseKernel(
        "T g, T inverse",
        "T result, raw int32 overflow",
        "if (!isfinite((double)g)) atomicOr(&overflow[0], 1); result = g * inverse;",
        "mf_unscale_finite",
    )


class DynamicLossScaler:
    def __init__(self, scale=128.0, growth_interval=128, maximum=2.0**24):
        if (
            not 1.0 <= scale <= maximum
            or not math.isfinite(maximum)
            or type(growth_interval) is not int
            or growth_interval < 1
        ):
            raise ValueError("expected positive bounded loss scale and growth interval")
        self.scale, self.maximum = float(scale), float(maximum)
        self.growth_interval, self.good_steps, self.skipped_steps = growth_interval, 0, 0
        self.overflow = False

    def unscale(self, gradients):
        arrays = [g for g in gradients if g is not None]
        if get_device() == "cuda":
            cp = _lazy_cupy()
            flag = cp.zeros(1, cp.int32)
            for g in arrays:
                _unscale_kernel()(g, g.dtype.type(1.0 / self.scale), g, flag)
            self.overflow |= bool(flag.item())
        else:
            for g in arrays:
                self.overflow |= not np.isfinite(g).all()
                g *= g.dtype.type(1.0 / self.scale)
        return not self.overflow

    def finish(self):
        if self.overflow:
            self.scale = max(1.0, self.scale * 0.5)
            self.skipped_steps += 1
            self.good_steps = 0
        else:
            self.good_steps += 1
            if self.good_steps == self.growth_interval:
                self.scale = min(self.maximum, self.scale * 2.0)
                self.good_steps = 0
        self.overflow = False

    def state_dict(self):
        return dict(vars(self))

    @classmethod
    def from_state(cls, state):
        instance = cls(state["scale"], state["growth_interval"], state["maximum"])
        good, skipped, overflow = (state[k] for k in ("good_steps", "skipped_steps", "overflow"))
        if (
            type(good) is not int
            or not 0 <= good < instance.growth_interval
            or type(skipped) is not int
            or skipped < 0
            or type(overflow) is not bool
        ):
            raise ValueError("invalid loss scaler state")
        instance.good_steps, instance.skipped_steps, instance.overflow = good, skipped, overflow
        return instance
