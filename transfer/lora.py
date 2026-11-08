"""Low-rank updates for dense and ordinary Conv2D layers, with permanent inference merging."""

import math
import numpy as np

from ..core.device import xp
from ..core.node import GradientOp, Variable
from ..layers.layer import Conv2D, Dense, Layer
from ..layers.resnet import BatchNorm2d
from ..ops.basic import Linear
from ..ops.convolution import Conv2D_Op
from ..utils.checkpoint import collect_layers


class _Residual(GradientOp):
    def __init__(self, base, update, scale):
        super().__init__(base, update)
        self.scale = scale

    def forward(self, base, update):
        # Preserve the base bit pattern (including signed zero) at zero-initialization.
        self.value = xp.where(update == 0, base, base + self.scale * update)

    def backward(self):
        base, update = self.parents
        if base.requires_grad:
            base.accumulate_grad(self.grad)
        if update.requires_grad:
            update.accumulate_grad(self.scale * self.grad, owned=True)


def _factors(a_shape, b_shape, dtype, fan_in, rank, alpha, seed, name):
    if not isinstance(rank, int) or rank < 1:
        raise ValueError("LoRA rank must be a positive integer")
    alpha = float(rank if alpha is None else alpha)
    if not math.isfinite(alpha):
        raise ValueError("LoRA alpha must be finite")
    rng = np.random.default_rng(seed)
    a = Variable(
        xp.asarray(rng.normal(0.0, fan_in**-0.5, a_shape), dtype=dtype), trainable=True, name=name + "_A"
    )
    b = Variable(xp.zeros(b_shape, dtype=dtype), trainable=True, name=name + "_B")
    return a, b, alpha / rank


class LoRADense(Layer):
    def __init__(self, base, rank, *, bias=None, alpha=None, seed=None, name="lora_dense", backend="auto"):
        super().__init__(name)
        if isinstance(base, Dense):
            self.weight, self.bias, self.activation, self.backend = (
                base.W,
                base.b,
                base.activation,
                base.backend,
            )
        else:
            self.weight, self.bias, self.activation, self.backend = base, bias, None, backend
        if not isinstance(self.weight, Variable) or self.weight.value.ndim != 2 or self.bias is None:
            raise ValueError("expected a Dense layer or a weight matrix and bias variable")
        n, m = self.weight.value.shape
        self.weight.trainable = self.bias.trainable = False
        self.A, self.B, self.scale = _factors(
            (n, rank), (rank, m), self.weight.value.dtype, n, rank, alpha, seed, name
        )
        self.zero_a = Variable(
            xp.zeros(rank, dtype=self.weight.value.dtype), constant=True, requires_grad=False
        )
        self.zero_b = Variable(xp.zeros(m, dtype=self.weight.value.dtype), constant=True, requires_grad=False)
        self.params = [self.weight, self.bias, self.A, self.B]
        self.merged = False

    def forward(self, x):
        value = Linear(x, self.weight, self.bias, backend=self.backend)
        if not self.merged:
            low = Linear(x, self.A, self.zero_a, backend=self.backend)
            value = _Residual(value, Linear(low, self.B, self.zero_b, backend=self.backend), self.scale)
        return self.activation(value) if self.activation else value

    def merge(self):
        if not self.merged:
            self.weight.value += self.scale * (self.A.value @ self.B.value)
            self.B.value.fill(0)
            self.A.trainable = self.B.trainable = False
            self.merged = True
        return self


class LoRAConv2d(Layer):
    def __init__(self, base, rank, *, alpha=None, seed=None, name="lora_conv"):
        super().__init__(name)
        if not isinstance(base, Conv2D) or base.groups != 1:
            raise ValueError("LoRAConv2d requires an ordinary groups=1 Conv2D")
        self.base, self.kernel, self.b = base, base.kernel, base.b
        out_c, in_c, kh, kw = self.kernel.value.shape
        for p in base.params:
            p.trainable = False
        self.A, self.B, self.scale = _factors(
            (rank, in_c, kh, kw),
            (out_c, rank, 1, 1),
            self.kernel.value.dtype,
            in_c * kh * kw,
            rank,
            alpha,
            seed,
            name,
        )
        self.params = [*base.params, self.A, self.B]
        self.merged = False

    def forward(self, x):
        if self.merged:
            return self.base(x)
        base = self.base
        common = {
            "stride": base.stride,
            "padding": base.padding,
            "dilation": base.dilation,
            "backend": base.backend,
            "recompute_im2col": base.recompute_im2col,
        }
        value = Conv2D_Op(x, self.kernel, bias=self.b, **common)
        low = Conv2D_Op(x, self.A, **common)
        high = Conv2D_Op(low, self.B, backend=base.backend, recompute_im2col=base.recompute_im2col)
        value = _Residual(value, high, self.scale)
        return base.activation(value) if base.activation else value

    def merge(self):
        if not self.merged:
            delta = self.B.value[:, :, 0, 0] @ self.A.value.reshape(self.A.value.shape[0], -1)
            self.kernel.value += self.scale * delta.reshape(self.kernel.value.shape)
            self.B.value.fill(0)
            self.A.trainable = self.B.trainable = False
            self.merged = True
        return self.base


def attach_resnet_lora(model, rank, *, alpha=None, seed=0):
    """Freeze the source network, then attach adapters to layer4 convolutions and the output head."""
    if model.arch != "resnet18" or hasattr(model, "transfer_head"):
        raise ValueError("this LoRA placement expects a ResNet18 with its original linear head")
    for p in model.params:
        p.trainable = False
    for layer in collect_layers([model]):
        if isinstance(layer, BatchNorm2d):
            layer.freeze(True)
    swaps = []
    targets = []
    for index, block in enumerate(model.layer4):
        for attr in ("conv1", "conv2", "downsample_conv"):
            base = getattr(block, attr, None)
            if base is None:
                continue
            name = f"layer4.{index}.{attr}"
            wrapped = LoRAConv2d(
                base, rank, alpha=alpha, seed=None if seed is None else seed + len(swaps), name=name
            )
            setattr(block, attr, wrapped)
            block.sub_layers = [wrapped if layer is base else layer for layer in block.sub_layers]
            model.sub_layers = [wrapped if layer is base else layer for layer in model.sub_layers]
            block.params.extend((wrapped.A, wrapped.B))
            model.params.extend((wrapped.A, wrapped.B))
            swaps.append((block, attr, wrapped))
            targets.append(name)
    model.transfer_head = LoRADense(
        model.fc_W,
        rank,
        bias=model.fc_b,
        alpha=alpha,
        seed=None if seed is None else seed + len(swaps),
        name="fc",
    )
    model.params.extend((model.transfer_head.A, model.transfer_head.B))
    model._lora_swaps = swaps
    targets.append("fc")
    return {
        "rank": rank,
        "alpha": rank if alpha is None else alpha,
        "targets": targets,
        "trainable_values": sum(int(p.value.size) for p in model.params if p.trainable),
    }


def merge_resnet_lora(model):
    if not hasattr(model, "_lora_swaps"):
        return model
    factors = set()
    for block, attr, wrapped in model._lora_swaps:
        base = wrapped.merge()
        setattr(block, attr, base)
        block.sub_layers = [base if layer is wrapped else layer for layer in block.sub_layers]
        model.sub_layers = [base if layer is wrapped else layer for layer in model.sub_layers]
        factors.update((wrapped.A, wrapped.B))
        block.params = [p for p in block.params if p not in factors]
    head = model.transfer_head.merge()
    factors.update((head.A, head.B))
    model.params = [p for p in model.params if p not in factors]
    del model._lora_swaps, model.transfer_head
    return model
